"""vLLM single-pass decision readout using targeted decision-token logprobs."""

from __future__ import annotations

import math
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from transformers import AutoTokenizer

from futurix_decision.datasets.schema import DecisionInput, State
from futurix_decision.errors import (
    CheckpointError,
    ConfigError,
    InvalidOutputError,
    MissingDependencyError,
)
from futurix_decision.inference.schemas import (
    DecisionResult,
    check_confidence_threshold,
    choice_input,
    noul_input,
    score_input,
)
from futurix_decision.models.bosun import BOSUN_PROMPT_SCHEMA, format_bosun_prompt
from futurix_decision.models.decider import FORMAT as DECIDER_FORMAT
from futurix_decision.models.decider import NOUL_ORDER as DECIDER_NOUL_ORDER
from futurix_decision.models.decider import letter_table, prepare_decider_rows
from futurix_decision.models.decision_head import decision_token
from futurix_decision.models.eikos import FORMAT as EIKOS_FORMAT
from futurix_decision.models.eikos import eikos_label_table, prepare_eikos_row
from futurix_decision.models.jpt import FORMAT as JPT_FORMAT
from futurix_decision.models.jpt import prepare_jpt_row
from futurix_decision.models.label_readout import presented_order
from futurix_decision.models.tokenizer import format_decision_prompt
from futurix_decision.utils.config import load_decision_config
from futurix_decision.vllm.logits_processor import probabilities_from_logprobs


class VLLMEngine:
    """Batch prompts; request exactly one restricted token step, never freeform text.

    vLLM's public API requires max_tokens>=1. Restrict processed logits to
    candidate IDs and request every candidate's top-k logprob after one
    prompt forward. The sampled token is ignored; no text is generated.
    """

    def __init__(
        self,
        checkpoint: str | Path,
        *,
        llm: Any | None = None,
        tokenizer: Any | None = None,
        **vllm_kwargs: Any,
    ) -> None:
        metadata = load_decision_config(checkpoint)
        if metadata.get("architecture") != "futurix-decision":
            raise CheckpointError("vLLM requires an exported Futurix decision_config.json")
        if metadata.get("head_type", "tokens") != "tokens":
            raise ConfigError(
                "vLLM token-logprob backend does not support custom decision heads; "
                "use the PyTorch inference backend"
            )
        self.decision_tokens = int(metadata["decision_tokens"])
        self.temperature = float(metadata["temperature"])
        self.seed = int(metadata.get("seed", 0))
        self.max_sequence_length = int(metadata.get("max_sequence_length", 1024))
        self.prompt_format = metadata.get("prompt_format", "futurix")
        self.prompt_schema = metadata.get("prompt_schema", BOSUN_PROMPT_SCHEMA)
        if (
            not 2 <= self.decision_tokens <= 256
            or not math.isfinite(self.temperature)
            or self.temperature <= 0
        ):
            raise CheckpointError("invalid decision token count or calibration temperature")
        self.tokenizer = tokenizer or AutoTokenizer.from_pretrained(str(checkpoint))
        if self.prompt_format == DECIDER_FORMAT:
            if "type_temperatures" not in metadata:
                raise CheckpointError("decider-v2.1 export is missing type-specific calibration")
            self.type_temperatures = metadata["type_temperatures"]
            if set(self.type_temperatures) != {"choice", "noul", "score"} or any(
                not isinstance(value, (float, int)) or not math.isfinite(value) or value <= 0
                for value in self.type_temperatures.values()
            ):
                raise CheckpointError("decider-v2.1 export has invalid type-specific calibration")
            _, self._token_ids = letter_table(self.tokenizer, self.decision_tokens)
        elif self.prompt_format == JPT_FORMAT:
            self._token_ids = []
        elif self.prompt_format == EIKOS_FORMAT:
            _, self._token_ids = eikos_label_table(self.tokenizer, self.decision_tokens)
        else:
            self._token_ids = [
                self.tokenizer.convert_tokens_to_ids(decision_token(i))
                for i in range(self.decision_tokens)
            ]
            if any(
                self.tokenizer.encode(decision_token(i), add_special_tokens=False) != [token_id]
                for i, token_id in enumerate(self._token_ids)
            ):
                raise CheckpointError("checkpoint is missing single-token decision slots")
        if llm is None:
            try:
                from vllm import LLM
            except ImportError as exc:
                raise MissingDependencyError(
                    "vLLM serving requires pip install futurix-decision[vllm]"
                ) from exc
            # Current GPU workers compute top-k logprobs and do not honor
            # logprob_token_ids. Restricted processed logits make top-k cover
            # *every* candidate without transferring the full LM vocabulary.
            vllm_kwargs.setdefault("max_logprobs", self.decision_tokens)
            vllm_kwargs.setdefault("logprobs_mode", "processed_logprobs")
            llm = LLM(model=str(checkpoint), **vllm_kwargs)
        self.llm = llm

    def batch_decide(
        self, examples: Sequence[DecisionInput], *, confidence_threshold: float | None = None
    ) -> list[DecisionResult]:
        check_confidence_threshold(confidence_threshold)
        if not examples:
            return []
        try:
            from vllm import SamplingParams
        except ImportError as exc:
            raise MissingDependencyError(
                "vLLM serving requires pip install futurix-decision[vllm]"
            ) from exc
        prompts, params, slots_list, group_sizes, orders = [], [], [], [], []
        for example in examples:
            if self.prompt_format == DECIDER_FORMAT:
                rows = prepare_decider_rows(
                    example, self.tokenizer, decision_tokens=self.decision_tokens
                )
                prepared = [(row.input_ids, row.label_ids) for row in rows]
                orders.append(presented_order(example, DECIDER_NOUL_ORDER, DECIDER_FORMAT))
            elif self.prompt_format == JPT_FORMAT:
                tokens, labels, order = prepare_jpt_row(
                    example, self.tokenizer, max_options=self.decision_tokens
                )
                prepared = [(tokens, labels)]
                orders.append(order)
            elif self.prompt_format == EIKOS_FORMAT:
                tokens, labels, order = prepare_eikos_row(
                    example, self.tokenizer, max_options=self.decision_tokens
                )
                prepared = [(tokens, labels)]
                orders.append(order)
            else:
                if self.prompt_format == "bosun-v3":
                    prompt, slots = format_bosun_prompt(
                        example,
                        self.tokenizer,
                        decision_tokens=self.decision_tokens,
                        seed=self.seed,
                        prompt_schema=self.prompt_schema,
                    )
                else:
                    prompt, slots = format_decision_prompt(
                        example, decision_tokens=self.decision_tokens, seed=self.seed
                    )
                prepared = [
                    (
                        self.tokenizer.encode(prompt, add_special_tokens=True),
                        [self._token_ids[slot] for slot in slots],
                    )
                ]
                orders.append(list(range(len(example.options))))
            group_sizes.append(len(prepared))
            for ids, selected_ids in prepared:
                if len(ids) > self.max_sequence_length:
                    raise ConfigError("prompt exceeds max_sequence_length")
                prompts.append({"prompt_token_ids": ids})
                params.append(
                    SamplingParams(
                        max_tokens=1,
                        temperature=0.0,
                        detokenize=False,
                        ignore_eos=True,
                        logprobs=len(selected_ids),  # After allowed-token masking.
                        allowed_token_ids=selected_ids,
                    )
                )
                slots_list.append(selected_ids)
        started = time.perf_counter()
        output = self.llm.generate(prompts, params, use_tqdm=False)
        # Report the batch's wall time per row, as every backend does.
        latency_ms = (time.perf_counter() - started) * 1000 / len(examples)
        if len(output) != len(prompts):
            raise InvalidOutputError("vLLM returned a different number of outputs")
        results = []
        cursor = 0
        for row, count in zip(examples, group_sizes):
            row_probabilities = []
            score_log_fits = []
            for selected_ids, generated in zip(
                slots_list[cursor : cursor + count], output[cursor : cursor + count]
            ):
                token_logprobs = generated.outputs[0].logprobs
                if not token_logprobs:
                    raise InvalidOutputError("vLLM returned no decision logprobs")
                temp = (
                    self.type_temperatures[row.kind]
                    * (self.temperature if row.kind != "score" else 1.0)
                    if self.prompt_format == DECIDER_FORMAT
                    else self.temperature
                )
                row_probabilities.append(
                    probabilities_from_logprobs(token_logprobs[0], selected_ids, temperature=temp)
                )
                if self.prompt_format == DECIDER_FORMAT and row.kind == "score":
                    scores = [
                        float(
                            getattr(
                                token_logprobs[0][token_id], "logprob", token_logprobs[0][token_id]
                            )
                        )
                        / temp
                        for token_id in selected_ids
                    ]
                    peak = max(scores)
                    score_log_fits.append(
                        scores[1] - peak - math.log(sum(math.exp(value - peak) for value in scores))
                    )
            cursor += count
            if self.prompt_format == DECIDER_FORMAT and row.kind == "score":
                highest = max(score_log_fits)
                fits = [math.exp((value - highest) / self.temperature) for value in score_log_fits]
                total = sum(fits)
                probabilities = [fit / total for fit in fits]
            else:
                probabilities = row_probabilities[0]
            if self.prompt_format in (JPT_FORMAT, EIKOS_FORMAT) or (
                self.prompt_format == DECIDER_FORMAT and row.kind != "score"
            ):
                ordered = [0.0] * len(probabilities)
                for position, original in enumerate(orders[len(results)]):
                    ordered[original] = probabilities[position]
                probabilities = ordered
            results.append(
                DecisionResult.from_probabilities(
                    row,
                    probabilities,
                    latency_ms=latency_ms,
                    confidence_threshold=confidence_threshold,
                )
            )
        return results

    def decide(
        self, example: DecisionInput, *, confidence_threshold: float | None = None
    ) -> DecisionResult:
        return self.batch_decide([example], confidence_threshold=confidence_threshold)[0]

    def choice(
        self,
        *,
        state: State,
        choices: Sequence[str],
        question: str = "Which option should be selected?",
        confidence_threshold: float | None = None,
    ) -> DecisionResult:
        return self.decide(
            choice_input(state, choices, question), confidence_threshold=confidence_threshold
        )

    def noul(
        self,
        *,
        state: State,
        question: str = "Is this true?",
        confidence_threshold: float | None = None,
    ) -> DecisionResult:
        return self.decide(noul_input(state, question), confidence_threshold=confidence_threshold)

    def score(
        self,
        *,
        state: State,
        question: str = "What score should be assigned?",
        min_score: int = 0,
        max_score: int = 5,
        confidence_threshold: float | None = None,
    ) -> DecisionResult:
        return self.decide(
            score_input(state, question, min_score, max_score),
            confidence_threshold=confidence_threshold,
        )

    def close(self) -> None:
        """Do nothing; part of ``DecisionBackend``. Delete the engine to free GPU memory."""
