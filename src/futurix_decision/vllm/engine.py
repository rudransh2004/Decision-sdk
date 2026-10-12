"""vLLM single-pass decision readout using targeted decision-token logprobs."""

from __future__ import annotations

import math
import time
from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
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


@dataclass(frozen=True)
class _Prepared:
    """One example's vLLM prompts and how to read their logprobs back."""

    rows: list[tuple[list[int], list[int]]]
    """Each prompt's token IDs and the label token IDs to read, in presented order."""
    order: list[int] | None = None
    """Original option index at each presented position; ``None`` when already original."""


def _logprob(value: Any) -> float:
    return float(getattr(value, "logprob", value))


def _restore_order(probabilities: list[float], order: list[int] | None) -> list[float]:
    if order is None:
        return probabilities
    ordered = [0.0] * len(probabilities)
    for position, original in enumerate(order):
        ordered[original] = probabilities[position]
    return ordered


class _PromptFormat(ABC):
    """How one ``prompt_format`` builds vLLM prompts and turns logprobs into probabilities."""

    def __init__(self, engine: VLLMEngine) -> None:
        self.engine = engine

    @abstractmethod
    def prepare(self, example: DecisionInput) -> _Prepared: ...

    def probabilities(
        self, example: DecisionInput, prepared: _Prepared, logprobs: list[Mapping[int, Any]]
    ) -> list[float]:
        """Option-order probabilities from each prompt's first-step logprobs."""
        (labels,) = [labels for _, labels in prepared.rows]
        probabilities = probabilities_from_logprobs(
            logprobs[0], labels, temperature=self.temperature(example)
        )
        return _restore_order(probabilities, prepared.order)

    def temperature(self, example: DecisionInput) -> float:
        return self.engine.temperature


class _TokenFormat(_PromptFormat):
    """Futurix and Bosun v3: added decision-token rows, listed in candidate order."""

    def __init__(self, engine: VLLMEngine) -> None:
        super().__init__(engine)
        tokenizer = engine.tokenizer
        self.token_ids = [
            tokenizer.convert_tokens_to_ids(decision_token(i))
            for i in range(engine.decision_tokens)
        ]
        if any(
            tokenizer.encode(decision_token(i), add_special_tokens=False) != [token_id]
            for i, token_id in enumerate(self.token_ids)
        ):
            raise CheckpointError("checkpoint is missing single-token decision slots")

    def prepare(self, example: DecisionInput) -> _Prepared:
        engine = self.engine
        if engine.prompt_format == "bosun-v3":
            prompt, slots = format_bosun_prompt(
                example,
                engine.tokenizer,
                decision_tokens=engine.decision_tokens,
                seed=engine.seed,
                prompt_schema=engine.prompt_schema,
            )
        else:
            prompt, slots = format_decision_prompt(
                example, decision_tokens=engine.decision_tokens, seed=engine.seed
            )
        ids = engine.tokenizer.encode(prompt, add_special_tokens=True)
        return _Prepared([(ids, [self.token_ids[slot] for slot in slots])])


class _LabelFormat(_PromptFormat):
    """JPT and Eikos: one prompt whose existing label tokens list the options."""

    def __init__(self, engine: VLLMEngine) -> None:
        super().__init__(engine)
        if engine.prompt_format == EIKOS_FORMAT:
            eikos_label_table(engine.tokenizer, engine.decision_tokens)  # Fails early if absent.

    def prepare(self, example: DecisionInput) -> _Prepared:
        engine = self.engine
        prepare = prepare_jpt_row if engine.prompt_format == JPT_FORMAT else prepare_eikos_row
        tokens, labels, order = prepare(
            example, engine.tokenizer, max_options=engine.decision_tokens
        )
        return _Prepared([(tokens, labels)], order)


class _DeciderFormat(_PromptFormat):
    """Decider v2.1: per-kind temperatures, and one yes/no prompt per score level."""

    def __init__(self, engine: VLLMEngine, metadata: Mapping[str, Any]) -> None:
        super().__init__(engine)
        if "type_temperatures" not in metadata:
            raise CheckpointError("decider-v2.1 export is missing type-specific calibration")
        self.type_temperatures = metadata["type_temperatures"]
        if set(self.type_temperatures) != {"choice", "noul", "score"} or any(
            not isinstance(value, (float, int)) or not math.isfinite(value) or value <= 0
            for value in self.type_temperatures.values()
        ):
            raise CheckpointError("decider-v2.1 export has invalid type-specific calibration")
        letter_table(engine.tokenizer, engine.decision_tokens)  # Fails early if absent.

    def prepare(self, example: DecisionInput) -> _Prepared:
        rows = prepare_decider_rows(
            example, self.engine.tokenizer, decision_tokens=self.engine.decision_tokens
        )
        order = (
            None
            if example.kind == "score"
            else presented_order(example, DECIDER_NOUL_ORDER, DECIDER_FORMAT)
        )
        return _Prepared([(row.input_ids, row.label_ids) for row in rows], order)

    def temperature(self, example: DecisionInput) -> float:
        if example.kind == "score":
            return float(self.type_temperatures["score"])
        return float(self.type_temperatures[example.kind]) * self.engine.temperature

    def probabilities(
        self, example: DecisionInput, prepared: _Prepared, logprobs: list[Mapping[int, Any]]
    ) -> list[float]:
        if example.kind != "score":
            return super().probabilities(example, prepared, logprobs)
        # Each level's prompt asks whether that level fits; label 1 is "yes".
        temperature = self.temperature(example)
        fits = []
        for (_, labels), step in zip(prepared.rows, logprobs, strict=True):
            probabilities_from_logprobs(step, labels, temperature=temperature)  # Validates.
            scores = [_logprob(step[token_id]) / temperature for token_id in labels]
            peak = max(scores)
            fits.append(
                scores[1] - peak - math.log(sum(math.exp(value - peak) for value in scores))
            )
        highest = max(fits)
        weights = [math.exp((value - highest) / self.engine.temperature) for value in fits]
        total = sum(weights)
        return [weight / total for weight in weights]


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
        self._format: _PromptFormat
        if self.prompt_format == DECIDER_FORMAT:
            self._format = _DeciderFormat(self, metadata)
        elif self.prompt_format in (JPT_FORMAT, EIKOS_FORMAT):
            self._format = _LabelFormat(self)
        else:
            self._format = _TokenFormat(self)
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

    def prompt_tokens(self, example: DecisionInput) -> int:
        """The longest prompt ``example`` sends to vLLM, in tokens."""
        return max(len(ids) for ids, _ in self._format.prepare(example).rows)

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
        prepared = [self._format.prepare(example) for example in examples]
        prompts, params = [], []
        for ids, labels in (row for item in prepared for row in item.rows):
            if len(ids) > self.max_sequence_length:
                raise ConfigError("prompt exceeds max_sequence_length")
            prompts.append({"prompt_token_ids": ids})
            params.append(
                SamplingParams(
                    max_tokens=1,
                    temperature=0.0,
                    detokenize=False,
                    ignore_eos=True,
                    logprobs=len(labels),  # After allowed-token masking.
                    allowed_token_ids=labels,
                )
            )
        started = time.perf_counter()
        output = self.llm.generate(prompts, params, use_tqdm=False)
        # Report the batch's wall time per row, as every backend does.
        latency_ms = (time.perf_counter() - started) * 1000 / len(examples)
        if len(output) != len(prompts):
            raise InvalidOutputError("vLLM returned a different number of outputs")
        # Outputs come back in prompt order; each example takes its own prompts' outputs.
        steps = iter(output)
        results = []
        for example, item in zip(examples, prepared, strict=True):
            logprobs = [_first_step(next(steps)) for _ in item.rows]
            results.append(
                DecisionResult.from_probabilities(
                    example,
                    self._format.probabilities(example, item, logprobs),
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


def _first_step(generated: Any) -> Mapping[int, Any]:
    token_logprobs = generated.outputs[0].logprobs
    if not token_logprobs:
        raise InvalidOutputError("vLLM returned no decision logprobs")
    return token_logprobs[0]
