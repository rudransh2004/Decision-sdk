"""The Mapika/decider-4b v2.1 text decision contract.

This is *not* a new decision-token model. It retains the released LM weights,
scores the pretrained A..Z/AA.. letter rows, and uses the upstream plain
state-first prompt and isolated score-level readout. Text/JSON states only.
"""

from __future__ import annotations

import hashlib
import json
import math
import random
import re
import string
import weakref
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from huggingface_hub import hf_hub_download
from transformers import AutoModelForCausalLM, AutoTokenizer

from futurix_decision.datasets.schema import DecisionInput
from futurix_decision.models.decision_model import DecisionBatch, DecisionModel

FORMAT = "decider-v2.1"
MAX_OPTIONS = 255
MAX_SCORE_LEVELS = 10
DEFAULT_TEMPERATURES = {"choice": 1.110, "noul": 1.560, "score": 1.287}
_LABEL_CACHE: weakref.WeakKeyDictionary[Any, tuple[list[str], list[int]]] = (
    weakref.WeakKeyDictionary()
)


@dataclass(frozen=True)
class DeciderRow:
    input_ids: list[int]
    letter_ids: list[int]


def letter_table(tokenizer: Any, width: int = MAX_OPTIONS) -> tuple[list[str], list[int]]:
    """Map labels to *existing* single-token vocabulary rows; never add tokens."""
    cached = _LABEL_CACHE.get(tokenizer)
    if cached is not None and len(cached[1]) >= width:
        return cached[0][:width], cached[1][:width]
    names: list[str] = []
    ids: list[int] = []
    candidates = list(string.ascii_uppercase) + [
        a + b for a in string.ascii_uppercase for b in string.ascii_uppercase
    ]
    for label in candidates:
        encoded = tokenizer.encode(label, add_special_tokens=False)
        if len(encoded) == 1 and encoded[0] not in ids:
            names.append(label)
            ids.append(encoded[0])
        if len(ids) == width:
            break
    if len(ids) != width:
        raise ValueError(
            f"decider-v2.1 needs {width} distinct existing letter tokens; "
            f"tokenizer only provides {len(ids)}"
        )
    if names[: min(width, 10)] != list("ABCDEFGHIJ")[: min(width, 10)]:
        raise ValueError("decider-v2.1's trained A-J option labels are unavailable")
    _LABEL_CACHE[tokenizer] = names, ids
    return names, ids


def _render_state(state: Any) -> str:
    if isinstance(state, str):
        return state

    def annotate(value: Any) -> Any:
        if isinstance(value, list):
            if len(value) >= 8:
                return [
                    {"_index": index, **annotate(row)}
                    if isinstance(row, dict)
                    else {"_index": index, "value": annotate(row)}
                    for index, row in enumerate(value)
                ]
            return [annotate(row) for row in value]
        if isinstance(value, dict):
            return {key: annotate(row) for key, row in value.items()}
        return value

    return json.dumps(annotate(state), ensure_ascii=False)


def _option_text(example: DecisionInput, position: int) -> str:
    if example.kind == "noul":
        if example.options[position] not in ("true", "false"):
            raise ValueError("decider-v2.1 noul requires literal true/false options")
        name = "yes" if example.options[position] == "true" else "no"
        description = (
            example.candidate_descriptions[position] if example.candidate_descriptions else None
        ) or (example.candidate_labels[position] if example.candidate_labels else None)
        return f"{name}: {description}" if description else name
    name = (
        example.candidate_labels[position]
        if example.candidate_labels
        else example.options[position]
    )
    description = (
        example.candidate_descriptions[position] if example.candidate_descriptions else None
    )
    return f"{name}: {description}" if description else name


def prepare_decider_rows(
    example: DecisionInput,
    tokenizer: Any,
    *,
    decision_tokens: int = MAX_OPTIONS,
    max_state_tokens: int = 32768,
    option_order: Sequence[int] | None = None,
) -> list[DeciderRow]:
    """Tokenize parts independently as the released decider.prompt.build does."""
    if len(example.options) > decision_tokens:
        raise ValueError(f"decider-v2.1 supports at most {decision_tokens} choices")
    names, token_ids = letter_table(tokenizer, decision_tokens)
    context = tokenizer.encode(
        "Context:\n" + _render_state(example.state), add_special_tokens=False
    )
    if len(context) > max_state_tokens:
        raise ValueError("state exceeds the decider-v2.1 maximum state length")
    if example.kind == "score" and len(example.options) > MAX_SCORE_LEVELS:
        raise ValueError("decider-v2.1 isolated score supports at most 10 levels")
    if option_order is None:
        option_order = (
            [example.options.index("false"), example.options.index("true")]
            if example.kind == "noul" and set(example.options) == {"false", "true"}
            else list(range(len(example.options)))
        )
    if sorted(option_order) != list(range(len(example.options))):
        raise ValueError("option_order must be a permutation of the original options")
    specs: list[tuple[str, list[str]]] = []
    if example.kind == "score":
        for index, level in enumerate(example.options):
            text = (
                example.candidate_descriptions[index]
                if example.candidate_descriptions and example.candidate_descriptions[index]
                else example.candidate_labels[index]
                if example.candidate_labels
                else level
            )
            text = re.sub(r"^\s*-?\d+\s*:\s*", "", text)
            specs.append(
                (
                    f"{example.question}\nProposed answer: {text}\nDoes the proposed answer fit?",
                    ["no", "yes"],
                )
            )
    else:
        specs.append(
            (
                example.question,
                [_option_text(example, index) for index in option_order],
            )
        )

    rows = []
    for question, options in specs:
        ids = context + tokenizer.encode(
            f"\n\nQuestion: {question}\nOptions:", add_special_tokens=False
        )
        if len(options) <= 10:
            ids += tokenizer.encode(
                "".join(f"\n({names[i]}) {option}" for i, option in enumerate(options)),
                add_special_tokens=False,
            )
        else:
            for i, option in enumerate(options):
                ids += tokenizer.encode("\n(", add_special_tokens=False)
                ids.append(token_ids[i])
                ids += tokenizer.encode(f") {option}", add_special_tokens=False)
        ids += tokenizer.encode("\nAnswer: (", add_special_tokens=False)
        rows.append(DeciderRow(ids, token_ids[: len(options)]))
    return rows


def _native_config(checkpoint: str | Path, **hub_kwargs: Any) -> dict[str, Any]:
    path = Path(checkpoint)
    filename = (
        path / "decider_config.json"
        if path.is_dir()
        else Path(
            hf_hub_download(
                repo_id=str(checkpoint),
                filename="decider_config.json",
                **{
                    key: hub_kwargs[key]
                    for key in ("revision", "token", "cache_dir", "local_files_only")
                    if key in hub_kwargs
                },
            )
        )
    )
    config = json.loads(filename.read_text(encoding="utf-8"))
    if (
        config.get("version") != "4b-v2.1"
        or config.get("layout") != "plain"
        or config.get("schema_first") is not False
        or config.get("isolated_levels") is not True
        or config.get("max_options") != MAX_OPTIONS
    ):
        raise ValueError(
            "only the Mapika decider-4b v2.1 plain, isolated-score contract is supported"
        )
    return config


class DeciderDecisionModel(DecisionModel):
    """Trainable SDK wrapper over Mapika's original letter head and prompt."""

    def __init__(
        self,
        backbone: torch.nn.Module,
        tokenizer: Any,
        *,
        type_temperatures: Mapping[str, float] | None = None,
        **kwargs: Any,
    ) -> None:
        if kwargs.get("prompt_format", FORMAT) != FORMAT:
            raise ValueError("decider-v2.1 wrapper requires its native prompt format")
        if kwargs.get("head_type", "tokens") != "tokens":
            raise ValueError("decider-v2.1 uses its trained LM letter readout, not a new head")
        kwargs["prompt_format"] = FORMAT
        kwargs.setdefault("decision_tokens", MAX_OPTIONS)
        self.type_temperatures = dict(type_temperatures or DEFAULT_TEMPERATURES)
        if set(self.type_temperatures) != {"choice", "noul", "score"} or any(
            not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0
            for value in self.type_temperatures.values()
        ):
            raise ValueError(
                "decider-v2.1 requires positive finite temperatures for each decision kind"
            )
        super().__init__(backbone, tokenizer, **kwargs)

    def _register_decision_tokens(self) -> list[int]:
        if self.tokenizer.pad_token is None:
            if self.tokenizer.eos_token is None:
                raise ValueError("decider-v2.1 tokenizer requires EOS or pad token")
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"
        _, ids = letter_table(self.tokenizer, self.decision_tokens)
        if max(ids) >= self.backbone.get_input_embeddings().num_embeddings:
            raise ValueError("decider tokenizer has letter tokens outside the backbone vocabulary")
        return ids

    @classmethod
    def from_native(
        cls,
        checkpoint: str | Path,
        *,
        max_sequence_length: int = 32768,
        seed: int = 0,
        **hf_kwargs: Any,
    ) -> DeciderDecisionModel:
        if str(checkpoint) == "Mapika/decider-4b" or (
            Path(checkpoint).is_dir() and (Path(checkpoint) / "decider_config.json").is_file()
        ):
            config = _native_config(checkpoint, **hf_kwargs)
        else:
            raise ValueError(
                "only Mapika/decider-4b or a local copy of its v2.1 release is supported"
            )
        tok_kwargs = {
            key: hf_kwargs[key]
            for key in ("revision", "token", "trust_remote_code", "cache_dir", "local_files_only")
            if key in hf_kwargs
        }
        tokenizer = AutoTokenizer.from_pretrained(str(checkpoint), **tok_kwargs)
        backbone = AutoModelForCausalLM.from_pretrained(str(checkpoint), **hf_kwargs)
        return cls(
            backbone,
            tokenizer,
            decision_tokens=MAX_OPTIONS,
            max_sequence_length=max_sequence_length,
            seed=seed,
            base_checkpoint=str(checkpoint),
            base_model_revision=hf_kwargs.get("revision"),
            source_checkpoint=str(checkpoint),
            source_revision=hf_kwargs.get("revision"),
            type_temperatures=config["temperature_by_type"],
        )

    def forward_decisions(
        self, examples: Sequence[DecisionInput], *, assignment_seed: int | None = None
    ) -> DecisionBatch:
        if not examples:
            raise ValueError("examples must not be empty")
        orders = []
        groups = []
        for example in examples:
            order = (
                [example.options.index("false"), example.options.index("true")]
                if example.kind == "noul" and set(example.options) == {"false", "true"}
                else list(range(len(example.options)))
            )
            if assignment_seed is not None and example.kind != "score":
                identity = json.dumps(
                    [
                        self.seed,
                        assignment_seed,
                        example.kind,
                        example.state,
                        example.question,
                        example.options,
                    ],
                    sort_keys=True,
                    ensure_ascii=False,
                )
                row_seed = int.from_bytes(
                    hashlib.sha256(identity.encode("utf-8")).digest()[:8], "big"
                )
                random.Random(row_seed).shuffle(order)
            orders.append(order)
            groups.append(
                prepare_decider_rows(
                    example,
                    self.tokenizer,
                    decision_tokens=self.decision_tokens,
                    option_order=order,
                )
            )
        flat = [row for group in groups for row in group]
        longest = max(len(row.input_ids) for row in flat)
        if longest > self.max_sequence_length:
            raise ValueError(
                f"prompt exceeds max_sequence_length={self.max_sequence_length}; "
                "increase the limit instead of silently truncating candidates"
            )
        device = next(self.backbone.parameters()).device
        pad_id = self.tokenizer.pad_token_id
        ids = torch.full((len(flat), longest), pad_id, dtype=torch.long, device=device)
        mask = torch.zeros_like(ids)
        for index, row in enumerate(flat):
            length = len(row.input_ids)
            ids[index, -length:] = torch.tensor(row.input_ids, device=device)
            mask[index, -length:] = 1
        output = self.backbone(input_ids=ids, attention_mask=mask, logits_to_keep=1)
        logits = output.logits[:, -1, :]
        width = max(len(example.options) for example in examples)
        results = logits.new_full((len(examples), width), float("-inf"))
        cursor = 0
        for index, (example, group) in enumerate(zip(examples, groups)):
            temperature = self.type_temperatures[example.kind]
            if example.kind == "score":
                fits = []
                for row in group:
                    selected = logits[cursor].index_select(
                        0, torch.tensor(row.letter_ids, device=device)
                    )
                    fits.append(torch.log_softmax(selected.float() / temperature, dim=0)[1])
                    cursor += 1
                results[index, : len(group)] = torch.stack(fits)
            else:
                row = group[0]
                selected = logits[cursor].index_select(
                    0, torch.tensor(row.letter_ids, device=device)
                )
                results[index, orders[index]] = selected.float() / temperature
                cursor += 1
        return DecisionBatch(results, [list(range(len(example.options))) for example in examples])

    def _save_decision_metadata(self, path: str | Path, *, adapter_base: str | None = None) -> None:
        super()._save_decision_metadata(path, adapter_base=adapter_base)
        filename = Path(path) / "decision_config.json"
        config = json.loads(filename.read_text(encoding="utf-8"))
        config["type_temperatures"] = self.type_temperatures
        filename.write_text(json.dumps(config, indent=2, allow_nan=False) + "\n", encoding="utf-8")
