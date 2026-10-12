"""The Mapika/decider-4b v2.1 text decision contract.

This is *not* a new decision-token model. It retains the released LM weights,
scores the pretrained A..Z/AA.. letter rows, and uses the upstream plain
state-first prompt and isolated score-level readout. Text/JSON states only.
"""

from __future__ import annotations

import json
import math
import re
import string
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import torch
from huggingface_hub import hf_hub_download

from futurix_decision.datasets.schema import DecisionInput
from futurix_decision.errors import CheckpointError, ConfigError
from futurix_decision.models.label_readout import (
    LabelReadoutModel,
    LabelRow,
    cached_label_table,
    hub_kwargs,
    noul_description,
    presented_order,
    score_description,
)

FORMAT = "decider-v2.1"
MAX_OPTIONS = 255
MAX_SCORE_LEVELS = 10
DEFAULT_TEMPERATURES = {"choice": 1.110, "noul": 1.560, "score": 1.287}
# The released prompt lists "no" before "yes", as the checkpoint was trained.
NOUL_ORDER = ("false", "true")


def letter_table(tokenizer: Any, width: int = MAX_OPTIONS) -> tuple[list[str], list[int]]:
    """Map labels to *existing* single-token vocabulary rows; never add tokens."""
    return cached_label_table(tokenizer, FORMAT, width, _build_letter_table)


def _build_letter_table(tokenizer: Any, width: int) -> tuple[list[str], list[int]]:
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
        raise CheckpointError(
            f"decider-v2.1 needs {width} distinct existing letter tokens; "
            f"tokenizer only provides {len(ids)}"
        )
    if names[: min(width, 10)] != list("ABCDEFGHIJ")[: min(width, 10)]:
        raise CheckpointError("decider-v2.1's trained A-J option labels are unavailable")
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
            raise ConfigError("decider-v2.1 noul requires literal true/false options")
        name = "yes" if example.options[position] == "true" else "no"
        description = noul_description(example, position, None)
    else:
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
) -> list[LabelRow]:
    """Tokenize parts independently as the released decider.prompt.build does."""
    if len(example.options) > decision_tokens:
        raise ConfigError(f"decider-v2.1 supports at most {decision_tokens} choices")
    names, token_ids = letter_table(tokenizer, decision_tokens)
    context = tokenizer.encode(
        "Context:\n" + _render_state(example.state), add_special_tokens=False
    )
    if len(context) > max_state_tokens:
        raise ConfigError("state exceeds the decider-v2.1 maximum state length")
    if example.kind == "score" and len(example.options) > MAX_SCORE_LEVELS:
        raise ConfigError("decider-v2.1 isolated score supports at most 10 levels")
    if option_order is None:
        option_order = presented_order(example, NOUL_ORDER, FORMAT)
    if sorted(option_order) != list(range(len(example.options))):
        raise ConfigError("option_order must be a permutation of the original options")
    specs: list[tuple[str, list[str]]] = []
    if example.kind == "score":
        for index in range(len(example.options)):
            text = re.sub(r"^\s*-?\d+\s*:\s*", "", score_description(example, index))
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
        rows.append(LabelRow(ids, token_ids[: len(options)]))
    return rows


def _native_config(checkpoint: str | Path, **hf_kwargs: Any) -> dict[str, Any]:
    path = Path(checkpoint)
    filename = (
        path / "decider_config.json"
        if path.is_dir()
        else Path(
            hf_hub_download(
                repo_id=str(checkpoint),
                filename="decider_config.json",
                **hub_kwargs(hf_kwargs),
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
        raise CheckpointError(
            "only the Mapika decider-4b v2.1 plain, isolated-score contract is supported"
        )
    return config


class DeciderDecisionModel(LabelReadoutModel):
    """Trainable SDK wrapper over Mapika's original letter head and prompt."""

    release_format = FORMAT
    family = FORMAT
    # False first: the v2.1 checkpoint was trained on "(A) no / (B) yes".
    noul_order = NOUL_ORDER
    # Score levels are isolated yes/no rows, so only choice and noul are permuted.
    seeded_kinds = frozenset({"choice", "noul"})

    def __init__(
        self,
        backbone: torch.nn.Module,
        tokenizer: Any,
        *,
        type_temperatures: Mapping[str, float] | None = None,
        **kwargs: Any,
    ) -> None:
        self.type_temperatures = dict(type_temperatures or DEFAULT_TEMPERATURES)
        if set(self.type_temperatures) != {"choice", "noul", "score"} or any(
            not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0
            for value in self.type_temperatures.values()
        ):
            raise ConfigError(
                "decider-v2.1 requires positive finite temperatures for each decision kind"
            )
        super().__init__(backbone, tokenizer, **kwargs)

    def _label_token_ids(self) -> list[int]:
        return letter_table(self.tokenizer, self.decision_tokens)[1]

    def _rows(self, example: DecisionInput, order: list[int]) -> list[LabelRow]:
        return prepare_decider_rows(
            example, self.tokenizer, decision_tokens=self.decision_tokens, option_order=order
        )

    def _presented_logits(
        self, example: DecisionInput, selected: list[torch.Tensor]
    ) -> torch.Tensor:
        temperature = self.type_temperatures[example.kind]
        if example.kind == "score":
            # Each level's row reads (A) no / (B) yes; the level's logit is log P(yes).
            return torch.stack([torch.log_softmax(row / temperature, dim=0)[1] for row in selected])
        return selected[0] / temperature

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
            raise CheckpointError(
                "only Mapika/decider-4b or a local copy of its v2.1 release is supported"
            )
        return cls._load_release(
            checkpoint,
            hf_kwargs,
            decision_tokens=MAX_OPTIONS,
            max_sequence_length=max_sequence_length,
            seed=seed,
            type_temperatures=config["temperature_by_type"],
        )

    def _save_decision_metadata(self, path: str | Path, *, adapter_base: str | None = None) -> None:
        super()._save_decision_metadata(path, adapter_base=adapter_base)
        filename = Path(path) / "decision_config.json"
        config = json.loads(filename.read_text(encoding="utf-8"))
        config["type_temperatures"] = self.type_temperatures
        filename.write_text(json.dumps(config, indent=2, allow_nan=False) + "\n", encoding="utf-8")
