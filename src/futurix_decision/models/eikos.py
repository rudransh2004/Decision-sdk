"""Eikos 4B/27B text-only SemIf letter-logit decision contract.

This follows the released decision_core.py's fixed system prompt and compact
JSON user payload. The published image route and >max_one_pass tournament
are outside this adapter; excess candidates are rejected, never truncated.
"""

from __future__ import annotations

import json
import string
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from huggingface_hub import hf_hub_download

from futurix_decision.datasets.schema import DecisionInput
from futurix_decision.errors import CheckpointError, ConfigError
from futurix_decision.models.label_readout import (
    LabelReadoutModel,
    LabelRow,
    cached_label_table,
    choice_description,
    hub_kwargs,
    noul_description,
    presented_order,
    score_description,
)

FORMAT = "eikos-letter-v1-semif"
SOURCES = {"caiovicentino1/Eikos-4B": 100, "caiovicentino1/Eikos-27B": 160}
SYSTEM = (
    "Apply the supplied criterion to the supplied evidence. Choose exactly one listed option. "
    "Respond with only its uppercase letter, with no explanation or reasoning."
)
# The released prompt lists "true" before "false", as the checkpoint was trained.
NOUL_ORDER = ("true", "false")
_SPLIT = frozenset(
    (
        "BQ BZ CJ CQ CZ DQ DZ EJ EY FJ FQ FV FZ GJ GK GQ GZ HJ IY JF JG JH JL JN JQ JW JX JY JZ "
        "KJ KQ KX KZ LH LJ LQ LW LX LZ OJ OQ OY OZ PQ PZ QD QF QI QJ QK QO QV QW QX QY QZ "
        "RQ RZ TJ TQ UJ UO UQ UW VH VJ VQ VU VW VX VY VZ WJ WQ WU WV WY WZ XG XJ XK XN XO "
        "XQ XU XV XW XZ YB YD YF YH YI YJ YK YQ YR YU YV YX "
        "ZB ZC ZD ZG ZJ ZK ZL ZM ZP ZQ ZS ZT ZU ZV"
    ).split()
)


def eikos_label_table(tokenizer: Any, width: int) -> tuple[list[str], list[int]]:
    return cached_label_table(tokenizer, FORMAT, width, _build_label_table)


def _build_label_table(tokenizer: Any, width: int) -> tuple[list[str], list[int]]:
    labels = list(string.ascii_uppercase) + [
        a + b for a in string.ascii_uppercase for b in string.ascii_uppercase if a + b not in _SPLIT
    ]
    selected = labels[:width]
    ids = [tokenizer.encode(label, add_special_tokens=False) for label in selected]
    if any(len(row) != 1 for row in ids):
        raise CheckpointError("Eikos tokenizer cannot encode its published labels as one token")
    singles = [row[0] for row in ids]
    if len(set(singles)) != width:
        raise CheckpointError("Eikos tokenizer aliases distinct published option labels")
    return selected, singles


def prepare_eikos_row(
    example: DecisionInput,
    tokenizer: Any,
    *,
    max_options: int,
    order: Sequence[int] | None = None,
) -> tuple[list[int], list[int], list[int]]:
    if len(example.options) > max_options:
        raise ConfigError(f"Eikos one-pass readout supports at most {max_options} options")
    if order is None:
        order = presented_order(example, NOUL_ORDER, "Eikos")
    if sorted(order) != list(range(len(example.options))):
        raise ConfigError("Eikos option order must be a permutation")
    labels, ids = eikos_label_table(tokenizer, max_options)
    options = []
    for position, original in enumerate(order):
        name = example.options[original]
        if example.kind == "noul":
            description = noul_description(example, original, "yes" if name == "true" else "no")
        elif example.kind == "score":
            description = score_description(example, original)
            name = str(position)
        else:
            joined = choice_description(example, original)
            description = name if joined is None else joined
        options.append(
            {
                "letter": labels[position],
                "description": f"{name}: {description}",
            }
        )
    payload = {
        "evidence": example.state,
        "criterion": example.question,
        "options": options,
    }
    prompt = tokenizer.apply_chat_template(
        [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
        ],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    return tokenizer.encode(prompt, add_special_tokens=False), ids[: len(order)], list(order)


def _native_config(checkpoint: str | Path, **hf_kwargs: Any) -> int:
    path = Path(checkpoint)
    downloader = hub_kwargs(hf_kwargs)
    config_path = (
        path / "decision_config.json"
        if path.is_dir()
        else Path(
            hf_hub_download(repo_id=str(checkpoint), filename="decision_config.json", **downloader)
        )
    )
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if (
        config.get("prompt_version") != "letter-v1-semif"
        or config.get("readout") != "letter-logit"
        or config.get("max_one_pass") not in (100, 160)
        or config.get("calib") != "calib.json"
    ):
        raise CheckpointError("only Eikos 4B/27B SemIf one-pass decision releases are supported")
    calib_path = (
        path / "calib.json"
        if path.is_dir()
        else Path(hf_hub_download(repo_id=str(checkpoint), filename="calib.json", **downloader))
    )
    calibration = json.loads(calib_path.read_text(encoding="utf-8"))
    if (
        calibration.get("mode") != "T1"
        or calibration.get("t_global") != 1.0
        or calibration.get("b") != 0.0
        or calibration.get("w") != [0, 0, 0, 0]
    ):
        raise CheckpointError("Eikos release uses an unsupported nonconstant calibration")
    expected = SOURCES.get(str(checkpoint))
    if expected is not None and config["max_one_pass"] != expected:
        raise CheckpointError("Eikos max_one_pass does not match the named release")
    return int(config["max_one_pass"])


class EikosDecisionModel(LabelReadoutModel):
    release_format = FORMAT
    family = "Eikos"
    # True first: the SemIf checkpoint was trained with "true" as option A.
    noul_order = NOUL_ORDER
    # The one-pass width comes from the release's decision_config.json.
    default_width = None
    seeded_kinds = frozenset()

    def _label_token_ids(self) -> list[int]:
        return eikos_label_table(self.tokenizer, self.decision_tokens)[1]

    def _rows(self, example: DecisionInput, order: list[int]) -> list[LabelRow]:
        tokens, labels, _ = prepare_eikos_row(
            example, self.tokenizer, max_options=self.decision_tokens, order=order
        )
        return [LabelRow(tokens, labels)]

    @classmethod
    def from_native(
        cls,
        checkpoint: str | Path,
        *,
        max_sequence_length: int = 32768,
        seed: int = 0,
        **hf_kwargs: Any,
    ) -> EikosDecisionModel:
        if str(checkpoint) not in SOURCES and not Path(checkpoint).is_dir():
            raise CheckpointError(
                "supported Eikos releases are caiovicentino1/Eikos-4B and Eikos-27B"
            )
        return cls._load_release(
            checkpoint,
            hf_kwargs,
            decision_tokens=_native_config(checkpoint, **hf_kwargs),
            max_sequence_length=max_sequence_length,
            seed=seed,
        )
