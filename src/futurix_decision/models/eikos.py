"""Eikos 4B/27B text-only SemIf letter-logit decision contract.

This follows the released decision_core.py's fixed system prompt and compact
JSON user payload. The published image route and >max_one_pass tournament
are outside this adapter; excess candidates are rejected, never truncated.
"""

from __future__ import annotations

import json
import string
import weakref
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import torch
from huggingface_hub import hf_hub_download
from transformers import AutoModelForCausalLM, AutoTokenizer

from futurix_decision.datasets.schema import DecisionInput
from futurix_decision.models.decision_model import DecisionBatch, DecisionModel

FORMAT = "eikos-letter-v1-semif"
SOURCES = {"caiovicentino1/Eikos-4B": 100, "caiovicentino1/Eikos-27B": 160}
SYSTEM = (
    "Apply the supplied criterion to the supplied evidence. Choose exactly one listed option. "
    "Respond with only its uppercase letter, with no explanation or reasoning."
)
_SPLIT = frozenset(
    (
        "BQ BZ CJ CQ CZ DQ DZ EJ EY FJ FQ FV FZ GJ GK GQ GZ HJ IY JF JG JH JL JN JQ JW JX JY JZ "
        "KJ KQ KX KZ LH LJ LQ LW LX LZ OJ OQ OY OZ PQ PZ QD QF QI QJ QK QO QV QW QX QY QZ "
        "RQ RZ TJ TQ UJ UO UQ UW VH VJ VQ VU VW VX VY VZ WJ WQ WU WV WY WZ XG XJ XK XN XO "
        "XQ XU XV XW XZ YB YD YF YH YI YJ YK YQ YR YU YV YX ZB ZC ZD ZG ZJ ZK ZL ZM ZP ZQ ZS ZT ZU ZV"
    ).split()
)
_LABEL_CACHE: weakref.WeakKeyDictionary[Any, tuple[list[str], list[int]]] = (
    weakref.WeakKeyDictionary()
)


def eikos_label_table(tokenizer: Any, width: int) -> tuple[list[str], list[int]]:
    cached = _LABEL_CACHE.get(tokenizer)
    if cached is not None and len(cached[1]) >= width:
        return cached[0][:width], cached[1][:width]
    labels = list(string.ascii_uppercase) + [
        a + b for a in string.ascii_uppercase for b in string.ascii_uppercase if a + b not in _SPLIT
    ]
    selected = labels[:width]
    ids = [tokenizer.encode(label, add_special_tokens=False) for label in selected]
    if any(len(row) != 1 for row in ids):
        raise ValueError("Eikos tokenizer cannot encode its published labels as one token")
    singles = [row[0] for row in ids]
    if len(set(singles)) != width:
        raise ValueError("Eikos tokenizer aliases distinct published option labels")
    _LABEL_CACHE[tokenizer] = selected, singles
    return selected, singles


def prepare_eikos_row(
    example: DecisionInput,
    tokenizer: Any,
    *,
    max_options: int,
) -> tuple[list[int], list[int], list[int]]:
    if len(example.options) > max_options:
        raise ValueError(f"Eikos one-pass readout supports at most {max_options} options")
    order = (
        [example.options.index("true"), example.options.index("false")]
        if example.kind == "noul" and set(example.options) == {"false", "true"}
        else list(range(len(example.options)))
    )
    if example.kind == "noul" and set(example.options) != {"false", "true"}:
        raise ValueError("Eikos noul requires literal true/false options")
    labels, ids = eikos_label_table(tokenizer, max_options)
    options = []
    for position, original in enumerate(order):
        name = example.options[original]
        if example.kind == "noul":
            description = (
                example.candidate_descriptions[original] if example.candidate_descriptions else None
            ) or (
                example.candidate_labels[original]
                if example.candidate_labels
                else "yes"
                if name == "true"
                else "no"
            )
        elif example.kind == "score":
            description = (
                example.candidate_descriptions[original]
                if example.candidate_descriptions and example.candidate_descriptions[original]
                else example.candidate_labels[original]
                if example.candidate_labels
                else name
            )
            name = str(position)
        else:
            parts = []
            if example.candidate_labels:
                parts.append(example.candidate_labels[original])
            if example.candidate_descriptions and example.candidate_descriptions[original]:
                parts.append(example.candidate_descriptions[original])
            description = " — ".join(parts) if parts else name
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
    return tokenizer.encode(prompt, add_special_tokens=False), ids[: len(order)], order


def _native_config(checkpoint: str | Path, **hub_kwargs: Any) -> int:
    path = Path(checkpoint)
    downloader = {
        key: hub_kwargs[key]
        for key in ("revision", "token", "cache_dir", "local_files_only")
        if key in hub_kwargs
    }
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
        raise ValueError("only Eikos 4B/27B SemIf one-pass decision releases are supported")
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
        raise ValueError("Eikos release uses an unsupported nonconstant calibration")
    expected = SOURCES.get(str(checkpoint))
    if expected is not None and config["max_one_pass"] != expected:
        raise ValueError("Eikos max_one_pass does not match the named release")
    return int(config["max_one_pass"])


class EikosDecisionModel(DecisionModel):
    def __init__(self, backbone: torch.nn.Module, tokenizer: Any, **kwargs: Any) -> None:
        if (
            kwargs.get("prompt_format", FORMAT) != FORMAT
            or kwargs.get("head_type", "tokens") != "tokens"
        ):
            raise ValueError("Eikos requires its SemIf prompt and existing letter readout")
        kwargs["prompt_format"] = FORMAT
        super().__init__(backbone, tokenizer, **kwargs)

    def _register_decision_tokens(self) -> list[int]:
        if self.tokenizer.pad_token is None:
            if self.tokenizer.eos_token is None:
                raise ValueError("Eikos tokenizer needs a pad or EOS token")
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"
        _, ids = eikos_label_table(self.tokenizer, self.decision_tokens)
        if max(ids) >= self.backbone.get_input_embeddings().num_embeddings:
            raise ValueError("Eikos tokenizer label ID is outside the backbone vocabulary")
        return ids

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
            raise ValueError("supported Eikos releases are caiovicentino1/Eikos-4B and Eikos-27B")
        width = _native_config(checkpoint, **hf_kwargs)
        tokenizer_kwargs = {
            key: hf_kwargs[key]
            for key in ("revision", "token", "trust_remote_code", "cache_dir", "local_files_only")
            if key in hf_kwargs
        }
        tokenizer = AutoTokenizer.from_pretrained(str(checkpoint), **tokenizer_kwargs)
        backbone = AutoModelForCausalLM.from_pretrained(str(checkpoint), **hf_kwargs)
        return cls(
            backbone,
            tokenizer,
            decision_tokens=width,
            max_sequence_length=max_sequence_length,
            seed=seed,
            base_checkpoint=str(checkpoint),
            base_model_revision=hf_kwargs.get("revision"),
            source_checkpoint=str(checkpoint),
            source_revision=hf_kwargs.get("revision"),
        )

    def forward_decisions(
        self, examples: Sequence[DecisionInput], *, assignment_seed: int | None = None
    ) -> DecisionBatch:
        if not examples:
            raise ValueError("examples must not be empty")
        prepared = [
            prepare_eikos_row(e, self.tokenizer, max_options=self.decision_tokens) for e in examples
        ]
        longest = max(len(ids) for ids, _, _ in prepared)
        if longest > self.max_sequence_length:
            raise ValueError("prompt exceeds max_sequence_length; increase the limit")
        device = next(self.backbone.parameters()).device
        ids = torch.full(
            (len(prepared), longest),
            self.tokenizer.pad_token_id,
            dtype=torch.long,
            device=device,
        )
        mask = torch.zeros_like(ids)
        for index, (tokens, _, _) in enumerate(prepared):
            ids[index, -len(tokens) :] = torch.tensor(tokens, device=device)
            mask[index, -len(tokens) :] = 1
        output = self.backbone(input_ids=ids, attention_mask=mask, logits_to_keep=1)
        logits = output.logits[:, -1, :]
        results = logits.new_full(
            (len(examples), max(len(e.options) for e in examples)), float("-inf")
        )
        for index, (_, labels, order) in enumerate(prepared):
            selected = logits[index].index_select(0, torch.tensor(labels, device=device))
            results[index, order] = selected.float()
        return DecisionBatch(results, [list(range(len(e.options))) for e in examples])
