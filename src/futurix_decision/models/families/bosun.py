"""Bosun v3.1 releases: Hanno Labs' Qwen base, PEFT adapter, and decision-token rows."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Any

from futurix_decision.errors import ConfigError
from futurix_decision.models.families.base import (
    FamilyAdapter,
    reject_futurix_options,
    supported_checkpoints,
)
from futurix_decision.models.families.futurix import on_device
from futurix_decision.models.sources import CheckpointFiles, read_checkpoint_json

if TYPE_CHECKING:
    from futurix_decision.models.decision_model import DecisionModel


class BosunAdapter(FamilyAdapter):
    name = "bosun"
    checkpoints = supported_checkpoints("bosun")
    extra = "bosun"

    def load(self, checkpoint: str, *, device: str | None = None, **kwargs: Any) -> DecisionModel:
        reject_futurix_options(kwargs, self.name)
        from futurix_decision.models.decision_model import DecisionModel

        # load_model already ran ``requires`` on a local copy.
        return on_device(DecisionModel.from_bosun(checkpoint, check_files=False, **kwargs), device)

    def requires(self, path: Path) -> None:
        files = CheckpointFiles(path)
        files.require("config.json")
        if files.has("config.json"):
            config = read_checkpoint_json(path / "config.json")
            files.require(
                str(config.get("decision_embeddings_file") or "decision_embeddings.safetensors")
            )
            adapter = str(config.get("adapter_subfolder") or "adapter")
            tokenizer = str(config.get("tokenizer_subfolder") or "tokenizer")
            files.require(f"{adapter}/adapter_config.json")
            files.one_of(f"{adapter}/adapter_model.safetensors", f"{adapter}/adapter_model.bin")
            files.one_of(f"{tokenizer}/tokenizer.json", f"{tokenizer}/vocab.json")
        files.check(self.name)

    def training_kwargs(self, section: Mapping[str, Any], *, method: str) -> dict[str, Any]:
        if section.get("head_type", "tokens") != "tokens":
            raise ConfigError("Bosun's prompt contract requires the token decision head")
        if method == "qlora":
            raise ConfigError("Bosun releases load unquantized; fine-tune them with full or lora")
        if "max_sequence_length" in section:
            return {"max_sequence_length": section["max_sequence_length"]}
        return {}
