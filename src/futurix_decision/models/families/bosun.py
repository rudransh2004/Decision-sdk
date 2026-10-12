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
from futurix_decision.models.sources import require_checkpoint_files

if TYPE_CHECKING:
    from futurix_decision.models.decision_model import DecisionModel


class BosunAdapter(FamilyAdapter):
    name = "bosun"
    checkpoints = supported_checkpoints("bosun")
    extra = "bosun"

    def load(self, checkpoint: str, *, device: str | None = None, **kwargs: Any) -> DecisionModel:
        reject_futurix_options(kwargs, self.name)
        from futurix_decision.models.decision_model import DecisionModel

        return on_device(DecisionModel.from_bosun(checkpoint, **kwargs), device)

    def requires(self, path: Path) -> None:
        require_checkpoint_files(path, "bosun")

    def training_kwargs(self, section: Mapping[str, Any], *, method: str) -> dict[str, Any]:
        if section.get("head_type", "tokens") != "tokens":
            raise ConfigError("Bosun's prompt contract requires the token decision head")
        if method == "qlora":
            raise ConfigError("Bosun releases load unquantized; fine-tune them with full or lora")
        if "max_sequence_length" in section:
            return {"max_sequence_length": section["max_sequence_length"]}
        return {}
