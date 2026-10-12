"""Futurix checkpoints: ``DecisionModel.from_pretrained`` on SDK-trained or plain LMs."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Any

from futurix_decision.errors import CheckpointError
from futurix_decision.models.families.base import FamilyAdapter
from futurix_decision.models.sources import CheckpointFiles, read_checkpoint_json

if TYPE_CHECKING:
    from futurix_decision.models.decision_model import DecisionModel

QWEN35 = ("Qwen/Qwen3.5-9B", "Qwen/Qwen3.5-9B-Base")


def on_device(model: DecisionModel, device: str | None) -> DecisionModel:
    """Move a loaded ``DecisionModel`` and its head to ``device`` when one is given."""
    if device is not None:
        model.backbone.to(device)
        if model.decision_head is not None:
            model.decision_head.to(device)
    return model


class FuturixAdapter(FamilyAdapter):
    """The SDK's own format. ``load`` also starts a new model from a plain causal LM."""

    name = "futurix"
    extra = "torch"

    def load(self, checkpoint: str, *, device: str | None = None, **kwargs: Any) -> DecisionModel:
        from futurix_decision.models.decision_model import DecisionModel

        return on_device(DecisionModel.from_pretrained(checkpoint, **kwargs), device)

    def requires(self, path: Path) -> None:
        files = CheckpointFiles(path)
        files.require("decision_config.json")
        metadata: dict[str, Any] = {}
        if files.has("decision_config.json"):
            metadata = read_checkpoint_json(path / "decision_config.json")
            if metadata.get("architecture") != "futurix-decision":
                raise CheckpointError(f"not a Futurix decision checkpoint: {path}")
            if metadata.get("head_type", "tokens") != "tokens":
                files.require("decision_head.safetensors")
        if files.has("adapter_config.json"):
            files.one_of("adapter_model.safetensors", "adapter_model.bin")
        else:
            files.require("config.json")
            files.weights()
        files.tokenizer()
        files.check(self.name)
        if metadata.get("source_checkpoint") in QWEN35:
            from futurix_decision.models.native import _requires_transformers

            _requires_transformers("qwen3.5", (5, 17))

    def training_kwargs(self, section: Mapping[str, Any], *, method: str) -> dict[str, Any]:
        kwargs = {
            key: section[key]
            for key in (
                "decision_tokens",
                "max_sequence_length",
                "seed",
                "head_type",
                "head_config",
            )
            if key in section
        }
        if section.get("prompt_format"):
            kwargs["prompt_format"] = section["prompt_format"]
        return kwargs
