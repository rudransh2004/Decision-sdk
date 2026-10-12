"""Kev: Jared Palmer's pointer head over a Qwen3.5 LoRA, run through the ``kev`` package."""

from __future__ import annotations

from futurix_decision.models.families.publisher import PublisherAdapter
from futurix_decision.models.sources import CheckpointFiles


class KevAdapter(PublisherAdapter):
    name = "kev"
    extra = "kev-native"

    def files(self, files: CheckpointFiles) -> None:
        files.require("head.pt")
        if files.has("adapter_config.json"):
            files.one_of("adapter_model.safetensors", "adapter_model.bin")
        else:
            files.require("config.json")
            files.weights()
