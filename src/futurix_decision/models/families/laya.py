"""Laya: ConvAI's encoder agent, run through the ``laya`` package."""

from __future__ import annotations

from futurix_decision.models.families.publisher import PublisherAdapter
from futurix_decision.models.sources import CheckpointFiles


class LayaAdapter(PublisherAdapter):
    name = "laya"
    extra = "laya"

    def files(self, files: CheckpointFiles) -> None:
        files.require("rl_agent_config.json", "model.safetensors")
