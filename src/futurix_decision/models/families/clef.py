"""Clef: Cloudflare's joint-schema head, run through the release's pinned model code."""

from __future__ import annotations

from futurix_decision.models.families.publisher import PublisherAdapter
from futurix_decision.models.sources import CheckpointFiles


class ClefAdapter(PublisherAdapter):
    name = "clef"
    extra = "clef"

    def files(self, files: CheckpointFiles) -> None:
        files.require(
            "config.json",
            "joint_schema_model.py",
            "joint_head_config.json",
            "joint_head.safetensors",
        )
        files.tokenizer()
        files.weights()
