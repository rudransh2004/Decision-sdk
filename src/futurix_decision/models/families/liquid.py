"""Liquid d1: LiquidAI's verbalizer over LFM2-VL, run through the release's remote code."""

from __future__ import annotations

from futurix_decision.models.families.publisher import PublisherAdapter
from futurix_decision.models.sources import CheckpointFiles


class LiquidAdapter(PublisherAdapter):
    name = "liquid-d1"
    extra = "liquid-d1"

    def files(self, files: CheckpointFiles) -> None:
        files.require(
            "config.json",
            "modeling_d1.py",
            "lfm2_vl.py",
            "runner.py",
            "hybrid.py",
            "api.py",
            "prompt.py",
            "tokenizer.json",
            "tokenizer_config.json",
            "processor_config.json",
            "chat_template.jinja",
        )
        files.weights()
