"""Liquid d1: LiquidAI's verbalizer over LFM2-VL, run through the release's remote code."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from futurix_decision.models.families.publisher import (
    PublisherAdapter,
    PublisherRuntime,
    load_images,
)
from futurix_decision.models.sources import CheckpointFiles


class LiquidRuntime(PublisherRuntime):
    def prepare_images(self, images: Sequence[Any]) -> list[Any]:
        return load_images(images)


class LiquidAdapter(PublisherAdapter):
    name = "liquid-d1"
    extra = "liquid-d1"
    readout = "verbalizer"
    runtime_type = LiquidRuntime
    images = True
    remote_code = True
    transformers = (5, 14)

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
