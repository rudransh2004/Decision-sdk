"""PPLX Decider: Perplexity's noncausal readout, run through the publisher's ``autojev``."""

from __future__ import annotations

from futurix_decision.models.families.publisher import PublisherAdapter
from futurix_decision.models.sources import CheckpointFiles


class PplxAdapter(PublisherAdapter):
    name = "pplx-decider"
    extra = "pplx-decider"

    def files(self, files: CheckpointFiles) -> None:
        files.require("config.json", "decision_config.json", "readout.safetensors")
        files.tokenizer()
        files.weights()
