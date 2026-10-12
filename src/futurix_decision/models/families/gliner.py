"""GLiNER2.5-Decide: Fastino's schema encoder, run through ``gliner2``'s classifier."""

from __future__ import annotations

from futurix_decision.models.families.publisher import PublisherAdapter
from futurix_decision.models.sources import CheckpointFiles


class GlinerAdapter(PublisherAdapter):
    name = "gliner"
    extra = "gliner"

    def files(self, files: CheckpointFiles) -> None:
        files.require("config.json", "encoder_config/config.json")
        files.one_of("model.safetensors", "pytorch_model.bin")
