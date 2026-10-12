"""Publisher runtimes: each family runs its publisher's own code through ``NativeDecisionModel``."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

from futurix_decision.errors import ConfigError
from futurix_decision.models.families.base import FamilyAdapter, reject_futurix_options
from futurix_decision.models.protocol import DecisionBackend
from futurix_decision.models.sources import CheckpointFiles

if TYPE_CHECKING:
    from futurix_decision.models.native import NativeDecisionModel

# Hub ID -> publisher runtime. Laya and GLiNER share the "encoder" row of the
# family table but run different code.
PUBLISHER_CHECKPOINTS = {
    "jaredpalmer/kev-9b": "kev",
    "convaiinnovations/laya": "laya",
    "convaiinnovations/laya-multilingual": "laya",
    "fastino/GLiNER2.5-Decide": "gliner",
    "Cloudflare/clef": "clef",
    "Cloudflare/clef-flash": "clef",
    "perplexity-ai/pplx-decider-v1.1-27b": "pplx-decider",
    "openjev/openjev": "openjev",
    "openjev/OpenJev-Flash-9B": "openjev",
    "vllm-sr/Decision-1.0-Nox-4B": "nox",
    "vllm-sr/Decision-2.0-Nox-4B": "nox",
    "vllm-sr/Decision-1.0-Lux-9B": "lux",
    "vllm-sr/Decision-2.0-Lux-9B": "lux",
    "LiquidAI/d1-3B": "liquid-d1",
}


class PublisherAdapter(FamilyAdapter):
    """One publisher runtime. Subclasses set ``name`` and ``extra`` and list their files."""

    def __init__(self) -> None:
        self.checkpoints = tuple(
            hub_id for hub_id, family in PUBLISHER_CHECKPOINTS.items() if family == self.name
        )

    def load(
        self, checkpoint: str, *, device: str | None = None, **kwargs: Any
    ) -> NativeDecisionModel:
        reject_futurix_options(kwargs, self.name)
        from futurix_decision.models.native import NativeDecisionModel

        return NativeDecisionModel.from_pretrained(
            checkpoint, family=self.name, device=device or "cpu", **kwargs
        )

    def requires(self, path: Path, *, source: str | None = None) -> None:
        """Check a local copy or Hub snapshot; ``source`` names the Hub ID in the error."""
        files = CheckpointFiles(path)
        self.files(files)
        files.check(self.name, source=source)

    def files(self, files: CheckpointFiles) -> None:
        """Mark each file the publisher loader reads. The default needs none."""

    def finetune(
        self, model: DecisionBackend, dataset: Any, output_dir: str | Path, **kwargs: Any
    ) -> NativeDecisionModel:
        from futurix_decision.models.native import NativeDecisionModel

        if not isinstance(model, NativeDecisionModel) or model.family != self.name:
            raise ConfigError(f"{self.name} fine-tuning needs a {self.name} NativeDecisionModel")
        return model.finetune(dataset, output_dir, **kwargs)
