"""Publisher runtimes: each family runs its publisher's own code through ``NativeDecisionModel``."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

from futurix_decision.errors import ConfigError
from futurix_decision.models.families.base import FamilyAdapter, reject_futurix_options
from futurix_decision.models.protocol import DecisionBackend
from futurix_decision.models.sources import require_checkpoint_files

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
    """Delegates to ``NativeDecisionModel``, which still branches on the family.

    ``local_files`` is false for OpenJev, which the SDK only reaches over HTTP.
    """

    def __init__(self, name: str, *, extra: str | None, local_files: bool = True) -> None:
        self.name = name
        self.checkpoints = tuple(
            checkpoint for checkpoint, family in PUBLISHER_CHECKPOINTS.items() if family == name
        )
        self.extra = extra
        self.local_files = local_files

    def load(
        self, checkpoint: str, *, device: str | None = None, **kwargs: Any
    ) -> NativeDecisionModel:
        reject_futurix_options(kwargs, self.name)
        from futurix_decision.models.native import NativeDecisionModel

        return NativeDecisionModel.from_pretrained(
            checkpoint, family=self.name, device=device or "cpu", **kwargs
        )

    def requires(self, path: Path) -> None:
        if self.local_files:
            require_checkpoint_files(path, self.name)

    def finetune(
        self, model: DecisionBackend, dataset: Any, output_dir: str | Path, **kwargs: Any
    ) -> NativeDecisionModel:
        from futurix_decision.models.native import NativeDecisionModel

        if not isinstance(model, NativeDecisionModel) or model.family != self.name:
            raise ConfigError(f"{self.name} fine-tuning needs a {self.name} NativeDecisionModel")
        return model.finetune(dataset, output_dir, **kwargs)
