"""Decider, JPT, and Eikos: released letter-logit readouts loaded as ``DecisionModel``s."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Any

from futurix_decision.errors import ConfigError
from futurix_decision.models.families.base import (
    FamilyAdapter,
    reject_futurix_options,
    supported_checkpoints,
)
from futurix_decision.models.families.futurix import on_device
from futurix_decision.models.sources import require_checkpoint_files

if TYPE_CHECKING:
    from futurix_decision.models.decision_model import DecisionModel


class LetterAdapter(FamilyAdapter):
    """One adapter class for the three letter families until they share a model base.

    ``prompt_format`` is the release's own prompt; training configs may not
    change it. ``fixed_width`` is false when the release metadata sets the
    option count. ``options`` are extra ``model:`` keys passed to the loader.
    """

    def __init__(
        self,
        name: str,
        *,
        prompt_format: str,
        fixed_width: bool = True,
        options: tuple[str, ...] = (),
    ) -> None:
        self.name = name
        self.checkpoints = supported_checkpoints(name)
        self.extra = name
        self.prompt_format = prompt_format
        self.fixed_width = fixed_width
        self.options = options

    def load(self, checkpoint: str, *, device: str | None = None, **kwargs: Any) -> DecisionModel:
        reject_futurix_options(kwargs, self.name)
        from futurix_decision.models.decision_model import DecisionModel

        loader = getattr(DecisionModel, f"from_{self.name}")
        return on_device(loader(checkpoint, **kwargs), device)

    def requires(self, path: Path) -> None:
        require_checkpoint_files(path, self.name)

    def training_kwargs(self, section: Mapping[str, Any], *, method: str) -> dict[str, Any]:
        if section.get("head_type", "tokens") != "tokens" or section.get("head_config"):
            raise ConfigError(
                "native letter models use their trained readout, not a new decision head"
            )
        if not self.fixed_width and "decision_tokens" in section:
            raise ConfigError(
                f"{self.name.capitalize()} one-pass width comes from the release metadata"
            )
        if self.fixed_width and section.get("decision_tokens", 255) != 255:
            raise ConfigError("native letter models support 255 existing option labels")
        if section.get("prompt_format", self.prompt_format) != self.prompt_format:
            raise ConfigError(f"{self.name} requires its native prompt")
        return {
            key: section[key]
            for key in ("max_sequence_length", "seed", *self.options)
            if key in section
        }
