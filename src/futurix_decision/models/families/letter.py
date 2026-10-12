"""Decider, JPT, and Eikos: released letter-logit readouts loaded as ``DecisionModel``s."""

from __future__ import annotations

import importlib
from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Any

from futurix_decision.errors import ConfigError
from futurix_decision.models.families.base import (
    FamilyAdapter,
    reject_futurix_options,
    supported_checkpoints,
)
from futurix_decision.models.sources import CheckpointFiles

if TYPE_CHECKING:
    from futurix_decision.models.decision_model import DecisionModel
    from futurix_decision.models.label_readout import LabelReadoutModel


class LetterAdapter(FamilyAdapter):
    """Adapter for a family whose model subclasses ``LabelReadoutModel``.

    ``model`` names that class as ``"module:Class"``, so the registry imports
    without torch. The release's prompt format, option width, and extra loader
    options come from the class. ``markers`` are the release files besides the
    config, tokenizer, and weights that every local copy needs.
    """

    def __init__(self, name: str, model: str, *, markers: tuple[str, ...]) -> None:
        self.name = name
        self.checkpoints = supported_checkpoints(name)
        self.extra = name
        self._model = model
        self._markers = markers

    def model_class(self) -> type[LabelReadoutModel]:
        module, _, attribute = self._model.partition(":")
        model: type[LabelReadoutModel] = getattr(importlib.import_module(module), attribute)
        return model

    def load(self, checkpoint: str, *, device: str | None = None, **kwargs: Any) -> DecisionModel:
        reject_futurix_options(kwargs, self.name)
        return self.model_class().from_native(checkpoint, **kwargs).to(device)

    def requires(self, path: Path) -> None:
        files = CheckpointFiles(path)
        files.require(*self._markers, "config.json")
        files.tokenizer()
        files.weights()
        files.check(self.name)

    def training_kwargs(self, section: Mapping[str, Any], *, method: str) -> dict[str, Any]:
        model = self.model_class()
        if section.get("head_type", "tokens") != "tokens" or section.get("head_config"):
            raise ConfigError(
                "native letter models use their trained readout, not a new decision head"
            )
        if model.default_width is None:
            if "decision_tokens" in section:
                raise ConfigError(f"{model.family} one-pass width comes from the release metadata")
        elif section.get("decision_tokens", model.default_width) != model.default_width:
            raise ConfigError(
                f"native letter models support {model.default_width} existing option labels"
            )
        if section.get("prompt_format", model.release_format) != model.release_format:
            raise ConfigError(f"{self.name} requires its native prompt")
        return {
            key: section[key]
            for key in ("max_sequence_length", "seed", *model.native_options)
            if key in section
        }
