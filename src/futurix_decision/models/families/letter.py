"""Decider, JPT, and Eikos: released letter-logit readouts loaded as ``DecisionModel``s."""

from __future__ import annotations

import importlib
from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Any

from futurix_decision.errors import ConfigError
from futurix_decision.models.families.base import FamilyAdapter, reject_futurix_options
from futurix_decision.models.sources import CheckpointFiles, optional_json

if TYPE_CHECKING:
    from futurix_decision.models.decision_model import DecisionModel
    from futurix_decision.models.label_readout import LabelReadoutModel


class LetterAdapter(FamilyAdapter):
    """Adapter for a family whose model subclasses ``LabelReadoutModel``.

    Subclasses set ``model`` to that class as ``"module:Class"``, so the
    registry imports without torch. The release's prompt format, option width,
    and extra loader options come from the class. ``required_files`` are the release
    files besides the config, tokenizer, and weights that every local copy needs.
    """

    model: str
    required_files: tuple[str, ...] = ()
    trainable = True
    vllm = "export"

    def model_class(self) -> type[LabelReadoutModel]:
        module, _, attribute = self.model.partition(":")
        model: type[LabelReadoutModel] = getattr(importlib.import_module(module), attribute)
        return model

    def load(self, checkpoint: str, *, device: str | None = None, **kwargs: Any) -> DecisionModel:
        reject_futurix_options(kwargs, self.name)
        return self.model_class().from_native(checkpoint, **kwargs).to(device)

    def requires(self, path: Path) -> None:
        files = CheckpointFiles(path)
        files.require(*self.required_files, "config.json")
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


class DeciderAdapter(LetterAdapter):
    name = "decider"
    extra = "decider"
    model = "futurix_decision.models.decider:DeciderDecisionModel"
    checkpoints = ("Mapika/decider-4b",)
    markers = ("decider_config.json",)
    required_files = ("decider_config.json",)


class JPTAdapter(LetterAdapter):
    name = "jpt"
    extra = "jpt"
    model = "futurix_decision.models.jpt:JPTDecisionModel"
    checkpoints = ("kirp/jpt-4b", "kirp/jpt-9b")
    unsupported_checkpoints = ("kirp/jpt-35b-a3b",)
    required_files = ("README.md",)

    def detect(self, directory: Path) -> bool:
        card = directory / "README.md"
        return (
            card.is_file() and "# JPT-" in card.read_text(encoding="utf-8", errors="replace")[:4096]
        )


class EikosAdapter(LetterAdapter):
    name = "eikos"
    extra = "eikos"
    model = "futurix_decision.models.eikos:EikosDecisionModel"
    checkpoints = ("caiovicentino1/Eikos-4B", "caiovicentino1/Eikos-27B")
    required_files = ("decision_config.json", "calib.json")

    def detect(self, directory: Path) -> bool:
        metadata = optional_json(directory, "decision_config.json")
        return metadata.get("prompt_version") == "letter-v1-semif"
