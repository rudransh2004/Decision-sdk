"""``DecisionModelConfig``: a ``DecisionModel``'s settings, saved as ``decision_config.json``."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from dataclasses import dataclass, field, fields
from typing import Any

from futurix_decision.errors import ConfigError
from futurix_decision.models.bosun import BOSUN_PROMPT_SCHEMA

# Settings a saved checkpoint fixes: its trained weights depend on them.
_LOCKED = frozenset(
    {
        "head_type",
        "head_config",
        "head_prompt_version",
        "train_backbone",
        "bosun_template_fallback",
        "learn_abstain",
        "attention_mode",
        "multimodal_loader",
    }
)
# The prompt contract each readout records; every other (custom) head reads span endpoints.
# Token readouts keep "marker-v1", the value their saved checkpoints already carry.
_PROMPT_VERSIONS = {"tokens": "marker-v1", "letter_code": "letter-code-v1"}
# Where the weights came from. ``from_pretrained`` derives these from the checkpoint.
_PROVENANCE = ("base_checkpoint", "source_checkpoint", "source_revision", "base_model_revision")


@dataclass
class DecisionModelConfig:
    """How a ``DecisionModel`` prompts its backbone and reads decisions from it.

    ``head_type`` picks the readout: ``"tokens"`` scores added decision-token
    rows, ``"letter_code"`` a fixed answer-code table, and any other registered
    head name (``"pointer"`` by default) a candidate-aware head over candidate
    span endpoints. ``head_prompt_version`` follows from ``head_type`` and is
    recorded so a checkpoint names its prompt contract. ``temperature`` and
    ``max_sequence_length`` may change after loading; changing the other
    fields of a live model is unsupported.
    """

    decision_tokens: int = 256
    temperature: float = 1.0
    seed: int = 0
    max_sequence_length: int = 1024
    prompt_format: str = "futurix"
    prompt_schema: str = BOSUN_PROMPT_SCHEMA
    head_type: str = "tokens"
    head_config: dict[str, Any] = field(default_factory=dict)
    head_prompt_version: str = ""
    train_backbone: bool = True
    bosun_template_fallback: bool = False
    learn_abstain: bool = False
    attention_mode: str = "causal"
    multimodal_loader: bool = False
    base_checkpoint: str | None = None
    source_checkpoint: str | None = None
    source_revision: str | None = None
    base_model_revision: str | None = None

    def __post_init__(self) -> None:
        self.head_config = dict(self.head_config or {})
        self.source_checkpoint = self.source_checkpoint or self.base_checkpoint
        tokens = self.head_type == "tokens"
        bosun = self.prompt_format == "bosun-v3"
        if not 2 <= self.decision_tokens <= 256:
            raise ConfigError("decision_tokens must be between 2 and 256")
        if not math.isfinite(self.temperature) or self.temperature <= 0:
            raise ConfigError("temperature must be finite and positive")
        if self.max_sequence_length <= 0:
            raise ConfigError("max_sequence_length must be positive")
        if bosun and self.decision_tokens != 256:
            raise ConfigError("Bosun v3 requires exactly 256 decision tokens")
        if not tokens and self.prompt_format != "futurix":
            raise ConfigError(
                "custom heads require prompt_format='futurix', not Bosun's token contract"
            )
        if tokens and self.head_config:
            raise ConfigError("token readout does not take a custom decision head")
        expected = _PROMPT_VERSIONS.get(self.head_type, "span-v1")
        if not self.head_prompt_version:
            self.head_prompt_version = expected
        elif self.head_prompt_version != expected:
            raise ConfigError(
                f"head_type={self.head_type!r} requires head_prompt_version={expected!r}"
            )
        if self.head_type == "letter_code" and self.decision_tokens > 255:
            raise ConfigError("letter-code readout supports at most 255 options")
        if self.attention_mode not in ("causal", "noncausal_full_attention"):
            raise ConfigError("attention_mode must be causal or noncausal_full_attention")
        if self.attention_mode != "causal" and self.head_type != "letter_code":
            raise ConfigError("noncausal attention is only implemented for letter-code readouts")
        for name in ("multimodal_loader", "train_backbone"):
            if not isinstance(getattr(self, name), bool):
                raise ConfigError(f"{name} must be a boolean")
        if not isinstance(self.bosun_template_fallback, bool) or (
            self.bosun_template_fallback and not bosun
        ):
            raise ConfigError("Bosun chat-template fallback requires prompt_format='bosun-v3'")
        if not isinstance(self.learn_abstain, bool) or (
            self.learn_abstain and not (self.head_type == "pointer" or (tokens and bosun))
        ):
            raise ConfigError("learn_abstain requires a pointer head or Bosun token readout")
        if self.head_type == "pointer" and "abstain" in self.head_config:
            raise ConfigError("configure pointer abstention with learn_abstain, not head_config")
        # Store only simple, JSON-safe architecture parameters in checkpoints.
        json.dumps(self.head_config, allow_nan=False)

    @classmethod
    def from_saved(
        cls, saved: Mapping[str, Any], requested: Mapping[str, Any], **provenance: str | None
    ) -> DecisionModelConfig:
        """Merge a checkpoint's ``decision_config.json`` with the caller's settings.

        ``requested`` values of ``None`` mean "use the saved value". A setting
        the checkpoint fixes (``_LOCKED``) must match the saved one.
        ``provenance`` replaces the saved weight-origin fields.
        """
        values: dict[str, Any] = {}
        for name in (item.name for item in fields(cls)):
            if name in _PROVENANCE:
                values[name] = provenance.get(name)
                continue
            stored = saved.get(name)
            wanted = requested.get(name)
            if name in _LOCKED and None not in (stored, wanted) and wanted != stored:
                raise ConfigError(f"cannot override the saved checkpoint's {name}")
            value = wanted if wanted is not None else stored
            if value is not None:
                values[name] = value
        return cls(**values)

    def metadata(self) -> dict[str, Any]:
        """The ``decision_config.json`` contents, except an adapter's ``base_checkpoint``."""
        return {
            "architecture": "futurix-decision",
            "head_type": self.head_type,
            "head_config": self.head_config,
            "head_prompt_version": self.head_prompt_version,
            "train_backbone": self.train_backbone,
            "bosun_template_fallback": self.bosun_template_fallback,
            "learn_abstain": self.learn_abstain,
            "attention_mode": self.attention_mode,
            "multimodal_loader": self.multimodal_loader,
            "decision_tokens": self.decision_tokens,
            "temperature": self.temperature,
            "seed": self.seed,
            "max_sequence_length": self.max_sequence_length,
            "prompt_format": self.prompt_format,
            "prompt_schema": self.prompt_schema,
            **{
                name: getattr(self, name)
                for name in ("source_checkpoint", "source_revision", "base_model_revision")
                if getattr(self, name)
            },
            "supported_types": ["choice", "noul", "score"],
        }


# Fields a caller may set when loading a checkpoint; the rest describe where it came from.
LOADABLE = tuple(
    item.name
    for item in fields(DecisionModelConfig)
    if item.name not in (*_PROVENANCE, "prompt_schema")
)
