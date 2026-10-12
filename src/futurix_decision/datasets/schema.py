"""Canonical, validated decision examples shared by datasets and inference."""

from __future__ import annotations

import json
import math
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, field_validator, model_validator

from futurix_decision.errors import ConfigError

DecisionKind = Literal["choice", "noul", "score"]
State = str | dict[str, Any] | list[Any]
ABSTAIN_TARGET = "__abstain__"


class DecisionInput(BaseModel):
    """One decision with runtime-defined, ordered candidates."""

    model_config = ConfigDict(extra="forbid")

    kind: DecisionKind = "choice"
    state: State
    question: str
    options: list[str]
    candidate_ids: list[str] | None = None
    candidate_labels: list[str] | None = None
    candidate_descriptions: list[str] | None = None
    native_question: dict[str, Any] | None = None
    images: list[str] | None = None
    preserve_order: bool = False  # Canonical benchmark rows may reverse Boolean candidates.

    @field_validator("state")
    @classmethod
    def check_state(cls, state: State) -> State:
        if isinstance(state, str) and not state.strip():
            raise ValueError("state must not be empty")
        if isinstance(state, list) and not state:
            raise ValueError("state must not be an empty list")
        if isinstance(state, (dict, list)):
            try:
                json.dumps(state, allow_nan=False, sort_keys=True)
            except (TypeError, ValueError) as exc:
                raise ValueError("state must be JSON-serializable with finite numbers") from exc
        return state

    @field_validator("question")
    @classmethod
    def check_question(cls, question: str) -> str:
        if not question.strip():
            raise ValueError("question must not be empty")
        return question

    @field_validator("native_question")
    @classmethod
    def check_native_question(cls, question: dict[str, Any] | None) -> dict[str, Any] | None:
        if question is not None:
            try:
                json.dumps(question, allow_nan=False, sort_keys=True)
            except (TypeError, ValueError) as exc:
                raise ValueError("native_question must be JSON-serializable") from exc
        return question

    @field_validator("images")
    @classmethod
    def check_images(cls, images: list[str] | None) -> list[str] | None:
        if images is not None and (
            not images or any(not isinstance(image, str) or not image.strip() for image in images)
        ):
            raise ValueError("images must be nonempty file paths or data URLs")
        return images

    @field_validator("options")
    @classmethod
    def check_options(cls, options: list[str]) -> list[str]:
        if not 2 <= len(options) <= 256:
            raise ValueError("options must contain between 2 and 256 candidates")
        if any(not option.strip() for option in options):
            raise ValueError("options must not contain empty candidates")
        if len(set(options)) != len(options):
            raise ValueError("options must be unique")
        return options

    @model_validator(mode="after")
    def check_kind(self) -> DecisionInput:
        for name in ("candidate_ids", "candidate_labels", "candidate_descriptions"):
            values = getattr(self, name)
            if values is not None and len(values) != len(self.options):
                raise ValueError(f"{name} must align with options")
        if self.candidate_ids is not None and len(set(self.candidate_ids)) != len(
            self.candidate_ids
        ):
            raise ValueError("candidate_ids must be unique")
        if self.kind == "noul" and self.options != ["false", "true"]:
            if not self.preserve_order or self.candidate_ids is None or len(self.options) != 2:
                raise ValueError('noul options must be ["false", "true"] in that order')
        if self.kind == "score":
            if not (self.preserve_order and self.candidate_ids is not None):
                try:
                    levels = [int(option) for option in self.options]
                except ValueError as exc:
                    raise ValueError("score options must be consecutive integer levels") from exc
                if [str(level) for level in levels] != self.options or levels != list(
                    range(levels[0], levels[0] + len(levels))
                ):
                    raise ValueError("score options must be consecutive integer levels")
        return self


class DecisionExample(DecisionInput):
    """Training/evaluation record; abstention is an explicit, supervised label."""

    id: str
    target: str | int | bool | list[float]
    group_id: str | None = None
    question_id: str | None = None
    domain: str | None = None
    family: str | None = None
    source: str | None = None

    @field_validator("id")
    @classmethod
    def check_id(cls, identifier: str) -> str:
        if not identifier.strip():
            raise ValueError("id must not be empty")
        return identifier

    @model_validator(mode="after")
    def check_target(self) -> DecisionExample:
        self.target_distribution(include_abstain=True)
        return self

    def target_distribution(self, *, include_abstain: bool = False) -> list[float]:
        """Return probabilities aligned with options, optionally followed by abstain."""
        if self.target == ABSTAIN_TARGET:
            if ABSTAIN_TARGET in self.options:
                raise ConfigError(f"{ABSTAIN_TARGET} is reserved for the abstain label")
            if not include_abstain:
                raise ConfigError("abstain targets require a model with learn_abstain=True")
            return [0.0] * len(self.options) + [1.0]
        if isinstance(self.target, list):
            values = self.target
            if len(values) not in (len(self.options), len(self.options) + 1):
                raise ConfigError("target length must match options or options plus abstain")
            if len(values) > len(self.options) and not include_abstain:
                raise ConfigError("abstain targets require a model with learn_abstain=True")
            if any(not math.isfinite(value) or not 0 <= value <= 1 for value in values):
                raise ConfigError("target probabilities must be finite and in [0, 1]")
            if not math.isclose(sum(values), 1.0, abs_tol=1e-5, rel_tol=0):
                raise ConfigError("target probabilities must sum to 1")
            return (
                values + [0.0] if include_abstain and len(values) == len(self.options) else values
            )

        label = self.target
        if isinstance(label, bool):
            if self.kind != "noul":
                raise ConfigError("boolean labels are only valid for noul")
            index = int(label)
        elif isinstance(label, str):
            if label not in self.options:
                raise ConfigError(f"target label {label!r} is not an option")
            index = self.options.index(label)
        elif isinstance(label, int):
            # Integer targets for a score are level values, not positional indices.
            if self.kind == "score":
                if str(label) not in self.options:
                    raise ConfigError(f"target score {label} is not an option")
                index = self.options.index(str(label))
            else:
                index = label
                if not 0 <= index < len(self.options):
                    raise ConfigError("target index is out of range")
        else:
            raise ConfigError("unsupported target type")
        values = [float(i == index) for i in range(len(self.options))]
        return values + [0.0] if include_abstain else values
