"""Decision inference result, independent of the model backend."""

from __future__ import annotations

import math
from collections.abc import Sequence

from pydantic import BaseModel, ConfigDict, Field, model_validator

from futurix_decision.datasets.schema import DecisionInput, DecisionKind, State
from futurix_decision.errors import ConfigError, InvalidOutputError

Decision = bool | int | str | None


def typed_decision(kind: DecisionKind, option: str) -> Decision:
    """Return the decision value that ``option`` names for ``kind``.

    Choice decisions are the option itself. Noul options ``"true"`` and
    ``"false"`` become booleans and score options integer levels. A
    ``preserve_order`` row whose candidate IDs name no such value (for example
    a DecisionBench binary row with IDs ``c000``/``c001``) has no typed
    decision, so this returns ``None``; read the winner from ``probabilities``.
    """
    if kind == "choice":
        return option
    if kind == "noul":
        return {"true": True, "false": False}.get(option)
    try:
        level = int(option)
    except ValueError:
        return None
    return level if str(level) == option else None


ERROR_HEADER = "X-Futurix-Error"
"""Header on a ``futurix-decision serve`` error response naming its cause.

Its value matches the JSON body's ``error`` field: ``config`` (HTTP 422),
``checkpoint`` (500), ``backend`` (502), or ``invalid_output`` (502). The
client retries ``backend`` but raises ``InvalidOutputError`` for
``invalid_output`` at once, because the same request would fail again.
"""
INVALID_OUTPUT = "invalid_output"
# How far above 1 a probability may round before it counts as invalid output.
_PROBABILITY_TOLERANCE = 1e-6


class DecisionResult(BaseModel):
    """One decision, as every backend returns it and the HTTP API serves it.

    ``decision`` is typed by ``kind``: noul is ``bool``, score ``int``, and
    choice ``str``. It is ``None`` when the model abstained (confidence below
    the threshold), or for a ``preserve_order`` row whose candidate IDs name no
    noul or score value. ``probabilities`` maps every option to its
    probability and ``confidence`` is the highest one. ``latency_ms`` is the
    wall time per row: batched calls divide the batch time by the row count.
    """

    model_config = ConfigDict(frozen=True)

    kind: DecisionKind
    decision: Decision
    confidence: float
    probabilities: dict[str, float]
    abstained: bool
    latency_ms: float

    @model_validator(mode="after")
    def check_decision_type(self) -> DecisionResult:
        if self.decision is None:
            return self
        if self.abstained:
            raise ValueError("an abstained result has no decision")
        expected = {"choice": str, "noul": bool, "score": int}[self.kind]
        if not isinstance(self.decision, expected) or (
            self.kind == "score" and isinstance(self.decision, bool)
        ):
            raise ValueError(f"{self.kind} decisions must be {expected.__name__}")
        return self

    @classmethod
    def from_probabilities(
        cls,
        example: DecisionInput,
        probabilities: Sequence[float],
        *,
        latency_ms: float,
        confidence_threshold: float | None = None,
    ) -> DecisionResult:
        """Build the result for ``example`` from probabilities in option order.

        Raises ``InvalidOutputError`` if a probability is not a finite number
        in [0, 1], since a NaN would otherwise slip past abstention.
        """
        if len(probabilities) != len(example.options):
            raise ConfigError("probabilities must align with the example's options")
        values = [float(value) for value in probabilities]
        if not all(0.0 <= value <= 1.0 + _PROBABILITY_TOLERANCE for value in values):
            raise InvalidOutputError(f"probabilities must be finite and in [0, 1]: {values}")
        winner = max(range(len(values)), key=values.__getitem__)
        confidence = values[winner]
        abstained = confidence_threshold is not None and confidence < confidence_threshold
        return cls(
            kind=example.kind,
            decision=None if abstained else typed_decision(example.kind, example.options[winner]),
            confidence=confidence,
            probabilities=dict(zip(example.options, values, strict=True)),
            abstained=abstained,
            latency_ms=latency_ms,
        )


def check_confidence_threshold(confidence_threshold: float | None) -> None:
    if confidence_threshold is not None and (
        not math.isfinite(confidence_threshold) or not 0 <= confidence_threshold <= 1
    ):
        raise ConfigError("confidence_threshold must be between 0 and 1")


def choice_input(state: State, choices: Sequence[str], question: str) -> DecisionInput:
    return DecisionInput(state=state, question=question, options=list(choices))


def noul_input(state: State, question: str) -> DecisionInput:
    return DecisionInput(kind="noul", state=state, question=question, options=["false", "true"])


def score_input(state: State, question: str, min_score: int, max_score: int) -> DecisionInput:
    return DecisionInput(
        kind="score",
        state=state,
        question=question,
        options=[str(level) for level in range(min_score, max_score + 1)],
    )


class DecisionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    kind: DecisionKind = "choice"
    state: State
    question: str = "Which option should be selected?"
    choices: list[str] | None = None
    min_score: int = 0
    max_score: int = 5
    confidence_threshold: float | None = Field(default=None, ge=0, le=1, allow_inf_nan=False)

    @model_validator(mode="after")
    def validate_decision(self) -> DecisionRequest:
        # Kind-specific endpoints supply their kind after request parsing.
        # Still validate state/question here even if a choice has no options.
        if self.kind == "choice" and self.choices is None:
            DecisionInput(
                kind="noul", state=self.state, question=self.question, options=["false", "true"]
            )
        else:
            self.to_input()
        return self

    def to_input(self) -> DecisionInput:
        if self.kind == "choice":
            if self.choices is None:
                raise ConfigError("choice requests require choices")
            options = self.choices
        elif self.kind == "noul":
            options = ["false", "true"]
        else:
            options = [str(level) for level in range(self.min_score, self.max_score + 1)]
        return DecisionInput(
            kind=self.kind, state=self.state, question=self.question, options=options
        )
