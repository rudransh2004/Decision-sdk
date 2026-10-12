"""Decision inference result, independent of the model backend."""

from __future__ import annotations

from dataclasses import dataclass

from pydantic import BaseModel, ConfigDict, Field, model_validator

from futurix_decision.datasets.schema import DecisionInput, DecisionKind, State


@dataclass(frozen=True)
class DecisionResult:
    decision: str | int | bool | None
    confidence: float
    probabilities: dict[str, float]
    abstained: bool
    latency_ms: float

    def to_dict(self) -> dict[str, object]:
        return {
            "decision": self.decision,
            "confidence": self.confidence,
            "probabilities": self.probabilities,
            "abstained": self.abstained,
            "latency_ms": self.latency_ms,
        }


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
                raise ValueError("choice requests require choices")
            options = self.choices
        elif self.kind == "noul":
            options = ["false", "true"]
        else:
            options = [str(level) for level in range(self.min_score, self.max_score + 1)]
        return DecisionInput(
            kind=self.kind, state=self.state, question=self.question, options=options
        )


class DecisionResponse(BaseModel):
    decision: str | int | bool | None
    confidence: float
    probabilities: dict[str, float]
    abstained: bool
    latency_ms: float
