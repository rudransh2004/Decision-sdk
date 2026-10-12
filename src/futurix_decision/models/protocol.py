"""The contract every decision backend implements. Imports without torch."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import TYPE_CHECKING, Protocol, runtime_checkable

from futurix_decision.datasets.schema import DecisionInput, State
from futurix_decision.inference.schemas import (
    DecisionResult,
    choice_input,
    noul_input,
    score_input,
)


@runtime_checkable
class DecisionBackend(Protocol):
    """Anything that answers decisions: local models, vLLM, or an HTTP client.

    ``DecisionModel``, ``NativeDecisionModel``, ``VLLMEngine``,
    ``DecisionClient``, and ``Calibrated`` implement it. ``isinstance`` checks
    only that the methods exist, not their signatures.
    """

    def decide(
        self, example: DecisionInput, *, confidence_threshold: float | None = None
    ) -> DecisionResult:
        """Decide one example."""

    def batch_decide(
        self, examples: Sequence[DecisionInput], *, confidence_threshold: float | None = None
    ) -> list[DecisionResult]:
        """Decide each example, returning results in input order."""

    def choice(
        self,
        *,
        state: State,
        choices: Sequence[str],
        question: str = "Which option should be selected?",
        confidence_threshold: float | None = None,
    ) -> DecisionResult:
        """Pick one of ``choices`` for ``state``."""

    def noul(
        self,
        *,
        state: State,
        question: str = "Is this true?",
        confidence_threshold: float | None = None,
    ) -> DecisionResult:
        """Answer a yes/no question about ``state``."""

    def score(
        self,
        *,
        state: State,
        question: str = "What score should be assigned?",
        min_score: int = 0,
        max_score: int = 5,
        confidence_threshold: float | None = None,
    ) -> DecisionResult:
        """Rate ``state`` from ``min_score`` to ``max_score``."""

    def close(self) -> None:
        """Release connections the backend opened; a no-op for in-process models."""
        ...


class DecisionShortcuts(ABC):
    """``choice``, ``noul``, and ``score`` as one-example ``decide`` calls."""

    @abstractmethod
    def decide(
        self, example: DecisionInput, *, confidence_threshold: float | None = None
    ) -> DecisionResult: ...

    def choice(
        self,
        *,
        state: State,
        choices: Sequence[str],
        question: str = "Which option should be selected?",
        confidence_threshold: float | None = None,
    ) -> DecisionResult:
        return self.decide(
            choice_input(state, choices, question), confidence_threshold=confidence_threshold
        )

    def noul(
        self,
        *,
        state: State,
        question: str = "Is this true?",
        confidence_threshold: float | None = None,
    ) -> DecisionResult:
        return self.decide(noul_input(state, question), confidence_threshold=confidence_threshold)

    def score(
        self,
        *,
        state: State,
        question: str = "What score should be assigned?",
        min_score: int = 0,
        max_score: int = 5,
        confidence_threshold: float | None = None,
    ) -> DecisionResult:
        return self.decide(
            score_input(state, question, min_score, max_score),
            confidence_threshold=confidence_threshold,
        )


if TYPE_CHECKING:
    from futurix_decision.client import DecisionClient
    from futurix_decision.models.calibrated import Calibrated
    from futurix_decision.models.decision_model import DecisionModel
    from futurix_decision.models.native import NativeDecisionModel
    from futurix_decision.vllm.engine import VLLMEngine

    # mypy checks each implementation's signatures against the protocol here;
    # a runtime isinstance check only sees that the method names exist.
    def _implementations(
        model: DecisionModel,
        native: NativeDecisionModel,
        engine: VLLMEngine,
        client: DecisionClient,
        calibrated: Calibrated,
    ) -> tuple[DecisionBackend, ...]:
        return (model, native, engine, client, calibrated)
