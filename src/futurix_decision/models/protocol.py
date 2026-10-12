"""The contract every decision backend implements. Imports without torch."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Protocol, runtime_checkable

from futurix_decision.datasets.schema import DecisionInput, State
from futurix_decision.inference.schemas import DecisionResult


@runtime_checkable
class DecisionBackend(Protocol):
    """Anything that answers decisions: local models, vLLM, or an HTTP client.

    ``DecisionModel``, ``NativeDecisionModel``, ``VLLMEngine``, and
    ``DecisionClient`` implement it. ``isinstance`` checks
    only that the methods exist, not their signatures.
    """

    def decide(
        self, example: DecisionInput, *, confidence_threshold: float | None = None
    ) -> DecisionResult: ...

    def batch_decide(
        self, examples: Sequence[DecisionInput], *, confidence_threshold: float | None = None
    ) -> list[DecisionResult]: ...

    def choice(
        self,
        *,
        state: State,
        choices: Sequence[str],
        question: str = "Which option should be selected?",
        confidence_threshold: float | None = None,
    ) -> DecisionResult: ...

    def noul(
        self,
        *,
        state: State,
        question: str = "Is this true?",
        confidence_threshold: float | None = None,
    ) -> DecisionResult: ...

    def score(
        self,
        *,
        state: State,
        question: str = "What score should be assigned?",
        min_score: int = 0,
        max_score: int = 5,
        confidence_threshold: float | None = None,
    ) -> DecisionResult: ...

    def close(self) -> None:
        """Release connections the backend opened; a no-op for in-process models."""
        ...


if TYPE_CHECKING:
    from futurix_decision.client import DecisionClient
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
    ) -> tuple[DecisionBackend, ...]:
        return (model, native, engine, client)
