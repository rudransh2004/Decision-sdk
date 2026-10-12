"""Temperature scaling over any backend's returned probabilities. Imports without torch."""

from __future__ import annotations

import math
from collections.abc import Sequence

from futurix_decision.datasets.schema import DecisionInput, State
from futurix_decision.errors import ConfigError
from futurix_decision.inference.schemas import (
    DecisionResult,
    check_confidence_threshold,
    choice_input,
    noul_input,
    score_input,
)
from futurix_decision.models.protocol import DecisionBackend


def scale_probabilities(probabilities: Sequence[float], temperature: float) -> list[float]:
    """Return ``softmax(log p / T)``. A zero probability stays zero.

    For ``p = softmax(logits)`` this equals ``softmax(logits / T)``: the
    log-normalizer is a constant shift that the softmax removes.
    """
    scaled = [math.log(p) / temperature if p > 0 else -math.inf for p in probabilities]
    peak = max(scaled)
    weights = [math.exp(value - peak) for value in scaled]
    total = sum(weights)
    return [weight / total for weight in weights]


class Calibrated:
    """A ``DecisionBackend`` whose probabilities are rescaled by ``temperature``.

    Each result's probabilities become ``softmax(log p / temperature)`` over
    the options, then the decision, confidence, and abstention are recomputed,
    so ``confidence_threshold`` applies to the calibrated confidence. Fit the
    temperature with ``fit_temperature(backend, heldout)``. A temperature above
    1 softens overconfident probabilities; below 1 sharpens them.
    """

    def __init__(self, backend: DecisionBackend, temperature: float) -> None:
        if (
            isinstance(temperature, bool)
            or not isinstance(temperature, (int, float))
            or not math.isfinite(temperature)
            or temperature <= 0
        ):
            raise ConfigError("temperature must be finite and positive")
        self.backend = backend
        self.temperature = float(temperature)

    def _calibrate(
        self,
        example: DecisionInput,
        result: DecisionResult,
        confidence_threshold: float | None,
    ) -> DecisionResult:
        probabilities = [result.probabilities[option] for option in example.options]
        return DecisionResult.from_probabilities(
            example,
            scale_probabilities(probabilities, self.temperature),
            latency_ms=result.latency_ms,
            confidence_threshold=confidence_threshold,
        )

    def decide(
        self, example: DecisionInput, *, confidence_threshold: float | None = None
    ) -> DecisionResult:
        check_confidence_threshold(confidence_threshold)
        return self._calibrate(example, self.backend.decide(example), confidence_threshold)

    def batch_decide(
        self, examples: Sequence[DecisionInput], *, confidence_threshold: float | None = None
    ) -> list[DecisionResult]:
        check_confidence_threshold(confidence_threshold)
        results = self.backend.batch_decide(examples)
        return [
            self._calibrate(example, result, confidence_threshold)
            for example, result in zip(examples, results, strict=True)
        ]

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

    def close(self) -> None:
        """Close the wrapped backend."""
        self.backend.close()
