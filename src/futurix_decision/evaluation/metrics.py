"""Metrics for typed probability distributions, including soft ground truth."""

from __future__ import annotations

import math
from collections.abc import Sequence

from futurix_decision.errors import ConfigError

DEFAULT_THRESHOLDS = (0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99)


def _mean(values: Sequence[float]) -> float:
    return sum(values) / len(values)


def compute_metrics(
    probabilities: Sequence[Sequence[float]],
    targets: Sequence[Sequence[float]],
    *,
    bins: int = 15,
    thresholds: Sequence[float] = DEFAULT_THRESHOLDS,
) -> dict:
    """Accuracy means expected correctness under soft targets (hard: 0/1)."""
    if not probabilities or len(probabilities) != len(targets) or bins < 1:
        raise ConfigError("provide equally many nonempty probabilities and targets")
    confidence, correct, nll, brier = [], [], [], []
    for raw_p, raw_y in zip(probabilities, targets, strict=True):
        p, y = [float(v) for v in raw_p], [float(v) for v in raw_y]
        if len(p) != len(y) or len(p) < 2:
            raise ConfigError("probabilities and target dimensions must match")
        if not all(map(math.isfinite, p + y)):
            raise ConfigError("distributions must be finite")
        if min(p) < 0 or min(y) < 0:
            raise ConfigError("distributions must be non-negative")
        if not math.isclose(sum(p), 1.0, rel_tol=0, abs_tol=1e-5) or not math.isclose(
            sum(y), 1.0, rel_tol=0, abs_tol=1e-5
        ):
            raise ConfigError("distributions must sum to 1")
        winner = max(range(len(p)), key=p.__getitem__)
        confidence.append(p[winner])
        correct.append(y[winner])
        # Match the stabilized log-score reward for underflowed probabilities.
        nll.append(-sum(t * math.log(max(q, 1e-12)) for q, t in zip(p, y, strict=True) if t > 0))
        brier.append(sum((q - t) ** 2 for q, t in zip(p, y, strict=True)))
    count = len(confidence)
    gaps = []
    ece = 0.0
    for i in range(bins):
        left, right = i / bins, (i + 1) / bins
        occupied = [
            j
            for j, c in enumerate(confidence)
            if c >= left and (c <= right if i == bins - 1 else c < right)
        ]
        if occupied:
            gap = abs(
                _mean([confidence[j] for j in occupied]) - _mean([correct[j] for j in occupied])
            )
            gaps.append(gap)
            ece += len(occupied) / count * gap
    selective_accuracy: dict[float, float | None] = {}
    coverage: dict[float, float] = {}
    for threshold in thresholds:
        if not 0 <= threshold <= 1:
            raise ConfigError("thresholds must be in [0, 1]")
        kept = [y for c, y in zip(confidence, correct, strict=True) if c >= threshold]
        coverage[float(threshold)] = len(kept) / count
        selective_accuracy[float(threshold)] = _mean(kept) if kept else None
    return {
        "accuracy": _mean(correct),
        "nll": _mean(nll),
        "brier": _mean(brier),
        "ece": ece,
        "mce": max(gaps),
        "mean_confidence": _mean(confidence),
        "selective_accuracy": selective_accuracy,
        "coverage": coverage,
        "count": count,
    }
