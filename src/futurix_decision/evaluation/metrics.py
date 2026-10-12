"""Metrics for typed probability distributions, including soft ground truth."""

from __future__ import annotations

from collections.abc import Sequence

import torch

DEFAULT_THRESHOLDS = (0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99)


def compute_metrics(
    probabilities: Sequence[torch.Tensor],
    targets: Sequence[torch.Tensor],
    *,
    bins: int = 15,
    thresholds: Sequence[float] = DEFAULT_THRESHOLDS,
) -> dict:
    """Accuracy means expected correctness under soft targets (hard: 0/1)."""
    if not probabilities or len(probabilities) != len(targets) or bins < 1:
        raise ValueError("provide equally many nonempty probabilities and targets")
    confidences, correctness, nll, brier = [], [], [], []
    for p, y in zip(probabilities, targets):
        p, y = p.float().flatten(), y.float().flatten()
        if p.numel() != y.numel() or p.numel() < 2:
            raise ValueError("probabilities and target dimensions must match")
        if not torch.isfinite(p).all() or not torch.isfinite(y).all():
            raise ValueError("distributions must be finite")
        if (p < 0).any() or (y < 0).any():
            raise ValueError("distributions must be non-negative")
        if not torch.isclose(p.sum(), p.new_tensor(1.0), atol=1e-5) or not torch.isclose(
            y.sum(), y.new_tensor(1.0), atol=1e-5
        ):
            raise ValueError("distributions must sum to 1")
        winner = int(p.argmax())
        confidences.append(float(p[winner]))
        correctness.append(float(y[winner]))
        # Match the stabilized log-score reward for underflowed probabilities.
        nll.append(float(-(y[y > 0] * p[y > 0].clamp_min(1e-12).log()).sum()))
        brier.append(float((p - y).square().sum()))
    confidence = torch.tensor(confidences)
    correct = torch.tensor(correctness)
    gaps = []
    ece = 0.0
    for i in range(bins):
        left, right = i / bins, (i + 1) / bins
        occupied = (confidence >= left) & (
            confidence <= right if i == bins - 1 else confidence < right
        )
        if occupied.any():
            gap = abs(float(confidence[occupied].mean() - correct[occupied].mean()))
            gaps.append(gap)
            ece += float(occupied.float().mean()) * gap
    selective_accuracy, coverage = {}, {}
    for threshold in thresholds:
        if not 0 <= threshold <= 1:
            raise ValueError("thresholds must be in [0, 1]")
        kept = confidence >= threshold
        coverage[float(threshold)] = float(kept.float().mean())
        selective_accuracy[float(threshold)] = float(correct[kept].mean()) if kept.any() else None
    return {
        "accuracy": float(correct.mean()),
        "nll": sum(nll) / len(nll),
        "brier": sum(brier) / len(brier),
        "ece": ece,
        "mce": max(gaps),
        "mean_confidence": float(confidence.mean()),
        "selective_accuracy": selective_accuracy,
        "coverage": coverage,
        "count": len(nll),
    }
