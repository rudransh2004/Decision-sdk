"""Held-out scalar temperature fitting for decision logits."""

from __future__ import annotations

import math
from collections.abc import Sequence

import torch
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence

from futurix_decision.evaluation.evaluator import EvaluationSource, example_batches
from futurix_decision.models import DecisionModel


def fit_temperature_from_logits(
    logits: Sequence[torch.Tensor],
    targets: Sequence[torch.Tensor],
    *,
    max_iter: int = 60,
) -> float:
    """Minimize held-out soft-label NLL in log-temperature space on CPU."""
    if not logits or len(logits) != len(targets) or max_iter < 1:
        raise ValueError("provide matching nonempty logits and targets")
    for z, y in zip(logits, targets):
        if z.ndim != 1 or z.shape != y.shape or z.numel() < 2:
            raise ValueError("each logit vector and target must have matching candidate dimensions")
        if not torch.isfinite(z).all() or not torch.isfinite(y).all() or (y < 0).any():
            raise ValueError("logits must be finite and targets non-negative finite")
        if not torch.isclose(y.float().sum(), torch.tensor(1.0), atol=1e-5):
            raise ValueError("targets must sum to 1")
    padded_logits = pad_sequence([z.detach().float().cpu() for z in logits], batch_first=True, padding_value=-1e9)
    padded_targets = pad_sequence([y.detach().float().cpu() for y in targets], batch_first=True)
    log_t = torch.zeros((), requires_grad=True)
    optimizer = torch.optim.LBFGS([log_t], lr=0.25, max_iter=max_iter, line_search_fn="strong_wolfe")

    def closure() -> torch.Tensor:
        optimizer.zero_grad()
        temperature = log_t.exp().clamp(min=0.05, max=20.0)
        loss = -(padded_targets * F.log_softmax(padded_logits / temperature, dim=-1)).sum(dim=-1).mean()
        loss.backward()
        return loss

    optimizer.step(closure)
    temperature = float(log_t.detach().exp().clamp(min=0.05, max=20.0))
    if not math.isfinite(temperature):
        raise ValueError("temperature optimization diverged")
    return temperature


def fit_temperature(
    model: DecisionModel, calibration_dataset: EvaluationSource, *, batch_size: int = 16
) -> float:
    """Fit and store T from a separate, held-out calibration dataset."""
    logits, targets = [], []
    model.backbone.eval()
    if model.decision_head is not None:
        model.decision_head.eval()
    with torch.inference_mode():
        for examples in example_batches(calibration_dataset, batch_size):
            batch = model.forward_decisions(examples)
            for i, (row, slots) in enumerate(zip(examples, batch.slots)):
                logits.append(batch.logits[i, slots].float().cpu())
                targets.append(torch.tensor(row.target_distribution(), dtype=torch.float32))
    model.temperature = fit_temperature_from_logits(logits, targets)
    return model.temperature
