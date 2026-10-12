"""Decision SFT losses; operate only on the candidates available in each row."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal

import torch
import torch.nn.functional as F

from futurix_decision.datasets.schema import DecisionExample
from futurix_decision.models.decision_model import DecisionBatch

LossMode = Literal["auto", "hard", "soft"]


def _check_smoothing(label_smoothing: float) -> None:
    if not 0 <= label_smoothing < 1:
        raise ValueError("label_smoothing must be in [0, 1)")


def hard_cross_entropy(
    logits: torch.Tensor, labels: torch.Tensor, *, label_smoothing: float = 0.0
) -> torch.Tensor:
    """Integer labels over the supplied candidate logits, never over vocabulary."""
    _check_smoothing(label_smoothing)
    if logits.ndim != 2 or labels.shape != (logits.shape[0],):
        raise ValueError("expected logits [batch, candidates] and labels [batch]")
    if labels.dtype not in (torch.int8, torch.int16, torch.int32, torch.int64):
        raise ValueError("hard labels must be integer indices")
    return F.cross_entropy(logits.float(), labels.long(), label_smoothing=label_smoothing)


def soft_cross_entropy(
    logits: torch.Tensor, targets: torch.Tensor, *, label_smoothing: float = 0.0
) -> torch.Tensor:
    """Proper cross-entropy for soft labels, with optional uniform smoothing."""
    _check_smoothing(label_smoothing)
    if logits.ndim != 2 or targets.shape != logits.shape:
        raise ValueError("logits and targets must share shape [batch, candidates]")
    targets = targets.float()
    if not torch.all(torch.isfinite(targets)) or torch.any(targets < 0):
        raise ValueError("targets must be finite non-negative probabilities")
    if not torch.allclose(targets.sum(dim=-1), torch.ones_like(targets[:, 0]), atol=1e-5):
        raise ValueError("each target must sum to 1")
    if label_smoothing:
        targets = (1 - label_smoothing) * targets + label_smoothing / logits.shape[-1]
    return -(targets * F.log_softmax(logits.float(), dim=-1)).sum(dim=-1).mean()


def decision_sft_loss(
    batch: DecisionBatch,
    examples: Sequence[DecisionExample],
    *,
    mode: LossMode = "auto",
    label_smoothing: float = 0.0,
) -> torch.Tensor:
    """Gather assigned logits per row before CE; avoids the ``0 * -inf`` trap."""
    _check_smoothing(label_smoothing)
    if mode not in ("auto", "hard", "soft"):
        raise ValueError("mode must be auto, hard, or soft")
    if batch.logits.ndim != 2 or len(examples) != batch.logits.shape[0] or len(batch.slots) != len(examples):
        raise ValueError("batch size does not match examples")
    losses = []
    for row, (example, slots) in enumerate(zip(examples, batch.slots)):
        if len(slots) != len(example.options):
            raise ValueError("slots must align with the example's options")
        logits = batch.logits[row, slots].float().unsqueeze(0)
        target = example.target_distribution()
        is_soft = isinstance(example.target, list)
        if mode == "hard" and is_soft and sum(value > 0 for value in target) > 1:
            raise ValueError("hard loss requires a single target label")
        if (mode == "auto" and not is_soft) or mode == "hard":
            label = torch.tensor([target.index(max(target))], device=logits.device)
            losses.append(hard_cross_entropy(logits, label, label_smoothing=label_smoothing))
        else:
            distribution = torch.tensor([target], device=logits.device)
            losses.append(
                soft_cross_entropy(logits, distribution, label_smoothing=label_smoothing)
            )
    return torch.stack(losses).mean()
