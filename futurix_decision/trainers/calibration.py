"""Probability-quality training followed by held-out temperature fitting."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
import math

import torch
import torch.nn.functional as F

from futurix_decision.datasets import DecisionExample
from futurix_decision.evaluation.calibration import fit_temperature
from futurix_decision.models import DecisionBatch, DecisionModel
from futurix_decision.trainers.sft import DatasetSource, SFTConfig, SFTTrainer


@dataclass(frozen=True)
class CalibrationConfig(SFTConfig):
    ce_weight: float = 1.0  # Cross-entropy is negative expected log score.
    brier_weight: float = 0.2

    def __post_init__(self) -> None:
        super().__post_init__()
        if any(
            not math.isfinite(value) or value < 0
            for value in (self.ce_weight, self.brier_weight)
        ) or self.ce_weight + self.brier_weight <= 0:
            raise ValueError("calibration loss weights must be non-negative and nonzero in total")


class CalibrationTrainer(SFTTrainer):
    def __init__(self, model: DecisionModel, *, config: CalibrationConfig | None = None) -> None:
        super().__init__(model, config=config or CalibrationConfig())
        self.config: CalibrationConfig

    def compute_loss(
        self, batch: DecisionBatch, examples: Sequence[DecisionExample]
    ) -> torch.Tensor:
        losses = []
        for row, (example, slots) in enumerate(zip(examples, batch.slots)):
            logits = batch.logits[row, slots].float()
            target = torch.tensor(example.target_distribution(), device=logits.device)
            if self.config.label_smoothing:
                target = (1 - self.config.label_smoothing) * target + self.config.label_smoothing / len(slots)
            p = torch.softmax(logits, dim=-1)
            ce = -(target * F.log_softmax(logits, dim=-1)).sum()
            brier = (p - target).square().sum()
            losses.append(self.config.ce_weight * ce + self.config.brier_weight * brier)
        return torch.stack(losses).mean()

    def train(
        self, dataset: DatasetSource, *, calibration_dataset: DatasetSource
    ) -> list[dict[str, float | int]]:
        history = super().train(dataset)
        fit_temperature(self.model, calibration_dataset, batch_size=self.config.batch_size)
        return history
