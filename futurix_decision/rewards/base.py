"""Composable proper scoring rules; the final axis is ordered candidates."""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class ProperScoringRule(ABC):
    weight: float = 1.0

    def __post_init__(self) -> None:
        if not math.isfinite(self.weight) or self.weight < 0:
            raise ValueError("reward weight must be finite and non-negative")

    @abstractmethod
    def reward(self, probabilities: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Return a reward for each report; larger is better."""
