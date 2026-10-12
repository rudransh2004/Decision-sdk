from __future__ import annotations

import torch

from futurix_decision.rewards.base import ProperScoringRule


class BrierScore(ProperScoringRule):
    def reward(self, probabilities: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return -(probabilities - target).square().sum(dim=-1)
