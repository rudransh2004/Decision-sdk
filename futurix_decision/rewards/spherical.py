from __future__ import annotations

import torch

from futurix_decision.rewards.base import ProperScoringRule


class SphericalScore(ProperScoringRule):
    def reward(self, probabilities: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return (probabilities * target).sum(dim=-1) / probabilities.norm(dim=-1).clamp_min(1e-12)
