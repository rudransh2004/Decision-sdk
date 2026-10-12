from __future__ import annotations

import torch

from futurix_decision.rewards.base import ProperScoringRule


class LogScore(ProperScoringRule):
    def reward(self, probabilities: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return (target * probabilities.clamp_min(1e-12).log()).sum(dim=-1)
