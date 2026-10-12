from __future__ import annotations

import torch

from futurix_decision.rewards.base import ProperScoringRule


class RankedProbabilityScore(ProperScoringRule):
    """Ordinal reward, excluding the final CDF entry (always one)."""

    def reward(self, probabilities: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return (
            -(probabilities.cumsum(dim=-1)[..., :-1] - target.cumsum(dim=-1)[..., :-1])
            .square()
            .sum(dim=-1)
        )
