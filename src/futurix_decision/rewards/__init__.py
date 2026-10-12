from typing import TYPE_CHECKING

from futurix_decision._lazy import lazy_exports

if TYPE_CHECKING:
    from futurix_decision.rewards.base import ProperScoringRule
    from futurix_decision.rewards.brier import BrierScore
    from futurix_decision.rewards.log_score import LogScore
    from futurix_decision.rewards.ranked_probability import RankedProbabilityScore
    from futurix_decision.rewards.spherical import SphericalScore

__getattr__, __dir__ = lazy_exports(
    __name__,
    {
        "ProperScoringRule": "futurix_decision.rewards.base",
        "BrierScore": "futurix_decision.rewards.brier",
        "LogScore": "futurix_decision.rewards.log_score",
        "RankedProbabilityScore": "futurix_decision.rewards.ranked_probability",
        "SphericalScore": "futurix_decision.rewards.spherical",
    },
)

__all__ = [
    "ProperScoringRule",
    "BrierScore",
    "LogScore",
    "RankedProbabilityScore",
    "SphericalScore",
]
