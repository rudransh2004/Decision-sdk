from typing import TYPE_CHECKING

from futurix_decision._lazy import lazy_exports

if TYPE_CHECKING:
    from futurix_decision.trainers.base import DecisionTrainer
    from futurix_decision.trainers.calibration import CalibrationConfig, CalibrationTrainer
    from futurix_decision.trainers.losses import (
        decision_sft_loss,
        hard_cross_entropy,
        soft_cross_entropy,
    )
    from futurix_decision.trainers.native import NativeSFTConfig
    from futurix_decision.trainers.rlcd import RLCDConfig, RLCDTrainer
    from futurix_decision.trainers.sft import SFTConfig, SFTTrainer

__getattr__, __dir__ = lazy_exports(
    __name__,
    {
        "DecisionTrainer": "futurix_decision.trainers.base",
        "CalibrationConfig": "futurix_decision.trainers.calibration",
        "CalibrationTrainer": "futurix_decision.trainers.calibration",
        "decision_sft_loss": "futurix_decision.trainers.losses",
        "hard_cross_entropy": "futurix_decision.trainers.losses",
        "soft_cross_entropy": "futurix_decision.trainers.losses",
        "NativeSFTConfig": "futurix_decision.trainers.native",
        "RLCDConfig": "futurix_decision.trainers.rlcd",
        "RLCDTrainer": "futurix_decision.trainers.rlcd",
        "SFTConfig": "futurix_decision.trainers.sft",
        "SFTTrainer": "futurix_decision.trainers.sft",
    },
)

__all__ = [
    "DecisionTrainer",
    "CalibrationConfig",
    "CalibrationTrainer",
    "RLCDConfig",
    "RLCDTrainer",
    "SFTConfig",
    "SFTTrainer",
    "NativeSFTConfig",
    "decision_sft_loss",
    "hard_cross_entropy",
    "soft_cross_entropy",
]
