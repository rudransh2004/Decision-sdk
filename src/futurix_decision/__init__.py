"""Futurix Decision SDK."""

from futurix_decision.datasets import DecisionExample, DecisionInput, load_decisions, load_open_decisions
from futurix_decision.evaluation import evaluate, fit_temperature
from futurix_decision.inference import DecisionResult
from futurix_decision.models import (
    DecisionModel, NativeCapabilities, NativeDecisionModel, PointerDecisionHead,
    load_model, register_decision_head,
)
from futurix_decision.trainers import (
    CalibrationConfig,
    CalibrationTrainer,
    DecisionTrainer,
    NativeSFTConfig,
    RLCDConfig,
    RLCDTrainer,
    SFTConfig,
    SFTTrainer,
)

__version__ = "0.1.0"

__all__ = [
    "DecisionExample",
    "DecisionInput",
    "DecisionModel",
    "NativeCapabilities",
    "NativeDecisionModel",
    "PointerDecisionHead",
    "DecisionResult",
    "DecisionTrainer",
    "NativeSFTConfig",
    "CalibrationConfig",
    "CalibrationTrainer",
    "RLCDConfig",
    "RLCDTrainer",
    "SFTConfig",
    "SFTTrainer",
    "evaluate",
    "fit_temperature",
    "load_decisions",
    "load_model",
    "load_open_decisions",
    "register_decision_head",
]
