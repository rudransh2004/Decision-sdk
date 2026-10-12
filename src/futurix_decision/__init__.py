"""Futurix Decision SDK.

The data schemas and loaders import without torch. Local models, trainers, and
evaluation load on first access and need ``futurix-decision[torch]``.
"""

__version__ = "0.1.0"

from typing import TYPE_CHECKING

from futurix_decision._lazy import lazy_exports
from futurix_decision.datasets import (
    DecisionExample,
    DecisionInput,
    load_decisions,
    load_open_decisions,
)
from futurix_decision.errors import FuturixError, MissingDependencyError
from futurix_decision.inference import DecisionResult

if TYPE_CHECKING:
    from futurix_decision.evaluation import evaluate, fit_temperature
    from futurix_decision.models import (
        DecisionModel,
        NativeCapabilities,
        NativeDecisionModel,
        PointerDecisionHead,
        load_model,
        register_decision_head,
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

__getattr__ = lazy_exports(
    __name__,
    {
        "evaluate": "futurix_decision.evaluation",
        "fit_temperature": "futurix_decision.evaluation",
        "DecisionModel": "futurix_decision.models",
        "NativeCapabilities": "futurix_decision.models",
        "NativeDecisionModel": "futurix_decision.models",
        "PointerDecisionHead": "futurix_decision.models",
        "load_model": "futurix_decision.models",
        "register_decision_head": "futurix_decision.models",
        "CalibrationConfig": "futurix_decision.trainers",
        "CalibrationTrainer": "futurix_decision.trainers",
        "DecisionTrainer": "futurix_decision.trainers",
        "NativeSFTConfig": "futurix_decision.trainers",
        "RLCDConfig": "futurix_decision.trainers",
        "RLCDTrainer": "futurix_decision.trainers",
        "SFTConfig": "futurix_decision.trainers",
        "SFTTrainer": "futurix_decision.trainers",
    },
)

__all__ = [
    "DecisionExample",
    "DecisionInput",
    "DecisionModel",
    "NativeCapabilities",
    "NativeDecisionModel",
    "PointerDecisionHead",
    "DecisionResult",
    "DecisionTrainer",
    "FuturixError",
    "MissingDependencyError",
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
