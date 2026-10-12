"""Futurix Decision SDK.

The data schemas and loaders import without torch. Local models, trainers, and
evaluation load on first access and need ``futurix-decision[torch]``.
"""

__version__ = "0.2.0"

import logging
from typing import TYPE_CHECKING

from futurix_decision._lazy import lazy_exports
from futurix_decision.client import AsyncDecisionClient, DecisionClient
from futurix_decision.datasets import (
    DecisionExample,
    DecisionInput,
    load_decisions,
    load_open_decisions,
)
from futurix_decision.errors import (
    BackendError,
    CheckpointError,
    ConfigError,
    FuturixError,
    InvalidOutputError,
    MissingDependencyError,
)
from futurix_decision.inference import DecisionResult
from futurix_decision.models import Calibrated, DecisionBackend, FamilyAdapter, register_family

logging.getLogger(__name__).addHandler(logging.NullHandler())

if TYPE_CHECKING:
    from futurix_decision.evaluation import evaluate, fit_temperature
    from futurix_decision.models import (
        DecisionModel,
        DecisionModelConfig,
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

__getattr__, __dir__ = lazy_exports(
    __name__,
    {
        "evaluate": "futurix_decision.evaluation",
        "fit_temperature": "futurix_decision.evaluation",
        "DecisionModel": "futurix_decision.models",
        "DecisionModelConfig": "futurix_decision.models",
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
    "AsyncDecisionClient",
    "Calibrated",
    "DecisionBackend",
    "BackendError",
    "CheckpointError",
    "ConfigError",
    "DecisionClient",
    "InvalidOutputError",
    "DecisionExample",
    "DecisionInput",
    "DecisionModel",
    "DecisionModelConfig",
    "FamilyAdapter",
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
    "register_family",
]
