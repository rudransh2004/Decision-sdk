from typing import TYPE_CHECKING

from futurix_decision._lazy import lazy_exports
from futurix_decision.models.calibrated import Calibrated
from futurix_decision.models.protocol import DecisionBackend

if TYPE_CHECKING:
    from futurix_decision.models.custom_head import PointerDecisionHead, register_decision_head
    from futurix_decision.models.decision_model import DecisionBatch, DecisionModel
    from futurix_decision.models.native import NativeCapabilities, NativeDecisionModel, load_model

__getattr__ = lazy_exports(
    __name__,
    {
        "PointerDecisionHead": "futurix_decision.models.custom_head",
        "register_decision_head": "futurix_decision.models.custom_head",
        "DecisionBatch": "futurix_decision.models.decision_model",
        "DecisionModel": "futurix_decision.models.decision_model",
        "NativeCapabilities": "futurix_decision.models.native",
        "NativeDecisionModel": "futurix_decision.models.native",
        "load_model": "futurix_decision.models.native",
    },
)

__all__ = [
    "Calibrated",
    "DecisionBackend",
    "DecisionBatch",
    "DecisionModel",
    "NativeCapabilities",
    "NativeDecisionModel",
    "PointerDecisionHead",
    "load_model",
    "register_decision_head",
]
