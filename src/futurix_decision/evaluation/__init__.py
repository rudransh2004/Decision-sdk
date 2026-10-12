from typing import TYPE_CHECKING

from futurix_decision._lazy import lazy_exports
from futurix_decision.evaluation.metrics import DEFAULT_THRESHOLDS, compute_metrics

if TYPE_CHECKING:
    from futurix_decision.evaluation.calibration import (
        fit_temperature,
        fit_temperature_from_logits,
    )
    from futurix_decision.evaluation.evaluator import evaluate

__getattr__ = lazy_exports(
    __name__,
    {
        "evaluate": "futurix_decision.evaluation.evaluator",
        "fit_temperature": "futurix_decision.evaluation.calibration",
        "fit_temperature_from_logits": "futurix_decision.evaluation.calibration",
    },
)

__all__ = [
    "DEFAULT_THRESHOLDS",
    "compute_metrics",
    "evaluate",
    "fit_temperature",
    "fit_temperature_from_logits",
]
