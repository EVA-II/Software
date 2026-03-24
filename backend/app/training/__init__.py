"""Training services and evaluation helpers for desktop retraining support."""

from .evaluation import build_metrics_report
from .registry import TrainingAssetRegistry
from .service import TrainingCallbacks, TrainingCancelledError, TrainingConfig, TrainingRunRecord, TrainingService

__all__ = [
    "TrainingAssetRegistry",
    "TrainingCallbacks",
    "TrainingCancelledError",
    "TrainingConfig",
    "TrainingRunRecord",
    "TrainingService",
    "build_metrics_report",
]
