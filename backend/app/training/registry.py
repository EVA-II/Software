"""Training asset registry skeleton."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from app.core.config import settings


@dataclass
class TrainingAssetRegistry:
    models_dir: Path = settings.models_dir
    config_dir: Path = settings.config_dir

    def expected_outputs(self) -> dict[str, str]:
        return {
            "model_weights_pattern": str(self.models_dir / "ensemble_model_{i}.pth"),
            "scaler_x": str(self.models_dir / "scaler_X.pkl"),
            "scaler_y": str(self.models_dir / "scaler_y.pkl"),
            "threshold_config": str(self.config_dir / "thresholds.json"),
        }
