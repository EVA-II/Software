"""Application settings and file-system backed configuration."""

from __future__ import annotations

import json
import os
import torch  # <--- 新增这一行
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


def _load_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    with path.open("r", encoding="utf-8-sig") as handle:
        return json.load(handle)


@dataclass
class Settings:
    project_name: str = "Bridge Intelligent Assessment System"
    api_prefix: str = "/api/v1"
    model_count: int = 5
    hidden_dim: int = 128
    train_feature_dim: int = 3
    confidence_z: float = 1.96
    device: str = field(
        default_factory=lambda: os.getenv(
            "BRIDGE_DEVICE", 
            "cuda" if torch.cuda.is_available() else "cpu"
        )
    )
    host: str = field(default_factory=lambda: os.getenv("BRIDGE_HOST", "127.0.0.1"))
    port: int = field(default_factory=lambda: int(os.getenv("BRIDGE_PORT", "8000")))
    allow_origins: list[str] = field(default_factory=lambda: ["*"])

    def __post_init__(self) -> None:
        self.root_dir = Path(__file__).resolve().parents[3]
        self.backend_dir = self.root_dir / "backend"
        self.assets_dir = Path(os.getenv("BRIDGE_ASSETS_DIR", self.root_dir / "assets"))
        self.models_dir = Path(os.getenv("BRIDGE_MODELS_DIR", self.assets_dir / "models"))
        self.mapping_dir = Path(os.getenv("BRIDGE_MAPPING_DIR", self.assets_dir / "mapping"))
        self.samples_dir = self.assets_dir / "samples"
        self.config_dir = self.assets_dir / "config"

        self.thresholds_path = Path(
            os.getenv("BRIDGE_THRESHOLDS_PATH", self.config_dir / "thresholds.json")
        )
        self.feature_columns_path = Path(
            os.getenv("BRIDGE_FEATURE_COLUMNS_PATH", self.config_dir / "feature_columns.json")
        )
        self.speed_profiles_path = Path(
            os.getenv("BRIDGE_SPEED_PROFILES_PATH", self.config_dir / "speed_profiles.json")
        )
        self.asset_manifest_path = Path(
            os.getenv("BRIDGE_ASSET_MANIFEST_PATH", self.config_dir / "asset_manifest.json")
        )
        self.scaler_x_path = Path(
            os.getenv("BRIDGE_SCALER_X_PATH", self.models_dir / "scaler_X.pkl")
        )
        self.scaler_y_path = Path(
            os.getenv("BRIDGE_SCALER_Y_PATH", self.models_dir / "scaler_y.pkl")
        )
        self.mapping_model_path = Path(
            os.getenv(
                "BRIDGE_MAPPING_MODEL_PATH",
                self.mapping_dir / "bridge_track_mapping.joblib",
            )
        )
        self.allowed_extensions = {".xlsx", ".xls", ".csv", ".json"}
        self.model_output_names = [
            "Vertical_Acceleration",
            "Derailment_Coefficient",
            "Wheel_Unloading_Rate",
            "Suspension_Force",
        ]
        self.warning_output_names = [
            "Vertical_Acceleration",
            "Derailment_Coefficient",
            "Wheel_Unloading_Rate",
        ]
        self.target_column_names = [
            "vertical_acceleration",
            "derailment_coefficient",
            "wheel_unloading_rate",
            "Force_predictor",
        ]
        self.thresholds = _load_json(self.thresholds_path, {})
        self.feature_config = _load_json(self.feature_columns_path, {})
        self.speed_profiles = _load_json(self.speed_profiles_path, {})
        self.asset_manifest = _load_json(self.asset_manifest_path, {})
        self.default_red_ratio = float(self.thresholds.get("red_ratio", 1.0))
        self.default_yellow_ratio = float(self.thresholds.get("yellow_ratio", 0.8))
        self.epistemic_tau = float(self.thresholds.get("epistemic_tau", 0.25))
        self.model_version = self.asset_manifest.get("model_version", "unregistered")

    @property
    def expected_model_paths(self) -> list[Path]:
        return [
            self.models_dir / f"ensemble_model_{index}.pth"
            for index in range(1, self.model_count + 1)
        ]

    @property
    def variable_thresholds(self) -> dict[str, dict[str, Any]]:
        return self.thresholds.get("variables", {})

    @property
    def required_feature_columns(self) -> list[str]:
        columns = self.feature_config.get("feature_columns", [])
        return list(columns)

    @property
    def train_feature_names(self) -> list[str]:
        names = self.feature_config.get("train_feature_names", [])
        return list(names)

    @property
    def bridge_feature_defaults(self) -> dict[str, float]:
        return {
            key: float(value)
            for key, value in self.feature_config.get("bridge_feature_defaults", {}).items()
        }

    @property
    def mapping_feature_name(self) -> str:
        return str(self.feature_config.get("mapping_feature_name", "mapped_track_irregularity"))


settings = Settings()


