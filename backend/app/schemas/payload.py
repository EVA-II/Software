"""Input and output schemas for prediction endpoints."""

from __future__ import annotations

from datetime import datetime
from typing import Dict, List, Literal, Optional, Union

from pydantic import BaseModel, Field, model_validator


ScalarMeta = Union[str, float, int, bool]


class MeasurementRecord(BaseModel):
    position: float
    deflection: float
    weight: float = 1.0


class FeatureRecord(BaseModel):
    position: float
    features: Dict[str, float]
    ground_truth: Optional[Dict[str, float]] = None


class PredictJSONRequest(BaseModel):
    scenario_id: str = Field(..., description="Logical scenario or batch identifier.")
    input_mode: Literal["engineered_features", "raw_measurements"] = "engineered_features"
    speed_level: Optional[str] = None
    train_features: Optional[List[float]] = Field(default=None, min_length=3, max_length=3)
    measurements: Optional[List[MeasurementRecord]] = None
    feature_records: Optional[List[FeatureRecord]] = None
    metadata: Dict[str, Union[str, float, int]] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_payload_shape(self) -> "PredictJSONRequest":
        if self.input_mode == "raw_measurements" and not self.measurements:
            raise ValueError("raw_measurements 模式下必须提供 measurements。")
        if self.input_mode == "engineered_features" and not self.feature_records:
            raise ValueError("engineered_features 模式下必须提供 feature_records。")
        return self


class PredictionSeries(BaseModel):
    name: str
    unit: str
    positions: List[float]
    mean: List[float]
    lower_95: List[float]
    upper_95: List[float]
    aleatoric_var: List[float]
    epistemic_var: List[float]
    total_var: List[float]
    ground_truth: Optional[List[float]] = None


class PredictionStatus(BaseModel):
    code: int
    level: str
    label: str
    color: str
    reason: str
    dominant_variable: Optional[str] = None
    threshold_ratio: float = 0.0
    epistemic_peak: float = 0.0


class PredictionMeta(BaseModel):
    scenario_id: str
    input_mode: str
    source: str
    model_version: str
    generated_at: datetime
    node_count: int
    model_count: int
    asset_ready: bool
    extra: Dict[str, ScalarMeta] = Field(default_factory=dict)


class PredictionResponse(BaseModel):
    status: PredictionStatus
    meta: PredictionMeta
    series: List[PredictionSeries]


class PredictionJobAcceptedResponse(BaseModel):
    job_id: str
    status: Literal["queued", "running", "completed", "failed"]
    progress: int
    message: str
    poll_url: str
    created_at: datetime


class PredictionJobStatusResponse(BaseModel):
    job_id: str
    status: Literal["queued", "running", "completed", "failed"]
    progress: int
    message: str
    created_at: datetime
    updated_at: datetime
    file_name: str
    scenario_id: Optional[str] = None
    error: Optional[str] = None
    result: Optional[PredictionResponse] = None


class HealthResponse(BaseModel):
    status: str
    model_version: str
    model_count: int
    loaded_model_count: int
    asset_ready: bool
    device: str
    missing_assets: List[str]


class ModelRegistryResponse(BaseModel):
    model_version: str
    required_model_count: int
    loaded_model_count: int
    scaler_x_loaded: bool
    scaler_y_loaded: bool
    mapping_model_loaded: bool
    expected_weight_paths: List[str]
    missing_assets: List[str]
