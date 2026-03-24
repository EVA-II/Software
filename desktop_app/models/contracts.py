"""Shared desktop contracts for the Bridge Assessment Workbench."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Optional


TASK_STATUSES = {"queued", "running", "completed", "failed", "cancelled"}


@dataclass
class DesktopTaskState:
    task_id: str
    kind: str
    status: str = "queued"
    progress: int = 0
    message: str = ""
    error: str = ""
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    finished_at: Optional[datetime] = None

    def update(
        self,
        *,
        status: Optional[str] = None,
        progress: Optional[int] = None,
        message: Optional[str] = None,
        error: Optional[str] = None,
    ) -> None:
        if status is not None:
            if status not in TASK_STATUSES:
                raise ValueError(f"Unsupported task status: {status}")
            self.status = status
        if progress is not None:
            self.progress = max(0, min(100, int(progress)))
        if message is not None:
            self.message = message
        if error is not None:
            self.error = error
        if self.status in {"completed", "failed", "cancelled"}:
            self.finished_at = datetime.utcnow()
        self.updated_at = datetime.utcnow()

    def as_record(self) -> dict[str, Any]:
        payload = asdict(self)
        for key in ("created_at", "updated_at", "finished_at"):
            value = payload.get(key)
            payload[key] = value.isoformat() if value else None
        return payload


@dataclass
class AppSettings:
    active_asset_version: str = "default"
    default_export_dir: str = ""
    sample_limit: int = 600
    last_dataset_dir: str = ""
    last_output_dir: str = ""
    preferred_device: str = "cpu"

    def merge(self, payload: dict[str, Any]) -> "AppSettings":
        updated = asdict(self)
        updated.update(payload or {})
        return AppSettings(**updated)


@dataclass
class AssetDescriptor:
    version: str
    title: str
    root_dir: str
    models_dir: str
    config_dir: str
    mapping_dir: str
    notes: str = ""
    model_count: int = 0
    is_active: bool = False
    created_at: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class TrainingConfig:
    dataset_path: str
    output_dir: str
    device: str = "cpu"
    train_size: float = 0.7
    val_size: float = 0.1
    test_size: float = 0.2
    batch_size: int = 16
    hidden_dim: int = 128
    num_models: int = 5
    num_epochs: int = 300
    patience: int = 80
    warmup_epochs: int = 40
    lr_unloading: float = 1e-3
    lr_other: float = 5e-4
    weight_decay: float = 1e-4
    alpha_physics: float = 1.0
    mass: float = 31600.0
    random_seed: int = 42
    trial_size: int = 200
    is_trial: bool = False
    num_workers: int = 0
    pin_memory: bool = False

    def validate(self) -> "TrainingConfig":
        path = Path(self.dataset_path)
        if not path.exists():
            raise ValueError(f"Dataset path does not exist: {path}")
        if self.batch_size <= 0:
            raise ValueError("batch_size must be > 0")
        if self.hidden_dim <= 0:
            raise ValueError("hidden_dim must be > 0")
        if self.num_models <= 0:
            raise ValueError("num_models must be > 0")
        if self.num_epochs <= 0:
            raise ValueError("num_epochs must be > 0")
        if self.patience <= 0:
            raise ValueError("patience must be > 0")
        if self.warmup_epochs < 0:
            raise ValueError("warmup_epochs must be >= 0")
        if self.warmup_epochs >= self.num_epochs:
            raise ValueError("warmup_epochs must be smaller than num_epochs")
        split_total = round(self.train_size + self.val_size + self.test_size, 6)
        if abs(split_total - 1.0) > 1e-6:
            raise ValueError("train_size + val_size + test_size must equal 1.0")
        if min(self.train_size, self.val_size, self.test_size) <= 0:
            raise ValueError("train/val/test sizes must all be > 0")
        return self

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class TrainingRunRecord:
    run_id: str
    version: str
    status: str
    output_dir: str
    models_dir: str
    reports_dir: str
    manifest_path: str
    model_paths: list[str] = field(default_factory=list)
    scaler_x_path: str = ""
    scaler_y_path: str = ""
    metrics_files: list[str] = field(default_factory=list)
    detail_files: list[str] = field(default_factory=list)
    history_files: list[str] = field(default_factory=list)
    data_split: dict[str, int] = field(default_factory=dict)
    started_at: str = ""
    finished_at: str = ""
    message: str = ""

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)
