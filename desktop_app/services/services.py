"""Desktop runtime services for the Bridge Assessment Workbench with i18n support."""

from __future__ import annotations

import json
import os
import shutil
import sqlite3
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any, Optional

import pandas as pd

from desktop_app.models.contracts import AppSettings, AssetDescriptor, DesktopTaskState, TrainingConfig


def load_json_file(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    with path.open("r", encoding="utf-8-sig") as handle:
        return json.load(handle)


class AppPaths:
    def __init__(self) -> None:
        if getattr(sys, "frozen", False):
            install_dir = Path(sys.executable).resolve().parent
            base_dir = Path(os.getenv("LOCALAPPDATA", install_dir / ".local")) / "BridgeAssessment"
        else:
            install_dir = Path(__file__).resolve().parents[2]
            base_dir = Path(os.getenv("BRIDGE_WORKSPACE_DIR", install_dir / ".bridge_assessment_runtime"))
        self.install_dir = install_dir
        self.workspace_dir = base_dir
        self.logs_dir = base_dir / "logs"
        self.exports_dir = base_dir / "exports"
        self.asset_library_dir = base_dir / "asset_library"
        self.active_assets_dir = base_dir / "active_assets"
        self.tasks_dir = base_dir / "tasks"
        self.training_dir = base_dir / "training_runs"
        self.db_path = base_dir / "workbench.sqlite3"

    def ensure(self) -> None:
        for directory in (
            self.workspace_dir,
            self.logs_dir,
            self.exports_dir,
            self.asset_library_dir,
            self.active_assets_dir,
            self.tasks_dir,
            self.training_dir,
        ):
            directory.mkdir(parents=True, exist_ok=True)


class StorageService:
    def __init__(self, db_path: Path) -> None:
        self.db_path = db_path
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.db_path)
        connection.row_factory = sqlite3.Row
        return connection

    def _init_db(self) -> None:
        with self._connect() as connection:
            connection.execute(
                "CREATE TABLE IF NOT EXISTS app_settings (key TEXT PRIMARY KEY, value TEXT NOT NULL)"
            )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS task_history (
                    task_id TEXT PRIMARY KEY,
                    kind TEXT NOT NULL,
                    status TEXT NOT NULL,
                    progress INTEGER NOT NULL,
                    message TEXT,
                    error TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    finished_at TEXT,
                    payload TEXT
                )
                """
            )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS assets (
                    version TEXT PRIMARY KEY,
                    title TEXT NOT NULL,
                    root_dir TEXT NOT NULL,
                    models_dir TEXT NOT NULL,
                    config_dir TEXT NOT NULL,
                    mapping_dir TEXT NOT NULL,
                    notes TEXT,
                    model_count INTEGER NOT NULL,
                    is_active INTEGER NOT NULL,
                    created_at TEXT,
                    metadata TEXT
                )
                """
            )

    def load_settings(self) -> AppSettings:
        with self._connect() as connection:
            row = connection.execute("SELECT value FROM app_settings WHERE key = 'app_settings'").fetchone()
        payload = json.loads(row["value"]) if row else {}
        
        # Filter out fields that are not in AppSettings
        from dataclasses import fields
        valid_fields = {f.name for f in fields(AppSettings)}
        filtered_payload = {k: v for k, v in payload.items() if k in valid_fields}
        
        return AppSettings(**filtered_payload) if filtered_payload else AppSettings()

    def save_settings(self, settings_obj: AppSettings) -> None:
        payload = json.dumps(asdict(settings_obj), ensure_ascii=False)
        with self._connect() as connection:
            connection.execute(
                "REPLACE INTO app_settings (key, value) VALUES (?, ?)",
                ("app_settings", payload),
            )

    def save_task(self, task: DesktopTaskState, payload: Optional[dict[str, Any]] = None) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                REPLACE INTO task_history (
                    task_id, kind, status, progress, message, error, created_at, updated_at, finished_at, payload
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    task.task_id,
                    task.kind,
                    task.status,
                    task.progress,
                    task.message,
                    task.error,
                    task.created_at.isoformat(),
                    task.updated_at.isoformat(),
                    task.finished_at.isoformat() if task.finished_at else None,
                    json.dumps(payload, ensure_ascii=False) if payload is not None else None,
                ),
            )

    def recent_tasks(self, limit: int = 20) -> list[dict[str, Any]]:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT * FROM task_history ORDER BY updated_at DESC LIMIT ?",
                (limit,),
            ).fetchall()
        return [dict(row) for row in rows]

    def upsert_asset(self, descriptor: AssetDescriptor) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                REPLACE INTO assets (
                    version, title, root_dir, models_dir, config_dir, mapping_dir, notes, model_count, is_active, created_at, metadata
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    descriptor.version,
                    descriptor.title,
                    descriptor.root_dir,
                    descriptor.models_dir,
                    descriptor.config_dir,
                    descriptor.mapping_dir,
                    descriptor.notes,
                    descriptor.model_count,
                    1 if descriptor.is_active else 0,
                    descriptor.created_at,
                    json.dumps(descriptor.metadata, ensure_ascii=False),
                ),
            )


class BackendSettingsBridge:
    @staticmethod
    def configure(asset_root: Path) -> None:
        from app.core.config import _load_json, settings

        settings.assets_dir = asset_root
        settings.models_dir = asset_root / "models"
        settings.mapping_dir = asset_root / "mapping"
        settings.samples_dir = asset_root / "samples"
        settings.config_dir = asset_root / "config"
        settings.thresholds_path = settings.config_dir / "thresholds.json"
        settings.feature_columns_path = settings.config_dir / "feature_columns.json"
        settings.speed_profiles_path = settings.config_dir / "speed_profiles.json"
        settings.asset_manifest_path = settings.config_dir / "asset_manifest.json"
        settings.scaler_x_path = settings.models_dir / "scaler_X.pkl"
        settings.scaler_y_path = settings.models_dir / "scaler_y.pkl"
        settings.mapping_model_path = settings.mapping_dir / "bridge_track_mapping.joblib"
        settings.thresholds = _load_json(settings.thresholds_path, {})
        settings.feature_config = _load_json(settings.feature_columns_path, {})
        settings.speed_profiles = _load_json(settings.speed_profiles_path, {})
        settings.asset_manifest = _load_json(settings.asset_manifest_path, {})
        settings.default_red_ratio = float(settings.thresholds.get("red_ratio", 1.0))
        settings.default_yellow_ratio = float(settings.thresholds.get("yellow_ratio", 0.8))
        settings.epistemic_tau = float(settings.thresholds.get("epistemic_tau", 0.25))
        settings.model_version = settings.asset_manifest.get("model_version", "unregistered")


class AssetService:
    def __init__(self, paths: AppPaths, storage: StorageService) -> None:
        self.paths = paths
        self.storage = storage
        self.install_assets_dir = paths.install_dir / "assets"
        self.bootstrap()

    def bootstrap(self) -> None:
        self.paths.ensure()
        manifest = load_json_file(self.install_assets_dir / "config" / "asset_manifest.json", {})
        default_version = manifest.get("model_version", "default")
        default_target = self.paths.asset_library_dir / default_version
        if self.install_assets_dir.exists() and not default_target.exists():
            shutil.copytree(self.install_assets_dir, default_target)
        settings_obj = self.storage.load_settings()
        active_version = settings_obj.active_asset_version or default_version
        if not (self.paths.asset_library_dir / active_version).exists():
            active_version = default_version
        self.activate(active_version)
        self.scan()

    def scan(self) -> list[AssetDescriptor]:
        settings_obj = self.storage.load_settings()
        descriptors = []
        for candidate in sorted(self.paths.asset_library_dir.iterdir(), key=lambda item: item.name, reverse=True):
            if not candidate.is_dir():
                continue
            descriptor = self._descriptor_from_dir(candidate, candidate.name == settings_obj.active_asset_version)
            if descriptor is None:
                continue
            descriptors.append(descriptor)
            self.storage.upsert_asset(descriptor)
        return descriptors

    def activate(self, version: str) -> AssetDescriptor:
        source = self.paths.asset_library_dir / version
        if not source.exists():
            raise FileNotFoundError(f"Unknown asset version: {version}")
        if self.paths.active_assets_dir.exists():
            shutil.rmtree(self.paths.active_assets_dir)
        shutil.copytree(source, self.paths.active_assets_dir)
        settings_obj = self.storage.load_settings().merge({"active_asset_version": version})
        self.storage.save_settings(settings_obj)
        descriptors = self.scan()
        for descriptor in descriptors:
            if descriptor.version == version:
                return descriptor
        raise FileNotFoundError(f"Activated asset version was not indexed: {version}")

    def register(self, run_output: str | Path) -> AssetDescriptor:
        source = Path(run_output).resolve()
        manifest = load_json_file(source / "config" / "asset_manifest.json", {})
        version = manifest.get("model_version", source.name)
        target = self.paths.asset_library_dir / version
        if target.exists():
            shutil.rmtree(target)
        shutil.copytree(source, target)
        descriptor = self._descriptor_from_dir(target, False)
        if descriptor is None:
            raise FileNotFoundError(f"Training output does not look like an asset package: {source}")
        self.storage.upsert_asset(descriptor)
        return descriptor

    def active_descriptor(self) -> Optional[AssetDescriptor]:
        active_version = self.storage.load_settings().active_asset_version
        for descriptor in self.scan():
            if descriptor.version == active_version:
                return descriptor
        return None

    def _descriptor_from_dir(self, root_dir: Path, is_active: bool) -> Optional[AssetDescriptor]:
        config_dir = root_dir / "config"
        models_dir = root_dir / "models"
        mapping_dir = root_dir / "mapping"
        manifest_path = config_dir / "asset_manifest.json"
        if not config_dir.exists() or not models_dir.exists() or not manifest_path.exists():
            return None
        manifest = load_json_file(manifest_path, {})
        version = manifest.get("model_version", root_dir.name)
        return AssetDescriptor(
            version=version,
            title=manifest.get("model_version", root_dir.name),
            root_dir=str(root_dir),
            models_dir=str(models_dir),
            config_dir=str(config_dir),
            mapping_dir=str(mapping_dir),
            notes=manifest.get("notes", ""),
            model_count=len(list(models_dir.glob("ensemble_model_*.pth"))),
            is_active=is_active,
            created_at=manifest.get("generated_at", ""),
            metadata=manifest,
        )


class InferenceFacade:
    def __init__(self, asset_service: AssetService) -> None:
        self.asset_service = asset_service
        self._service: Optional[Any] = None
        self._loaded_version: str = ""

    def health(self) -> dict[str, Any]:
        return self._ensure_service().health()

    def predict_file(
        self,
        file_path: str,
        scenario_id: str | None = None,
        speed_level: str | None = None,
        train_features: list[float] | None = None,
    ) -> Any:
        path = Path(file_path)
        content = path.read_bytes()
        return self._ensure_service().predict_from_upload(
            filename=path.name,
            content=content,
            scenario_id=scenario_id,
            speed_level=speed_level,
            train_features=train_features,
        )

    def predict_json(self, payload: dict[str, Any]) -> Any:
        return self._ensure_service().predict_from_json(payload)

    def reload(self) -> None:
        self._service = None
        self._loaded_version = ""

    def _ensure_service(self) -> Any:
        from app.services.job_inference import ModelInferenceService

        active = self.asset_service.active_descriptor()
        if active is None:
            raise FileNotFoundError("No active asset version is available.")
        if self._service is None or self._loaded_version != active.version:
            BackendSettingsBridge.configure(self.asset_service.paths.active_assets_dir)
            self._service = ModelInferenceService()
            self._service.load_assets()
            self._loaded_version = active.version
        return self._service


class TrainingFacade:
    def __init__(self, asset_service: AssetService, paths: AppPaths) -> None:
        self.asset_service = asset_service
        self.paths = paths

    def start(self, config: TrainingConfig, callbacks: Optional[dict[str, Any]] = None) -> Any:
        from app.training import TrainingCallbacks as BackendTrainingCallbacks
        from app.training import TrainingConfig as BackendTrainingConfig
        from app.training import TrainingService

        backend_config = BackendTrainingConfig(**config.validate().as_dict())
        callback_object = None
        if callbacks is not None:
            callback_object = BackendTrainingCallbacks(**callbacks)
        record = TrainingService().start(backend_config, callback_object)
        self.asset_service.register(record.output_dir)
        return record


def sample_series_points(series: dict[str, Any], limit: int) -> list[dict[str, Any]]:
    total = len(series.get("positions", []))
    if total <= limit:
        indices = range(total)
    else:
        step = max(total / float(limit), 1.0)
        indices = [min(int(round(index * step)), total - 1) for index in range(limit)]
        indices = list(dict.fromkeys(indices))
    points = []
    ground_truth = series.get("ground_truth") or []
    for index in indices:
        point = {
            "x": float(series["positions"][index]),
            "mean": float(series["mean"][index]),
            "lower": float(series["lower_95"][index]),
            "upper": float(series["upper_95"][index]),
        }
        if ground_truth:
            point["truth"] = float(ground_truth[index])
        points.append(point)
    return points


def build_qml_prediction_payload(response: Any, sample_limit: int) -> dict[str, Any]:
    payload = response.model_dump(mode="json")
    for series in payload["series"]:
        series["points"] = sample_series_points(series, sample_limit)
    return payload


def write_prediction_json(response: Any, target_path: str | Path) -> Path:
    path = Path(target_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(response.model_dump_json(indent=2), encoding="utf-8")
    return path


def write_prediction_csv(response: Any, target_path: str | Path) -> Path:
    payload = response.model_dump(mode="json")
    rows = []
    for series in payload["series"]:
        truth = series.get("ground_truth") or []
        for index, position in enumerate(series["positions"]):
            row = {
                "scenario_id": payload["meta"]["scenario_id"],
                "metric": series["name"],
                "unit": series["unit"],
                "position": position,
                "mean": series["mean"][index],
                "lower_95": series["lower_95"][index],
                "upper_95": series["upper_95"][index],
                "aleatoric_var": series["aleatoric_var"][index],
                "epistemic_var": series["epistemic_var"][index],
                "total_var": series["total_var"][index],
            }
            if truth:
                row["ground_truth"] = truth[index]
            rows.append(row)
    path = Path(target_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


# ===== Internationalization (i18n) support =====

class TranslationManager:
    """Manages translations for the application."""

    # Translation dictionary: English -> Chinese/Other languages
    TRANSLATIONS = {
        # Main UI
        "Inference": "推理",
        "Analysis": "分析",
        "Training": "训练",
        "Assets": "资产",
        "Settings && Logs": "设置与日志",
        
        # Buttons
        "Run Inference": "运行推理",
        "Choose Dataset": "选择数据集",
        "Export JSON": "导出 JSON",
        "Cancel": "取消",
        "Upload": "上传",
        "Clear": "清除",
        "Save": "保存",
        "Close": "关闭",
        "Start": "开始",
        "Stop": "停止",
        "Refresh": "刷新",
        
        # Labels & Headers
        "Dataset": "数据集",
        "Select inference dataset": "选择推理数据集",
        "Export prediction JSON": "导出预测 JSON",
        "Select training output directory": "选择训练输出目录",
        "Scenario ID": "工况 ID",
        "Speed Level": "速度等级",
        "Train Features": "训练特征",
        "Model Version": "模型版本",
        "Node Count": "节点数",
        
        # Messages
        "Inference completed successfully": "推理完成",
        "Inference failed": "推理失败",
        "Dataset loaded successfully": "数据集加载成功",
        "Failed to load dataset": "数据集加载失败",
        "Please select a dataset first": "请先选择数据集",
        "No inference results to export": "没有推理结果可导出",
        
        # Status
        "Ready": "就绪",
        "Running": "运行中",
        "Completed": "已完成",
        "Error": "错误",
        "Warning": "警告",
        "Info": "信息",
        
        # Settings
        "Language": "语言",
        "Theme": "主题",
        "Auto-save": "自动保存",
        "Device": "设备",
        "Export Directory": "导出目录",
        "Workspace": "工作区",
        
        # Logs & Results
        "Logs": "日志",
        "Results": "结果",
        "Positions": "位置",
        "Predictions": "预测值",
        "Confidence Interval": "置信区间",
        "Ground Truth": "真实值",
        
        # Error messages
        "Invalid file format": "无效的文件格式",
        "File not found": "文件未找到",
        "Permission denied": "权限被拒绝",
        "Operation failed": "操作失败",
        "Please check the logs for details": "请查看日志了解详情",
        
        # Training
        "Train Model": "训练模型",
        "Epoch": "轮次",
        "Loss": "损失",
        "Validation Loss": "验证损失",
        "Training Progress": "训练进度",
        "Batch Size": "批量大小",
        "Learning Rate": "学习率",
        
        # Asset Management
        "Activate": "激活",
        "Delete": "删除",
        "New Version": "新版本",
        "Active Version": "活跃版本",
        "Upload New Model": "上传新模型",
        
        # Help & About
        "Help": "帮助",
        "About": "关于",
        "Documentation": "文档",
        "Version": "版本",
        "English": "English",
        "中文": "中文",
    }

    def __init__(self, language: str = "English") -> None:
        """Initialize translation manager.

        Args:
            language: Current language ("English" or "中文")
        """
        self.language = language
        self.available_languages = ["English", "中文"]

    def translate(self, text: str) -> str:
        """Translate text based on current language.

        Args:
            text: English text to translate

        Returns:
            Translated text, or original text if translation not found
        """
        if self.language == "English":
            return text

        if self.language == "中文":
            return self.TRANSLATIONS.get(text, text)

        return text

    def set_language(self, language: str) -> bool:
        """Set current language.

        Args:
            language: Language code ("English" or "中文")

        Returns:
            True if language was set successfully
        """
        if language in self.available_languages:
            self.language = language
            return True
        return False

    def get_languages(self) -> list[str]:
        """Get list of available languages.

        Returns:
            List of available language codes
        """
        return self.available_languages.copy()

    def __call__(self, text: str) -> str:
        """Allow using the manager as a callable for convenience.

        Args:
            text: Text to translate

        Returns:
            Translated text
        """
        return self.translate(text)


# Global translator instance
_translator: TranslationManager | None = None


def get_translator() -> TranslationManager:
    """Get or create the global translator instance.

    Returns:
        Global TranslationManager instance
    """
    global _translator
    if _translator is None:
        _translator = TranslationManager()
    return _translator


def set_translator(translator: TranslationManager) -> None:
    """Set the global translator instance.

    Args:
        translator: TranslationManager instance to use globally
    """
    global _translator
    _translator = translator


def tr(text: str) -> str:
    """Convenience function for translating text.

    Args:
        text: Text to translate

    Returns:
        Translated text
    """
    return get_translator().translate(text)
