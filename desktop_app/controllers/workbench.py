"""QObject controller exposed to the QML workbench."""

from __future__ import annotations

import os
import uuid
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from PySide6.QtCore import QObject, Property, QThreadPool, QUrl, Signal, Slot
from PySide6.QtWidgets import QFileDialog

from desktop_app.models.contracts import DesktopTaskState, TrainingConfig
from desktop_app.services.services import (
    AppPaths,
    AssetService,
    InferenceFacade,
    StorageService,
    TrainingFacade,
    build_qml_prediction_payload,
    write_prediction_csv,
    write_prediction_json,
)
from desktop_app.workers.tasks import FunctionTask


class WorkbenchController(QObject):
    busyChanged = Signal()
    taskChanged = Signal()
    healthChanged = Signal()
    assetsChanged = Signal()
    logsChanged = Signal()
    errorChanged = Signal()
    resultChanged = Signal()
    metricNamesChanged = Signal()
    chartPointsChanged = Signal()
    settingsChanged = Signal()
    historyChanged = Signal()
    trainingChanged = Signal()

    def __init__(
        self,
        paths: AppPaths,
        storage: StorageService,
        asset_service: AssetService,
        inference: InferenceFacade,
        training: TrainingFacade,
    ) -> None:
        super().__init__()
        self.paths = paths
        self.storage = storage
        self.asset_service = asset_service
        self.inference = inference
        self.training = training
        self.thread_pool = QThreadPool.globalInstance()
        self._settings = self.storage.load_settings()
        settings_changed = {}
        if not self._settings.default_export_dir:
            settings_changed["default_export_dir"] = str(self.paths.exports_dir)
        if not self._settings.last_output_dir:
            settings_changed["last_output_dir"] = str(self.paths.training_dir)
        if settings_changed:
            self._settings = self._settings.merge(settings_changed)
            self.storage.save_settings(self._settings)
        
        self._busy = False
        self._last_error = ""
        self._logs = ""
        self._health: dict[str, Any] = {}
        self._assets: list[dict[str, Any]] = []
        self._history: list[dict[str, Any]] = []
        self._result: dict[str, Any] = {}
        self._metric_names: list[str] = []
        self._selected_metric = ""
        self._chart_points: list[dict[str, Any]] = []
        self._task: Optional[DesktopTaskState] = None
        self._current_worker: Optional[FunctionTask] = None
        self._last_prediction = None
        self._last_training_run: dict[str, Any] = {}
        self.refreshOverview()

    @Property(bool, notify=busyChanged)
    def busy(self) -> bool:
        return self._busy

    @Property("QVariantMap", notify=taskChanged)
    def task(self) -> dict[str, Any]:
        return self._task.as_record() if self._task else {}

    @Property("QVariantMap", notify=healthChanged)
    def health(self) -> dict[str, Any]:
        return self._health

    @Property("QVariantList", notify=assetsChanged)
    def assets(self) -> list[dict[str, Any]]:
        return self._assets

    @Property(str, notify=logsChanged)
    def logs(self) -> str:
        return self._logs

    @Property(str, notify=errorChanged)
    def lastError(self) -> str:
        return self._last_error

    @Property("QVariantMap", notify=resultChanged)
    def result(self) -> dict[str, Any]:
        return self._result

    @Property("QVariantList", notify=metricNamesChanged)
    def metricNames(self) -> list[str]:
        return self._metric_names

    @Property(str, notify=metricNamesChanged)
    def selectedMetric(self) -> str:
        return self._selected_metric

    @Property("QVariantList", notify=chartPointsChanged)
    def chartPoints(self) -> list[dict[str, Any]]:
        return self._chart_points

    @Property("QVariantMap", notify=settingsChanged)
    def appSettings(self) -> dict[str, Any]:
        return asdict(self._settings)

    @Property("QVariantList", notify=historyChanged)
    def recentTasks(self) -> list[dict[str, Any]]:
        return self._history

    @Property("QVariantMap", notify=trainingChanged)
    def lastTrainingRun(self) -> dict[str, Any]:
        return self._last_training_run

    @Slot()
    def refreshOverview(self) -> None:
        self._load_assets()
        self._load_history()
        self.settingsChanged.emit()
        try:
            self._health = self.inference.health()
        except Exception as exc:
            self._health = {
                "status": "error",
                "asset_ready": False,
                "device": self._settings.preferred_device,
                "message": str(exc),
            }
        self.healthChanged.emit()

    @Slot()
    def refreshAssets(self) -> None:
        self._load_assets()

    @Slot(str)
    def activateAsset(self, version: str) -> None:
        try:
            self.asset_service.activate(version)
            self.inference.reload()
            self._settings = self._settings.merge({"active_asset_version": version})
            self.storage.save_settings(self._settings)
            self.settingsChanged.emit()
            self._load_assets()
            self.refreshOverview()
            self._append_log(f"Activated asset version: {version}")
        except Exception as exc:
            self._set_error(str(exc))

    @Slot(str)
    def setSelectedMetric(self, metric: str) -> None:
        self._selected_metric = metric
        self._rebuild_chart_points()
        self.metricNamesChanged.emit()

    @Slot(int)
    def updateSampleLimit(self, limit: int) -> None:
        self._settings = self._settings.merge({"sample_limit": max(50, int(limit))})
        self.storage.save_settings(self._settings)
        self.settingsChanged.emit()
        if self._last_prediction is not None:
            self._result = build_qml_prediction_payload(self._last_prediction, self._settings.sample_limit)
            self._metric_names = [series["name"] for series in self._result.get("series", [])]
            self._rebuild_chart_points()
            self.resultChanged.emit()
            self.metricNamesChanged.emit()

    @Slot()
    def clearLogs(self) -> None:
        self._logs = ""
        self.logsChanged.emit()

    @Slot(result=str)
    def browseInferenceFile(self) -> str:
        start_dir = self._settings.last_dataset_dir or str(self.paths.install_dir)
        file_path, _ = QFileDialog.getOpenFileName(
            None,
            "Select inference dataset",
            start_dir,
            "Data Files (*.xlsx *.xls *.csv *.json)",
        )
        return file_path or ""

    @Slot(result=str)
    def browseTrainingDataset(self) -> str:
        start_dir = self._settings.last_dataset_dir or str(self.paths.install_dir)
        file_path, _ = QFileDialog.getOpenFileName(
            None,
            "Select training dataset",
            start_dir,
            "Excel Files (*.xlsx *.xls)",
        )
        return file_path or ""

    @Slot(result=str)
    def browseTrainingOutputDir(self) -> str:
        start_dir = self._settings.last_output_dir or str(self.paths.training_dir)
        directory = QFileDialog.getExistingDirectory(None, "Select training output directory", start_dir)
        return directory or ""

    @Slot(result=str)
    def browseExportJsonPath(self) -> str:
        start_dir = self._settings.default_export_dir or str(self.paths.exports_dir)
        file_path, _ = QFileDialog.getSaveFileName(
            None,
            "Export prediction JSON",
            str(Path(start_dir) / f"prediction-{datetime.utcnow().strftime('%Y%m%d-%H%M%S')}.json"),
            "JSON files (*.json)",
        )
        return file_path or ""

    @Slot(result=str)
    def browseExportCsvPath(self) -> str:
        start_dir = self._settings.default_export_dir or str(self.paths.exports_dir)
        file_path, _ = QFileDialog.getSaveFileName(
            None,
            "Export prediction CSV",
            str(Path(start_dir) / f"prediction-{datetime.utcnow().strftime('%Y%m%d-%H%M%S')}.csv"),
            "CSV files (*.csv)",
        )
        return file_path or ""

    @Slot(result=str)
    def browseExportImagePath(self) -> str:
        start_dir = self._settings.default_export_dir or str(self.paths.exports_dir)
        file_path, _ = QFileDialog.getSaveFileName(
            None,
            "Export chart snapshot",
            str(Path(start_dir) / f"prediction-{datetime.utcnow().strftime('%Y%m%d-%H%M%S')}.png"),
            "PNG files (*.png)",
        )
        return file_path or ""

    @Slot()
    def cancelActiveTask(self) -> None:
        if self._current_worker is not None:
            self._current_worker.cancel()
            if self._task is not None:
                self._task.update(status="cancelled", progress=self._task.progress, message="Cancellation requested")
                self.storage.save_task(self._task)
                self.taskChanged.emit()
            self._append_log("Cancellation requested for active task.")

    @Slot(str, str, str, str)
    def runInference(self, filePath: str, scenarioId: str, speedLevel: str, trainFeatures: str = "") -> None:
        resolved_path = self._normalize_path(filePath)
        if not resolved_path or not Path(resolved_path).exists():
            self._set_error("Please select a valid dataset file.")
            return
        try:
            parsed_features = self._parse_train_features(trainFeatures)
        except Exception as exc:
            self._set_error(f"Invalid train_features: {exc}")
            return
        self._settings = self._settings.merge({"last_dataset_dir": str(Path(resolved_path).parent)})
        self.storage.save_settings(self._settings)
        self.settingsChanged.emit()
        task = self._begin_task("inference", f"Queued inference for {Path(resolved_path).name}")
        worker = FunctionTask(
            self._execute_inference,
            resolved_path,
            scenarioId.strip() or None,
            speedLevel.strip() or None,
            parsed_features,
        )
        self._wire_worker(task, worker, self._handle_inference_result, f"Inference failed for {Path(resolved_path).name}")

    @Slot(str, str, str, float, int, int, int, int, int)
    def startTraining(
        self,
        datasetPath: str,
        outputDir: str,
        device: str,
        trainSize: float,
        batchSize: int,
        numModels: int,
        numEpochs: int,
        patience: int,
        warmupEpochs: int,
    ) -> None:
        resolved_dataset = self._normalize_path(datasetPath)
        resolved_output = self._normalize_path(outputDir) or str(self.paths.training_dir)
        config = TrainingConfig(
            dataset_path=resolved_dataset,
            output_dir=resolved_output,
            device=device or self._settings.preferred_device,
            train_size=float(trainSize),
            val_size=0.1,
            test_size=round(1.0 - float(trainSize) - 0.1, 6),
            batch_size=int(batchSize),
            num_models=int(numModels),
            num_epochs=int(numEpochs),
            patience=int(patience),
            warmup_epochs=int(warmupEpochs),
        )
        try:
            config.validate()
        except Exception as exc:
            self._set_error(str(exc))
            return
        self._settings = self._settings.merge(
            {
                "last_dataset_dir": str(Path(resolved_dataset).parent),
                "last_output_dir": resolved_output,
                "preferred_device": device or self._settings.preferred_device,
            }
        )
        self.storage.save_settings(self._settings)
        self.settingsChanged.emit()
        task = self._begin_task("training", f"Queued training for {Path(resolved_dataset).name}")
        worker = FunctionTask(self._execute_training, config)
        self._wire_worker(task, worker, self._handle_training_result, "Training failed")

    @Slot(str)
    def exportLastResultJson(self, targetPath: str) -> None:
        if self._last_prediction is None:
            self._set_error("No prediction result is available to export.")
            return
        path = self._normalize_path(targetPath) or str(self.paths.exports_dir / f"prediction-{datetime.utcnow().strftime('%Y%m%d-%H%M%S')}.json")
        written = write_prediction_json(self._last_prediction, path)
        self._append_log(f"Exported prediction JSON: {written}")

    @Slot(str)
    def exportLastResultCsv(self, targetPath: str) -> None:
        if self._last_prediction is None:
            self._set_error("No prediction result is available to export.")
            return
        path = self._normalize_path(targetPath) or str(self.paths.exports_dir / f"prediction-{datetime.utcnow().strftime('%Y%m%d-%H%M%S')}.csv")
        written = write_prediction_csv(self._last_prediction, path)
        self._append_log(f"Exported prediction CSV: {written}")

    def _wire_worker(
        self,
        task: DesktopTaskState,
        worker: FunctionTask,
        result_handler: Any,
        failure_prefix: str,
    ) -> None:
        self._set_busy(True)
        self._current_worker = worker
        worker.signals.status_changed.connect(self._handle_worker_status)
        worker.signals.log_emitted.connect(self._append_log)
        worker.signals.metrics_emitted.connect(self._handle_worker_metrics)
        worker.signals.result_ready.connect(result_handler)
        worker.signals.failed.connect(lambda text: self._handle_worker_failure(failure_prefix, text))
        worker.signals.finished.connect(self._handle_worker_finished)
        self.thread_pool.start(worker)
        self.storage.save_task(task)

    def _begin_task(self, kind: str, message: str) -> DesktopTaskState:
        self._set_error("")
        task = DesktopTaskState(task_id=uuid.uuid4().hex[:12], kind=kind, status="queued", progress=0, message=message)
        self._task = task
        self.taskChanged.emit()
        self._append_log(message)
        return task

    def _execute_inference(
        self,
        file_path: str,
        scenario_id: str | None,
        speed_level: str | None,
        train_features: list[float] | None,
        *,
        proxy: Any,
    ) -> Any:
        proxy.status("running", 10, "Preparing inference request")
        proxy.log(f"Running inference on {file_path}")
        result = self.inference.predict_file(file_path, scenario_id, speed_level, train_features)
        proxy.status("running", 95, "Rendering prediction result")
        return result

    def _execute_training(self, config: TrainingConfig, *, proxy: Any) -> dict[str, Any]:
        callbacks = {
            "on_status": proxy.status,
            "on_log": proxy.log,
            "on_metrics": proxy.metrics,
            "check_cancelled": proxy.is_cancelled,
        }
        record = self.training.start(config, callbacks)
        return record.as_dict()

    def _handle_worker_status(self, status: str, progress: int, message: str) -> None:
        if self._task is None:
            return
        self._task.update(status=status, progress=progress, message=message)
        self.storage.save_task(self._task)
        self.taskChanged.emit()

    def _handle_worker_metrics(self, payload: dict[str, Any]) -> None:
        self._append_log(f"Metrics update: {payload}")

    def _handle_worker_failure(self, prefix: str, traceback_text: str) -> None:
        message = traceback_text.strip().splitlines()[-1] if traceback_text.strip() else prefix
        self._set_error(f"{prefix}: {message}")
        self._append_log(traceback_text)
        if self._task is not None:
            self._task.update(status="failed", progress=self._task.progress, message=prefix, error=message)
            self.storage.save_task(self._task)
            self.taskChanged.emit()

    def _handle_worker_finished(self) -> None:
        self._current_worker = None
        self._set_busy(False)
        self._load_history()

    def _handle_inference_result(self, result: Any) -> None:
        self._last_prediction = result
        self._result = build_qml_prediction_payload(result, self._settings.sample_limit)
        self._metric_names = [series["name"] for series in self._result.get("series", [])]
        self._selected_metric = self._metric_names[0] if self._metric_names else ""
        self._rebuild_chart_points()
        if self._task is not None:
            self._task.update(status="completed", progress=100, message="Inference completed")
            payload = {
                "scenario_id": self._result.get("meta", {}).get("scenario_id"),
                "status": self._result.get("status", {}).get("label"),
            }
            self.storage.save_task(self._task, payload)
            self.taskChanged.emit()
        self.resultChanged.emit()
        self.metricNamesChanged.emit()
        self._append_log("Inference completed.")
        self.refreshOverview()

    def _handle_training_result(self, result: dict[str, Any]) -> None:
        self._last_training_run = result
        version = result.get("version")
        if version:
            self.asset_service.activate(version)
            self.inference.reload()
        if self._task is not None:
            self._task.update(status="completed", progress=100, message="Training completed")
            self.storage.save_task(self._task, result)
            self.taskChanged.emit()
        self.trainingChanged.emit()
        self._load_assets()
        self.refreshOverview()
        self._append_log(f"Training completed. Registered version: {version}")

    def _load_assets(self) -> None:
        self._assets = [asdict(item) for item in self.asset_service.scan()]
        self.assetsChanged.emit()

    def _load_history(self) -> None:
        self._history = self.storage.recent_tasks()
        self.historyChanged.emit()

    def _rebuild_chart_points(self) -> None:
        self._chart_points = []
        if not self._selected_metric:
            self.chartPointsChanged.emit()
            return
        for series in self._result.get("series", []):
            if series.get("name") == self._selected_metric:
                self._chart_points = list(series.get("points", []))
                break
        self.chartPointsChanged.emit()

    def _append_log(self, message: str) -> None:
        if not message:
            return
        line = str(message).rstrip()
        self._logs = f"{self._logs}\n{line}".strip()
        self.logsChanged.emit()

    def _set_error(self, message: str) -> None:
        self._last_error = message
        self.errorChanged.emit()

    def _set_busy(self, busy: bool) -> None:
        self._busy = busy
        self.busyChanged.emit()

    @staticmethod
    def _normalize_path(value: str) -> str:
        if not value:
            return ""
        if value.startswith("file:"):
            return QUrl(value).toLocalFile()
        return os.path.expanduser(value)

    @staticmethod
    def _parse_train_features(text: str) -> list[float] | None:
        if not text.strip():
            return None
        values = [item.strip() for item in text.split(",") if item.strip()]
        return [float(item) for item in values]

    # ========== Internationalization (i18n) Methods ==========


