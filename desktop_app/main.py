"""Desktop application bootstrap for the Bridge Assessment Workbench."""

from __future__ import annotations

import os
import sys
import traceback
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
BACKEND_DIR = ROOT_DIR / "backend"
for candidate in (ROOT_DIR, BACKEND_DIR):
    candidate_str = str(candidate)
    if candidate_str not in sys.path:
        sys.path.insert(0, candidate_str)

os.environ.setdefault("QT_QUICK_CONTROLS_STYLE", "Basic")

from PySide6.QtCore import QUrl
from PySide6.QtQml import QQmlApplicationEngine
from PySide6.QtQuickControls2 import QQuickStyle
from PySide6.QtWidgets import QApplication, QMessageBox

from desktop_app.controllers.workbench import WorkbenchController
from desktop_app.services.services import AppPaths, AssetService, InferenceFacade, StorageService, TrainingFacade


def create_controller() -> WorkbenchController:
    paths = AppPaths()
    paths.ensure()
    storage = StorageService(paths.db_path)
    asset_service = AssetService(paths, storage)
    inference = InferenceFacade(asset_service)
    training = TrainingFacade(asset_service, paths)
    return WorkbenchController(paths, storage, asset_service, inference, training)


def _report_startup_failure(message: str, details: str = "") -> None:
    print(message, file=sys.stderr, flush=True)
    if details:
        print(details, file=sys.stderr, flush=True)
    try:
        QMessageBox.critical(None, "Bridge Assessment Workbench", message + ("\n\n" + details if details else ""))
    except Exception:
        pass


def main() -> int:
    try:
        app = QApplication(sys.argv)
        QQuickStyle.setStyle("Basic")
        app.setApplicationName("Bridge Assessment Workbench")
        app.setOrganizationName("BridgeAssessment")

        controller = create_controller()
        engine = QQmlApplicationEngine()
        app._controller = controller
        engine._controller = controller
        qml_warnings: list[str] = []

        def on_warnings(warnings: list[object]) -> None:
            for warning in warnings:
                text = warning.toString()
                qml_warnings.append(text)
                print(text, file=sys.stderr, flush=True)

        engine.warnings.connect(on_warnings)
        engine.rootContext().setContextProperty("workbench", controller)
        main_qml = ROOT_DIR / "desktop_app" / "qml" / "Main.qml"
        print(f"Loading QML: {main_qml}", flush=True)
        engine.load(QUrl.fromLocalFile(str(main_qml.resolve())))
        if not engine.rootObjects():
            details = "\n".join(qml_warnings) if qml_warnings else f"No root object was created from {main_qml}."
            _report_startup_failure("Desktop UI failed to load.", details)
            return 1
        return app.exec()
    except Exception as exc:
        _report_startup_failure(f"Desktop startup failed: {exc}", traceback.format_exc())
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
