"""Background worker primitives for non-blocking desktop tasks."""

from __future__ import annotations

import traceback
from typing import Any, Callable

from PySide6.QtCore import QObject, QRunnable, Signal, Slot


class WorkerSignals(QObject):
    status_changed = Signal(str, int, str)
    log_emitted = Signal(str)
    metrics_emitted = Signal("QVariantMap")
    result_ready = Signal(object)
    failed = Signal(str)
    finished = Signal()


class TaskProxy:
    def __init__(self, signals: WorkerSignals) -> None:
        self._signals = signals
        self._cancelled = False

    def status(self, status: str, progress: int, message: str) -> None:
        self._signals.status_changed.emit(status, int(progress), message)

    def log(self, message: str) -> None:
        self._signals.log_emitted.emit(message)

    def metrics(self, payload: dict[str, Any]) -> None:
        self._signals.metrics_emitted.emit(payload)

    def cancel(self) -> None:
        self._cancelled = True

    def is_cancelled(self) -> bool:
        return self._cancelled


class FunctionTask(QRunnable):
    def __init__(self, fn: Callable[..., Any], *args: Any, **kwargs: Any) -> None:
        super().__init__()
        self.fn = fn
        self.args = args
        self.kwargs = kwargs
        self.signals = WorkerSignals()
        self.proxy = TaskProxy(self.signals)
        self.setAutoDelete(True)

    def cancel(self) -> None:
        self.proxy.cancel()

    @Slot()
    def run(self) -> None:
        try:
            result = self.fn(*self.args, proxy=self.proxy, **self.kwargs)
        except Exception:
            self.signals.failed.emit(traceback.format_exc())
        else:
            self.signals.result_ready.emit(result)
        finally:
            self.signals.finished.emit()
