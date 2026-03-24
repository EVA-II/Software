"""In-memory async job orchestration for file inference."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from datetime import datetime
from threading import Lock
from typing import Optional
from uuid import uuid4

from app.core.exceptions import AssetValidationError, BridgeAppError
from app.core.logging import get_logger
from app.schemas.payload import (
    PredictionJobAcceptedResponse,
    PredictionJobStatusResponse,
    PredictionResponse,
)
from app.services.inference import ModelInferenceService
from app.services.preprocess import PreparedScenario

logger = get_logger(__name__)


@dataclass
class PredictionJob:
    job_id: str
    file_name: str
    scenario_id: Optional[str]
    speed_level: Optional[str]
    status: str = "queued"
    progress: int = 5
    message: str = "Queued for inference."
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    error: Optional[str] = None
    result: Optional[PredictionResponse] = None


class PredictionJobManager:
    def __init__(self, inference_service: ModelInferenceService, max_jobs: int = 24) -> None:
        self.inference_service = inference_service
        self.max_jobs = max_jobs
        self.jobs: dict[str, PredictionJob] = {}
        self._jobs_lock = Lock()
        self._runner_semaphore: Optional[asyncio.Semaphore] = None

    async def create_file_job(
        self,
        filename: str,
        content: bytes,
        scenario_id: Optional[str] = None,
        speed_level: Optional[str] = None,
        train_features: Optional[list[float]] = None,
    ) -> PredictionJobAcceptedResponse:
        job = PredictionJob(
            job_id=uuid4().hex,
            file_name=filename,
            scenario_id=scenario_id,
            speed_level=speed_level,
        )
        with self._jobs_lock:
            self.jobs[job.job_id] = job
            self._prune_jobs()

        asyncio.create_task(
            self._run_file_job(
                job_id=job.job_id,
                filename=filename,
                content=content,
                scenario_id=scenario_id,
                speed_level=speed_level,
                train_features=train_features,
            )
        )
        return PredictionJobAcceptedResponse(
            job_id=job.job_id,
            status=job.status,
            progress=job.progress,
            message=job.message,
            poll_url=f"/api/v1/predict/file/jobs/{job.job_id}",
            created_at=job.created_at,
        )

    def get_job(self, job_id: str) -> Optional[PredictionJobStatusResponse]:
        with self._jobs_lock:
            job = self.jobs.get(job_id)
            if job is None:
                return None
            return self._to_status_response(job)

    async def _run_file_job(
        self,
        job_id: str,
        filename: str,
        content: bytes,
        scenario_id: Optional[str],
        speed_level: Optional[str],
        train_features: Optional[list[float]],
    ) -> None:
        semaphore = self._get_runner_semaphore()
        await self._update_job(job_id, status="queued", progress=10, message="Waiting for worker slot.")
        try:
            async with semaphore:
                await self._update_job(job_id, status="running", progress=20, message="Parsing uploaded file.")
                prepared = await asyncio.to_thread(
                    self._prepare_upload,
                    filename,
                    content,
                    scenario_id,
                    speed_level,
                    train_features,
                )
                await self._update_job(
                    job_id,
                    status="running",
                    progress=55,
                    message=f"Running ensemble inference on {prepared.node_count} nodes.",
                )
                prediction = await asyncio.to_thread(self.inference_service.predict_prepared, prepared)
                await self._update_job(
                    job_id,
                    status="completed",
                    progress=100,
                    message="Inference completed.",
                    result=prediction,
                    error=None,
                )
        except (BridgeAppError, AssetValidationError) as exc:
            await self._update_job(
                job_id,
                status="failed",
                progress=100,
                message="Inference failed.",
                error=str(exc),
            )
        except Exception as exc:
            logger.exception("Unexpected failure while processing prediction job %s", job_id)
            await self._update_job(
                job_id,
                status="failed",
                progress=100,
                message="Inference failed unexpectedly.",
                error=str(exc),
            )

    def _prepare_upload(
        self,
        filename: str,
        content: bytes,
        scenario_id: Optional[str],
        speed_level: Optional[str],
        train_features: Optional[list[float]],
    ) -> PreparedScenario:
        self.inference_service.ensure_loaded()
        assert self.inference_service.preprocessor is not None
        return self.inference_service.preprocessor.prepare_uploaded_file(
            filename=filename,
            content=content,
            scenario_id=scenario_id,
            speed_level=speed_level,
            train_features=train_features,
        )

    async def _update_job(
        self,
        job_id: str,
        *,
        status: Optional[str] = None,
        progress: Optional[int] = None,
        message: Optional[str] = None,
        error: Optional[str] = None,
        result: Optional[PredictionResponse] = None,
    ) -> None:
        with self._jobs_lock:
            job = self.jobs.get(job_id)
            if job is None:
                return
            if status is not None:
                job.status = status
            if progress is not None:
                job.progress = progress
            if message is not None:
                job.message = message
            job.error = error
            if result is not None:
                job.result = result
            job.updated_at = datetime.utcnow()

    def _to_status_response(self, job: PredictionJob) -> PredictionJobStatusResponse:
        return PredictionJobStatusResponse(
            job_id=job.job_id,
            status=job.status,
            progress=job.progress,
            message=job.message,
            created_at=job.created_at,
            updated_at=job.updated_at,
            file_name=job.file_name,
            scenario_id=job.scenario_id,
            error=job.error,
            result=job.result,
        )

    def _prune_jobs(self) -> None:
        if len(self.jobs) <= self.max_jobs:
            return

        completed_ids = [
            job.job_id
            for job in sorted(self.jobs.values(), key=lambda item: item.updated_at)
            if job.status in {"completed", "failed"}
        ]
        while len(self.jobs) > self.max_jobs and completed_ids:
            self.jobs.pop(completed_ids.pop(0), None)

    def _get_runner_semaphore(self) -> asyncio.Semaphore:
        if self._runner_semaphore is None:
            self._runner_semaphore = asyncio.Semaphore(1)
        return self._runner_semaphore