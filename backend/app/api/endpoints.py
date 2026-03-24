"""REST endpoints for bridge intelligent assessment."""

from __future__ import annotations

import json
from typing import Annotated, List, Optional

from fastapi import APIRouter, File, Form, HTTPException, Request, UploadFile

from app.core.exceptions import AssetValidationError, BridgeAppError
from app.schemas.payload import (
    HealthResponse,
    ModelRegistryResponse,
    PredictionJobAcceptedResponse,
    PredictionJobStatusResponse,
    PredictJSONRequest,
    PredictionResponse,
)

router = APIRouter()


def get_service(request: Request):
    return request.app.state.inference_service


def get_job_manager(request: Request):
    return request.app.state.job_manager


def _parse_train_features(raw: Optional[str]) -> Optional[List[float]]:
    if not raw:
        return None
    text = raw.strip()
    if not text:
        return None
    try:
        if text.startswith("["):
            values = json.loads(text)
        else:
            values = [value.strip() for value in text.split(",")]
        return [float(value) for value in values]
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"train_features parse failed: {exc}") from exc


@router.get(
    "/health",
    response_model=HealthResponse,
    tags=["系统状态 / System"],
    summary="健康检查 / Health Check",
    description="返回后端服务状态、模型加载状态、设备信息和缺失资产列表。",
)
def health(request: Request) -> HealthResponse:
    service = get_service(request)
    return HealthResponse(**service.health())


@router.get(
    "/models",
    response_model=ModelRegistryResponse,
    tags=["系统状态 / System"],
    summary="模型注册表 / Model Registry",
    description="返回当前已加载的基模型数量、scaler 状态、映射模型状态和缺失资产信息。",
)
def models(request: Request) -> ModelRegistryResponse:
    service = get_service(request)
    return ModelRegistryResponse(**service.model_registry())


@router.post(
    "/predict/json",
    response_model=PredictionResponse,
    tags=["推理预测 / Prediction"],
    summary="JSON 推理 / JSON Inference",
    description=(
        "接收标准 JSON 结构进行推理。支持两种模式："
        "engineered_features（直接传工程化特征）和 raw_measurements（传原始测点数据）。"
    ),
)
def predict_json(payload: PredictJSONRequest, request: Request) -> PredictionResponse:
    service = get_service(request)
    try:
        return service.predict_from_json(payload)
    except AssetValidationError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    except BridgeAppError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post(
    "/predict/file",
    response_model=PredictionResponse,
    tags=["推理预测 / Prediction"],
    summary="文件推理 / File Inference",
    description=(
        "同步上传 Excel / CSV / JSON 文件并执行推理。"
        "大文件推荐优先使用异步任务接口 /predict/file/jobs。"
    ),
)
async def predict_file(
    request: Request,
    file: Annotated[UploadFile, File(...)],
    scenario_id: Annotated[Optional[str], Form()] = None,
    speed_level: Annotated[Optional[str], Form()] = None,
    train_features: Annotated[Optional[str], Form()] = None,
) -> PredictionResponse:
    service = get_service(request)
    try:
        content = await file.read()
        return service.predict_from_upload(
            filename=file.filename or "upload.bin",
            content=content,
            scenario_id=scenario_id,
            speed_level=speed_level,
            train_features=_parse_train_features(train_features),
        )
    except AssetValidationError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    except BridgeAppError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post(
    "/predict/file/jobs",
    response_model=PredictionJobAcceptedResponse,
    status_code=202,
    tags=["推理预测 / Prediction"],
    summary="提交异步文件推理任务 / Submit Async File Prediction Job",
    description="上传文件后立即返回任务编号，前端可轮询任务状态，避免长请求超时。",
)
async def create_prediction_job(
    request: Request,
    file: Annotated[UploadFile, File(...)],
    scenario_id: Annotated[Optional[str], Form()] = None,
    speed_level: Annotated[Optional[str], Form()] = None,
    train_features: Annotated[Optional[str], Form()] = None,
) -> PredictionJobAcceptedResponse:
    job_manager = get_job_manager(request)
    try:
        content = await file.read()
        return await job_manager.create_file_job(
            filename=file.filename or "upload.bin",
            content=content,
            scenario_id=scenario_id,
            speed_level=speed_level,
            train_features=_parse_train_features(train_features),
        )
    except BridgeAppError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.get(
    "/predict/file/jobs/{job_id}",
    response_model=PredictionJobStatusResponse,
    tags=["推理预测 / Prediction"],
    summary="查询异步任务状态 / Get Async Prediction Job Status",
    description="返回文件推理任务的当前状态、进度、错误信息，以及完成后的预测结果。",
)
def get_prediction_job(job_id: str, request: Request) -> PredictionJobStatusResponse:
    job_manager = get_job_manager(request)
    job = job_manager.get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"Prediction job not found: {job_id}")
    return job