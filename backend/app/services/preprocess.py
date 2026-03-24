"""Input parsing and feature preparation for bridge assessment inference."""

from __future__ import annotations

import json
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from pydantic import ValidationError
from scipy.interpolate import UnivariateSpline

from app.core.python_compat import patch_typing_extensions_self

patch_typing_extensions_self()

from torch_geometric.data import Data

from app.core.config import settings
from app.core.exceptions import InputValidationError
from app.ml_models.boosting_model import build_graph_sample
from app.schemas.payload import PredictJSONRequest


@dataclass
class PreparedScenario:
    scenario_id: str
    graph: Data
    source: str
    input_mode: str
    node_count: int
    metadata: dict[str, Any]


class BridgePreprocessor:
    """Convert uploaded files or JSON payloads into graph model inputs."""

    def __init__(self, scaler_x: Any, scaler_y: Any, mapping_model: Any = None) -> None:
        self.scaler_x = scaler_x
        self.scaler_y = scaler_y
        self.mapping_model = mapping_model

    def prepare_json_payload(self, payload: PredictJSONRequest) -> PreparedScenario:
        if payload.input_mode == "engineered_features":
            dataframe = self._feature_records_to_dataframe(payload)
            metadata = {
                "mapping_mode": "not_required",
                "ground_truth_available": all(
                    record.ground_truth is not None for record in payload.feature_records or []
                ),
            }
            return self._engineered_dataframe_to_graph(
                dataframe,
                scenario_id=payload.scenario_id,
                source="json",
                input_mode=payload.input_mode,
                metadata=metadata,
            )

        dataframe, metadata = self._raw_measurements_to_dataframe(payload)
        return self._engineered_dataframe_to_graph(
            dataframe,
            scenario_id=payload.scenario_id,
            source="json",
            input_mode=payload.input_mode,
            metadata=metadata,
        )

    def prepare_uploaded_file(
        self,
        filename: str,
        content: bytes,
        scenario_id: str | None = None,
        speed_level: str | None = None,
        train_features: list[float] | None = None,
    ) -> PreparedScenario:
        suffix = Path(filename).suffix.lower()
        if suffix not in settings.allowed_extensions:
            raise InputValidationError(f"不支持的文件类型: {suffix}")

        if suffix in {".xlsx", ".xls"}:
            return self._prepare_excel(content, filename, scenario_id, speed_level, train_features)
        if suffix == ".csv":
            dataframe = pd.read_csv(BytesIO(content))
            return self._prepare_dataframe_upload(
                dataframe,
                filename,
                scenario_id=scenario_id,
                speed_level=speed_level,
                train_features=train_features,
            )

        try:
            raw = json.loads(content.decode("utf-8"))
        except json.JSONDecodeError as exc:
            raise InputValidationError(f"JSON 文件解析失败: {exc}") from exc
        try:
            payload = PredictJSONRequest.model_validate(raw)
        except ValidationError as exc:
            raise InputValidationError(f"JSON 内容不符合接口约定: {exc}") from exc
        return self.prepare_json_payload(payload)

    def _prepare_excel(
        self,
        content: bytes,
        filename: str,
        scenario_id: str | None,
        speed_level: str | None,
        train_features: list[float] | None,
    ) -> PreparedScenario:
        workbook = pd.ExcelFile(BytesIO(content))
        sheets = set(workbook.sheet_names)
        if {"bridge_data", "train_params"}.issubset(sheets):
            bridge_data = pd.read_excel(workbook, sheet_name="bridge_data")
            train_data = pd.read_excel(workbook, sheet_name="train_params")
            response_data = (
                pd.read_excel(workbook, sheet_name="response_data")
                if "response_data" in sheets
                else None
            )
            bridge_data = self._filter_by_scenario_id(bridge_data, scenario_id, "bridge_data")
            train_data = self._filter_by_scenario_id(train_data, scenario_id, "train_params")
            if response_data is not None:
                response_data = self._filter_by_scenario_id(
                    response_data, scenario_id, "response_data"
                )
            dataframe = self._merge_training_format(bridge_data, train_data, response_data)
            return self._prepare_dataframe_upload(
                dataframe,
                filename,
                scenario_id=scenario_id,
                speed_level=speed_level,
                train_features=train_features,
            )

        dataframe = pd.read_excel(workbook)
        return self._prepare_dataframe_upload(
            dataframe,
            filename,
            scenario_id=scenario_id,
            speed_level=speed_level,
            train_features=train_features,
        )

    def _prepare_dataframe_upload(
        self,
        dataframe: pd.DataFrame,
        source: str,
        scenario_id: str | None,
        speed_level: str | None,
        train_features: list[float] | None,
    ) -> PreparedScenario:
        dataframe = dataframe.copy()
        dataframe = self._select_single_scenario(dataframe, scenario_id, source)
        dataframe = self._inject_train_features(dataframe, speed_level, train_features)
        scenario_value = scenario_id or self._resolve_scenario_id(dataframe, source)
        metadata = {
            "mapping_mode": "from_file_features",
            "ground_truth_available": all(
                column in dataframe.columns for column in settings.target_column_names
            ),
        }
        return self._engineered_dataframe_to_graph(
            dataframe,
            scenario_id=scenario_value,
            source=source,
            input_mode="engineered_features",
            metadata=metadata,
        )

    def _feature_records_to_dataframe(self, payload: PredictJSONRequest) -> pd.DataFrame:
        rows: list[dict[str, Any]] = []
        for record in payload.feature_records or []:
            row: dict[str, Any] = {"position": record.position}
            row.update(record.features)
            if record.ground_truth:
                row.update(record.ground_truth)
            rows.append(row)

        dataframe = pd.DataFrame(rows)
        dataframe = self._inject_train_features(dataframe, payload.speed_level, payload.train_features)
        return dataframe

    def _raw_measurements_to_dataframe(
        self,
        payload: PredictJSONRequest,
    ) -> tuple[pd.DataFrame, dict[str, Any]]:
        measurements = payload.measurements or []
        if len(measurements) < 4:
            raise InputValidationError("raw_measurements 模式至少需要 4 个测点。")

        positions = np.array([item.position for item in measurements], dtype=float)
        deflection = np.array([item.deflection for item in measurements], dtype=float)
        weights = np.array([item.weight for item in measurements], dtype=float)

        order = np.argsort(positions)
        positions = positions[order]
        deflection = deflection[order]
        weights = weights[order]
        smooth_curve = self._run_wcss_reconstruction(positions, deflection, weights)
        track_deformation, mapping_mode = self._map_bridge_to_track(smooth_curve)

        dataframe = pd.DataFrame(
            {
                "position": positions,
                "vertical_irregularity": np.zeros_like(positions),
                "gauge_irregularity": np.zeros_like(positions),
                "horizontal_irregularity": np.zeros_like(positions),
                "track_deformation": track_deformation,
            }
        )
        dataframe = self._inject_train_features(dataframe, payload.speed_level, payload.train_features)
        metadata = {
            "mapping_mode": mapping_mode,
            "ground_truth_available": False,
            "wcss_points": len(positions),
        }
        return dataframe, metadata

    def _merge_training_format(
        self,
        bridge_data: pd.DataFrame,
        train_data: pd.DataFrame,
        response_data: pd.DataFrame | None,
    ) -> pd.DataFrame:
        missing_bridge = {"scenario_id", "position"} - set(bridge_data.columns)
        missing_train = {"scenario_id", "velocity", "weight", "train_type"} - set(train_data.columns)
        if missing_bridge:
            raise InputValidationError(f"bridge_data 缺少列: {sorted(missing_bridge)}")
        if missing_train:
            raise InputValidationError(f"train_params 缺少列: {sorted(missing_train)}")

        dataframe = pd.merge(bridge_data, train_data, on=["scenario_id"], how="left")
        if response_data is not None:
            required_response = {"scenario_id", "position"} | set(settings.target_column_names)
            missing_response = required_response - set(response_data.columns)
            if missing_response:
                raise InputValidationError(f"response_data 缺少列: {sorted(missing_response)}")
            dataframe = pd.merge(
                dataframe,
                response_data,
                on=["scenario_id", "position"],
                how="left",
            )
        return dataframe

    def _engineered_dataframe_to_graph(
        self,
        dataframe: pd.DataFrame,
        scenario_id: str,
        source: str,
        input_mode: str,
        metadata: dict[str, Any],
    ) -> PreparedScenario:
        required_columns = ["position", *settings.required_feature_columns]
        missing_columns = [column for column in required_columns if column not in dataframe.columns]
        if missing_columns:
            raise InputValidationError(f"输入数据缺少必要列: {missing_columns}")

        dataframe = dataframe.copy().sort_values("position").reset_index(drop=True)
        feature_frame = dataframe[settings.required_feature_columns].astype(float)
        positions = dataframe["position"].astype(float).to_numpy()
        targets = None
        if all(column in dataframe.columns for column in settings.target_column_names):
            targets_raw = dataframe[settings.target_column_names].astype(float).to_numpy()
            try:
                targets = self.scaler_y.transform(targets_raw)
            except Exception as exc:
                raise InputValidationError(f"目标值标准化失败: {exc}") from exc

        try:
            scaled_features = self.scaler_x.transform(feature_frame.to_numpy())
        except Exception as exc:
            raise InputValidationError(f"特征标准化失败: {exc}") from exc

        graph = build_graph_sample(
            scaled_features=scaled_features,
            positions=positions,
            scenario_id=str(scenario_id),
            targets=targets,
        )
        metadata.update(
            {
                "feature_columns": settings.required_feature_columns,
                "source_rows": len(dataframe),
            }
        )
        return PreparedScenario(
            scenario_id=str(scenario_id),
            graph=graph,
            source=source,
            input_mode=input_mode,
            node_count=len(dataframe),
            metadata=metadata,
        )

    def _inject_train_features(
        self,
        dataframe: pd.DataFrame,
        speed_level: str | None,
        train_features: list[float] | None,
    ) -> pd.DataFrame:
        dataframe = dataframe.copy()
        train_names = settings.train_feature_names
        if all(name in dataframe.columns for name in train_names):
            return dataframe

        feature_values = None
        if train_features is not None:
            if len(train_features) != len(train_names):
                raise InputValidationError(
                    f"train_features 必须包含 {len(train_names)} 个值: {train_names}"
                )
            feature_values = [float(value) for value in train_features]
        elif speed_level:
            profile = settings.speed_profiles.get(speed_level)
            if profile is None:
                raise InputValidationError(f"未知 speed_level: {speed_level}")
            feature_values = [float(value) for value in profile]

        if feature_values is None:
            return dataframe

        for name, value in zip(train_names, feature_values):
            dataframe[name] = value
        return dataframe

    def _select_single_scenario(
        self,
        dataframe: pd.DataFrame,
        scenario_id: str | None,
        source: str,
    ) -> pd.DataFrame:
        if "scenario_id" not in dataframe.columns:
            return dataframe

        normalized = dataframe["scenario_id"].dropna().astype(str)
        unique_ids = normalized.unique().tolist()
        if not unique_ids:
            raise InputValidationError(f"{source} 中不存在有效的 scenario_id。")

        if scenario_id is None:
            if len(unique_ids) > 1:
                preview = ", ".join(unique_ids[:5])
                raise InputValidationError(
                    f"{source} 包含多个工况，请填写 scenario_id。示例可选值: {preview}"
                )
            return dataframe

        scenario_text = str(scenario_id)
        filtered = dataframe.loc[dataframe["scenario_id"].astype(str) == scenario_text].copy()
        if filtered.empty:
            preview = ", ".join(unique_ids[:10])
            raise InputValidationError(
                f"未在 {source} 中找到 scenario_id={scenario_text}。可选值示例: {preview}"
            )
        return filtered

    def _filter_by_scenario_id(
        self,
        dataframe: pd.DataFrame,
        scenario_id: str | None,
        sheet_name: str,
    ) -> pd.DataFrame:
        if scenario_id is None or "scenario_id" not in dataframe.columns:
            return dataframe

        scenario_text = str(scenario_id)
        filtered = dataframe.loc[dataframe["scenario_id"].astype(str) == scenario_text].copy()
        if filtered.empty:
            raise InputValidationError(
                f"{sheet_name} 中未找到 scenario_id={scenario_text}。"
            )
        return filtered

    def _resolve_scenario_id(self, dataframe: pd.DataFrame, source: str) -> str:
        if "scenario_id" not in dataframe.columns:
            return Path(source).stem
        unique_ids = dataframe["scenario_id"].dropna().astype(str).unique().tolist()
        if not unique_ids:
            return Path(source).stem
        if len(unique_ids) > 1:
            raise InputValidationError("单次请求只能包含一个 scenario_id。")
        return unique_ids[0]

    def _run_wcss_reconstruction(
        self,
        positions: np.ndarray,
        deflection: np.ndarray,
        weights: np.ndarray,
    ) -> np.ndarray:
        smoothing = max(len(positions) * 1e-4, 1e-6)
        spline_order = min(3, len(positions) - 1)
        spline = UnivariateSpline(positions, deflection, w=weights, s=smoothing, k=spline_order)
        return spline(positions)

    def _map_bridge_to_track(self, smooth_curve: np.ndarray) -> tuple[np.ndarray, str]:
        if self.mapping_model is None:
            return smooth_curve.astype(float), "passthrough_placeholder"

        model_input = smooth_curve.reshape(-1, 1)
        if hasattr(self.mapping_model, "predict"):
            mapped = self.mapping_model.predict(model_input)
        elif callable(self.mapping_model):
            mapped = self.mapping_model(model_input)
        else:
            raise InputValidationError("桥-轨映射模型不支持 predict 调用。")

        mapped_array = np.asarray(mapped, dtype=float).reshape(-1)
        if len(mapped_array) != len(smooth_curve):
            raise InputValidationError("桥-轨映射模型输出长度与输入测点数量不一致。")
        return mapped_array, "mapping_model"
