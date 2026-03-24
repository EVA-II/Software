"""Model asset loading and probabilistic ensemble inference."""

from __future__ import annotations

from datetime import datetime
from typing import Any

import joblib
import numpy as np
import torch

from app.core.config import settings
from app.core.exceptions import AssetValidationError, InferenceExecutionError
from app.core.logging import get_logger
from app.ml_models.boosting_model import ProbabilisticBridgeTrainGNN
from app.schemas.payload import PredictionMeta, PredictionResponse, PredictionSeries, PredictionStatus
from app.services.preprocess import BridgePreprocessor, PreparedScenario
from app.services.rule_engine import RuleEngine

logger = get_logger(__name__)


class ModelInferenceService:
    """Load assets once and serve ensemble probabilistic inference."""

    def __init__(self) -> None:
        self.device = torch.device(settings.device)
        self.models: list[ProbabilisticBridgeTrainGNN] = []
        self.scaler_x: Any = None
        self.scaler_y: Any = None
        self.mapping_model: Any = None
        self.preprocessor: BridgePreprocessor | None = None
        self.rule_engine = RuleEngine()
        self.loaded = False
        self.missing_assets: list[str] = []

    def load_assets(self) -> None:
        self.missing_assets = []
        if not settings.required_feature_columns:
            self.missing_assets.append(str(settings.feature_columns_path))
        for path in settings.expected_model_paths:
            if not path.exists():
                self.missing_assets.append(str(path))
        if not settings.scaler_x_path.exists():
            self.missing_assets.append(str(settings.scaler_x_path))
        if not settings.scaler_y_path.exists():
            self.missing_assets.append(str(settings.scaler_y_path))
        if self.missing_assets:
            raise AssetValidationError("缺少必需模型资产: " + ", ".join(self.missing_assets))

        self.scaler_x = joblib.load(settings.scaler_x_path)
        self.scaler_y = joblib.load(settings.scaler_y_path)
        self.mapping_model = self._load_optional_mapping_model()
        self.preprocessor = BridgePreprocessor(self.scaler_x, self.scaler_y, self.mapping_model)

        node_feature_dim = len(settings.required_feature_columns) - settings.train_feature_dim
        if node_feature_dim <= 0:
            raise AssetValidationError("feature_columns 配置不正确，无法推导桥梁特征维度。")

        self.models = []
        for weight_path in settings.expected_model_paths:
            model = ProbabilisticBridgeTrainGNN(
                node_features=node_feature_dim,
                train_features_dim=settings.train_feature_dim,
                hidden_dim=settings.hidden_dim,
            ).to(self.device)
            state = torch.load(weight_path, map_location=self.device)
            state_dict = state.get("state_dict", state) if isinstance(state, dict) else state
            normalized_state = self._normalize_state_dict_keys(state_dict)
            model.load_state_dict(normalized_state)
            model.eval()
            self.models.append(model)
            logger.info("Loaded model weight: %s", weight_path)

        self.loaded = True
        logger.info(
            "Loaded %s ensemble models, scaler_x=%s, scaler_y=%s, mapping_model=%s",
            len(self.models),
            settings.scaler_x_path,
            settings.scaler_y_path,
            bool(self.mapping_model),
        )

    def ensure_loaded(self) -> None:
        if not self.loaded:
            self.load_assets()

    def predict_from_json(self, payload: Any) -> PredictionResponse:
        self.ensure_loaded()
        assert self.preprocessor is not None
        prepared = self.preprocessor.prepare_json_payload(payload)
        return self.predict_prepared(prepared)

    def predict_from_upload(
        self,
        filename: str,
        content: bytes,
        scenario_id: str | None = None,
        speed_level: str | None = None,
        train_features: list[float] | None = None,
    ) -> PredictionResponse:
        self.ensure_loaded()
        assert self.preprocessor is not None
        prepared = self.preprocessor.prepare_uploaded_file(
            filename=filename,
            content=content,
            scenario_id=scenario_id,
            speed_level=speed_level,
            train_features=train_features,
        )
        return self.predict_prepared(prepared)

    def predict_prepared(self, prepared: PreparedScenario) -> PredictionResponse:
        self.ensure_loaded()
        graph = prepared.graph.to(self.device)
        with torch.no_grad():
            batch_means = []
            batch_vars = []
            for model in self.models:
                pred_mean, pred_logvar = model(graph)
                batch_means.append(pred_mean)
                batch_vars.append(torch.exp(pred_logvar))

        mean_stack = torch.stack(batch_means)
        var_stack = torch.stack(batch_vars)
        ensemble_mean = torch.mean(mean_stack, dim=0).cpu().numpy()
        aleatoric_var = torch.mean(var_stack, dim=0).cpu().numpy()
        if len(self.models) > 1:
            epistemic_var = torch.var(mean_stack, dim=0, unbiased=True).cpu().numpy()
        else:
            epistemic_var = np.zeros_like(ensemble_mean)

        try:
            mean_real = self.scaler_y.inverse_transform(ensemble_mean)
        except Exception as exc:
            raise InferenceExecutionError(f"反标准化失败: {exc}") from exc

        scale_sq = np.asarray(self.scaler_y.scale_, dtype=float) ** 2
        aleatoric_var_real = aleatoric_var * scale_sq
        epistemic_var_real = epistemic_var * scale_sq
        total_var_real = aleatoric_var_real + epistemic_var_real
        std_real = np.sqrt(np.maximum(total_var_real, 0.0))
        lower_95 = mean_real - settings.confidence_z * std_real
        upper_95 = mean_real + settings.confidence_z * std_real

        ground_truth = None
        if hasattr(graph, "y") and graph.y is not None:
            ground_truth = self.scaler_y.inverse_transform(graph.y.detach().cpu().numpy())

        prediction = self._build_prediction_record(
            prepared,
            mean_real,
            aleatoric_var_real,
            epistemic_var_real,
            total_var_real,
            lower_95,
            upper_95,
            ground_truth,
        )
        decision = self.rule_engine.evaluate(prediction)
        return self._build_response(prepared, prediction, decision)

    def _build_prediction_record(
        self,
        prepared: PreparedScenario,
        mean_real: np.ndarray,
        aleatoric_var_real: np.ndarray,
        epistemic_var_real: np.ndarray,
        total_var_real: np.ndarray,
        lower_95: np.ndarray,
        upper_95: np.ndarray,
        ground_truth: np.ndarray | None,
    ) -> dict[str, Any]:
        positions = prepared.graph.positions.detach().cpu().numpy().astype(float).tolist()
        record: dict[str, Any] = {
            "scenario_id": prepared.scenario_id,
            "positions": positions,
            "mean": {},
            "aleatoric_var": {},
            "epistemic_var": {},
            "total_var": {},
            "lower_95": {},
            "upper_95": {},
            "ground_truth": {},
            "metadata": prepared.metadata,
        }
        for index, variable in enumerate(settings.model_output_names):
            record["mean"][variable] = mean_real[:, index].astype(float).tolist()
            record["aleatoric_var"][variable] = aleatoric_var_real[:, index].astype(float).tolist()
            record["epistemic_var"][variable] = epistemic_var_real[:, index].astype(float).tolist()
            record["total_var"][variable] = total_var_real[:, index].astype(float).tolist()
            record["lower_95"][variable] = lower_95[:, index].astype(float).tolist()
            record["upper_95"][variable] = upper_95[:, index].astype(float).tolist()
            if ground_truth is not None:
                record["ground_truth"][variable] = ground_truth[:, index].astype(float).tolist()
        return record

    def _build_response(self, prepared: PreparedScenario, prediction: dict[str, Any], decision: Any) -> PredictionResponse:
        series: list[PredictionSeries] = []
        for variable in settings.model_output_names:
            threshold = settings.variable_thresholds.get(variable, {})
            series.append(
                PredictionSeries(
                    name=variable,
                    unit=str(threshold.get("unit", "-")),
                    positions=prediction["positions"],
                    mean=prediction["mean"][variable],
                    lower_95=prediction["lower_95"][variable],
                    upper_95=prediction["upper_95"][variable],
                    aleatoric_var=prediction["aleatoric_var"][variable],
                    epistemic_var=prediction["epistemic_var"][variable],
                    total_var=prediction["total_var"][variable],
                    ground_truth=prediction["ground_truth"].get(variable),
                )
            )

        return PredictionResponse(
            status=PredictionStatus(**decision.to_dict()),
            meta=PredictionMeta(
                scenario_id=prepared.scenario_id,
                input_mode=prepared.input_mode,
                source=prepared.source,
                model_version=settings.model_version,
                generated_at=datetime.utcnow(),
                node_count=prepared.node_count,
                model_count=len(self.models),
                asset_ready=True,
                extra={
                    "mapping_model_loaded": bool(self.mapping_model),
                    "mapping_mode": prepared.metadata.get("mapping_mode", "unknown"),
                    "ground_truth_available": bool(prepared.metadata.get("ground_truth_available", False)),
                },
            ),
            series=series,
        )

    def health(self) -> dict[str, Any]:
        return {
            "status": "ok" if self.loaded else "initializing",
            "model_version": settings.model_version,
            "model_count": settings.model_count,
            "loaded_model_count": len(self.models),
            "asset_ready": self.loaded and not self.missing_assets,
            "device": str(self.device),
            "missing_assets": self.missing_assets,
        }

    def model_registry(self) -> dict[str, Any]:
        return {
            "model_version": settings.model_version,
            "required_model_count": settings.model_count,
            "loaded_model_count": len(self.models),
            "scaler_x_loaded": self.scaler_x is not None,
            "scaler_y_loaded": self.scaler_y is not None,
            "mapping_model_loaded": self.mapping_model is not None,
            "expected_weight_paths": [str(path) for path in settings.expected_model_paths],
            "missing_assets": self.missing_assets,
        }

    def _load_optional_mapping_model(self) -> Any:
        candidates = [
            settings.mapping_model_path,
            settings.mapping_dir / "bridge_track_mapping.pkl",
            settings.mapping_dir / "bridge_track_mapping.pt",
            settings.mapping_dir / "bridge_track_mapping.pth",
        ]
        for candidate in candidates:
            if not candidate.exists():
                continue
            suffix = candidate.suffix.lower()
            logger.info("Loading optional mapping model: %s", candidate)
            if suffix in {".joblib", ".pkl"}:
                return joblib.load(candidate)
            if suffix in {".pt", ".pth"}:
                return torch.load(candidate, map_location=self.device)
        return None

    @staticmethod
    def _normalize_state_dict_keys(state_dict: dict[str, Any]) -> dict[str, Any]:
        normalized = {}
        for key, value in state_dict.items():
            if key.startswith("module."):
                normalized[key[len("module."):]] = value
            else:
                normalized[key] = value
        return normalized

