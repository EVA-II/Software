"""Serviceized probabilistic ensemble training entrypoint."""

from __future__ import annotations

import json
import random
import shutil
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Optional

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from torch.cuda.amp import GradScaler, autocast

from app.core.python_compat import patch_typing_extensions_self

patch_typing_extensions_self()

from torch_geometric.loader import DataLoader

from app.core.config import settings
from app.ml_models.boosting_model import ProbabilisticBridgeTrainGNN, compute_evaluation_metrics
from train import create_graph_data, load_data_from_excel, preprocess_data


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
        dataset = Path(self.dataset_path)
        if not dataset.exists():
            raise ValueError(f"Dataset path does not exist: {dataset}")
        split_total = round(self.train_size + self.val_size + self.test_size, 6)
        if abs(split_total - 1.0) > 1e-6:
            raise ValueError("train_size + val_size + test_size must equal 1.0")
        if min(self.train_size, self.val_size, self.test_size) <= 0:
            raise ValueError("Split sizes must all be > 0")
        if self.batch_size <= 0 or self.num_models <= 0 or self.hidden_dim <= 0:
            raise ValueError("batch_size, num_models and hidden_dim must be > 0")
        if self.num_epochs <= 0 or self.patience <= 0:
            raise ValueError("num_epochs and patience must be > 0")
        if self.warmup_epochs < 0 or self.warmup_epochs >= self.num_epochs:
            raise ValueError("warmup_epochs must be >= 0 and smaller than num_epochs")
        return self


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


@dataclass
class TrainingCallbacks:
    on_status: Optional[Callable[[str, int, str], None]] = None
    on_log: Optional[Callable[[str], None]] = None
    on_metrics: Optional[Callable[[dict[str, Any]], None]] = None
    check_cancelled: Optional[Callable[[], bool]] = None


class TrainingCancelledError(RuntimeError):
    """Raised when a desktop task requests cancellation."""


class GaussianNLLPhysicsLoss(nn.Module):
    def __init__(self, alpha: float = 10.0, mass: float = 31600, scaler_y: Any = None, device: str = "cpu") -> None:
        super().__init__()
        self.alpha = alpha
        self.mass = mass
        self.smooth = nn.SmoothL1Loss()
        if scaler_y is not None:
            self.y_mean = torch.tensor(scaler_y.mean_, dtype=torch.float32, device=device)
            self.y_scale = torch.tensor(scaler_y.scale_, dtype=torch.float32, device=device)
            self.has_scaler = True
        else:
            self.has_scaler = False

    def forward(self, pred_mean: torch.Tensor, pred_logvar: torch.Tensor, target: torch.Tensor, *, is_warmup: bool = False) -> torch.Tensor:
        pred_mean = pred_mean.float()
        pred_logvar = pred_logvar.float()
        target = target.float()
        if is_warmup:
            data_loss = self.smooth(pred_mean, target)
        else:
            precision = torch.exp(-pred_logvar)
            nll_loss = 0.5 * precision * (target - pred_mean) ** 2 + 0.5 * pred_logvar
            var_reg = 0.05 * (pred_logvar ** 2)
            data_loss = torch.mean(nll_loss + var_reg)

        if self.alpha == 0 or not self.has_scaler:
            return data_loss

        pred_original = pred_mean * self.y_scale + self.y_mean
        pred_acc_real = pred_original[:, 0]
        pred_force_real = pred_original[:, 3]
        theoretical_acc_real = 9.81 + (pred_force_real / self.mass)
        theoretical_acc_norm = (theoretical_acc_real - self.y_mean[0]) / self.y_scale[0]
        physics_loss = self.smooth(pred_mean[:, 0], theoretical_acc_norm)
        return data_loss + self.alpha * physics_loss


class TrainingService:
    """Run probabilistic ensemble training without notebook dependencies."""

    def start(self, config: TrainingConfig, callbacks: TrainingCallbacks | None = None) -> TrainingRunRecord:
        config = config.validate()
        callbacks = callbacks or TrainingCallbacks()
        started_at = datetime.utcnow()
        run_id = uuid.uuid4().hex[:12]
        version = f"desktop-{started_at.strftime('%Y%m%d-%H%M%S')}-{run_id}"
        run_root = Path(config.output_dir).resolve() / version
        models_dir = run_root / "models"
        reports_dir = run_root / "reports"
        history_dir = run_root / "history"
        config_dir = run_root / "config"
        for directory in (models_dir, reports_dir, history_dir, config_dir):
            directory.mkdir(parents=True, exist_ok=True)

        record = TrainingRunRecord(
            run_id=run_id,
            version=version,
            status="running",
            output_dir=str(run_root),
            models_dir=str(models_dir),
            reports_dir=str(reports_dir),
            manifest_path=str(config_dir / "training_run.json"),
            started_at=started_at.isoformat(),
            message="Training initialized.",
        )
        self._emit_status(callbacks, "running", 2, "Loading training dataset")
        self._emit_log(callbacks, f"Loading dataset from {config.dataset_path}")
        self._set_seed(config.random_seed)
        self._raise_if_cancelled(callbacks)

        bridge_data, train_data, response_data = load_data_from_excel(config.dataset_path)
        features_df, labels_df, scaler_x, scaler_y = preprocess_data(bridge_data, train_data, response_data)
        max_id = int(features_df["scenario_id"].max())
        graph_data_list = create_graph_data(features_df, labels_df, max_id + 1)
        if config.is_trial and len(graph_data_list) > config.trial_size:
            graph_data_list = graph_data_list[: config.trial_size]
            self._emit_log(callbacks, f"Trial mode enabled, truncated dataset to {len(graph_data_list)} scenarios")
        if len(graph_data_list) < 3:
            raise ValueError("At least 3 scenarios are required to split train/val/test datasets.")

        train_idx, val_idx, test_idx = self._split_indices(len(graph_data_list), config)
        record.data_split = {"train": len(train_idx), "val": len(val_idx), "test": len(test_idx)}
        self._emit_metrics(callbacks, {"phase": "split", **record.data_split})

        device = self._resolve_device(config.device)
        node_features = graph_data_list[0].x.shape[1] - settings.train_feature_dim
        criterion = GaussianNLLPhysicsLoss(
            alpha=config.alpha_physics,
            mass=config.mass,
            scaler_y=scaler_y,
            device=str(device),
        )
        train_loader = self._build_loader(graph_data_list, train_idx, config, shuffle=True)
        val_loader = self._build_loader(graph_data_list, val_idx, config, shuffle=False)
        test_loader = self._build_loader(graph_data_list, test_idx, config, shuffle=False)

        scaler_x_path = models_dir / "scaler_X.pkl"
        scaler_y_path = models_dir / "scaler_y.pkl"
        joblib.dump(scaler_x, scaler_x_path)
        joblib.dump(scaler_y, scaler_y_path)
        record.scaler_x_path = str(scaler_x_path)
        record.scaler_y_path = str(scaler_y_path)
        self._emit_status(callbacks, "running", 8, "Training ensemble models")

        for model_id in range(1, config.num_models + 1):
            self._raise_if_cancelled(callbacks)
            history_path, model_path = self._train_single_model(
                model_id=model_id,
                model_count=config.num_models,
                node_features=node_features,
                criterion=criterion,
                train_loader=train_loader,
                val_loader=val_loader,
                config=config,
                device=device,
                models_dir=models_dir,
                history_dir=history_dir,
                callbacks=callbacks,
            )
            record.history_files.append(str(history_path))
            record.model_paths.append(str(model_path))

        self._emit_status(callbacks, "running", 85, "Evaluating ensemble uncertainty")
        eval_results = self._evaluate_all_splits(
            node_features=node_features,
            scaler_y=scaler_y,
            config=config,
            device=device,
            loaders={"Train": train_loader, "Val": val_loader, "Test": test_loader},
            models_dir=models_dir,
            reports_dir=reports_dir,
        )
        record.metrics_files.extend(eval_results["metrics_files"])
        record.detail_files.extend(eval_results["detail_files"])
        self._copy_runtime_assets(run_root, version)
        record.finished_at = datetime.utcnow().isoformat()
        record.status = "completed"
        record.message = "Training completed successfully."
        self._write_manifest(record, config)
        self._emit_status(callbacks, "completed", 100, "Training completed")
        return record
    def _train_single_model(
        self,
        *,
        model_id: int,
        model_count: int,
        node_features: int,
        criterion: GaussianNLLPhysicsLoss,
        train_loader: DataLoader,
        val_loader: DataLoader,
        config: TrainingConfig,
        device: torch.device,
        models_dir: Path,
        history_dir: Path,
        callbacks: TrainingCallbacks,
    ) -> tuple[Path, Path]:
        torch.manual_seed(config.random_seed + model_id * 100)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(config.random_seed + model_id * 100)

        model = ProbabilisticBridgeTrainGNN(
            node_features=node_features,
            train_features_dim=settings.train_feature_dim,
            hidden_dim=config.hidden_dim,
        ).to(device)

        unloading_params = []
        other_params = []
        for name, param in model.named_parameters():
            if "unloading" in name:
                unloading_params.append(param)
            else:
                other_params.append(param)
        optimizer = optim.AdamW(
            [
                {"params": unloading_params, "lr": config.lr_unloading, "weight_decay": config.weight_decay},
                {"params": other_params, "lr": config.lr_other, "weight_decay": config.weight_decay},
            ]
        )
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=20)
        amp_enabled = device.type == "cuda"
        scaler = GradScaler(enabled=amp_enabled)
        best_val_loss = float("inf")
        patience_counter = 0
        model_path = models_dir / f"ensemble_model_{model_id}.pth"
        history_path = history_dir / f"ensemble_model_{model_id}_loss.csv"
        history_rows: list[dict[str, Any]] = []

        for epoch in range(config.num_epochs):
            self._raise_if_cancelled(callbacks)
            is_warmup = epoch < config.warmup_epochs
            train_loss = self._run_epoch(
                model=model,
                loader=train_loader,
                optimizer=optimizer,
                scaler=scaler,
                criterion=criterion,
                device=device,
                is_warmup=is_warmup,
                amp_enabled=amp_enabled,
            )
            val_loss = self._evaluate_epoch(
                model=model,
                loader=val_loader,
                criterion=criterion,
                device=device,
                is_warmup=is_warmup,
            )
            scheduler.step(val_loss)

            phase = "Warm-up(MSE)" if is_warmup else "Joint(NLL)"
            history_rows.append(
                {
                    "epoch": epoch + 1,
                    "train_loss": float(train_loss),
                    "val_loss": float(val_loss),
                    "phase": phase,
                }
            )
            progress_ratio = ((model_id - 1) + ((epoch + 1) / config.num_epochs)) / model_count
            progress = 8 + int(progress_ratio * 70)
            self._emit_status(
                callbacks,
                "running",
                progress,
                f"Training ensemble model {model_id}/{model_count}: epoch {epoch + 1}/{config.num_epochs}",
            )

            if not is_warmup:
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                    torch.save(model.state_dict(), model_path)
                else:
                    patience_counter += 1
                    if patience_counter >= config.patience:
                        break

        if not model_path.exists():
            torch.save(model.state_dict(), model_path)
        pd.DataFrame(history_rows).to_csv(history_path, index=False)
        self._emit_metrics(
            callbacks,
            {
                "phase": "model_complete",
                "model_id": model_id,
                "best_val_loss": None if best_val_loss == float("inf") else float(best_val_loss),
                "history_file": str(history_path),
            },
        )
        return history_path, model_path

    def _run_epoch(
        self,
        *,
        model: nn.Module,
        loader: DataLoader,
        optimizer: optim.Optimizer,
        scaler: GradScaler,
        criterion: GaussianNLLPhysicsLoss,
        device: torch.device,
        is_warmup: bool,
        amp_enabled: bool,
    ) -> float:
        model.train()
        total_loss = 0.0
        dataset_size = max(len(loader.dataset), 1)
        for batch in loader:
            batch = batch.to(device)
            optimizer.zero_grad(set_to_none=True)
            with autocast(enabled=amp_enabled):
                pred_mean, pred_logvar = model(batch)
                loss = criterion(pred_mean, pred_logvar, batch.y, is_warmup=is_warmup)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            total_loss += float(loss.item()) * getattr(batch, "num_graphs", 1)
        return total_loss / dataset_size

    def _evaluate_epoch(
        self,
        *,
        model: nn.Module,
        loader: DataLoader,
        criterion: GaussianNLLPhysicsLoss,
        device: torch.device,
        is_warmup: bool,
    ) -> float:
        model.eval()
        total_loss = 0.0
        dataset_size = max(len(loader.dataset), 1)
        with torch.no_grad():
            for batch in loader:
                batch = batch.to(device)
                pred_mean, pred_logvar = model(batch)
                loss = criterion(pred_mean, pred_logvar, batch.y, is_warmup=is_warmup)
                total_loss += float(loss.item()) * getattr(batch, "num_graphs", 1)
        return total_loss / dataset_size

    def _split_indices(self, dataset_size: int, config: TrainingConfig) -> tuple[list[int], list[int], list[int]]:
        indices = list(range(dataset_size))
        train_val_idx, test_idx = train_test_split(
            indices,
            test_size=config.test_size,
            random_state=config.random_seed,
        )
        val_share = config.val_size / (config.train_size + config.val_size)
        train_idx, val_idx = train_test_split(
            train_val_idx,
            test_size=val_share,
            random_state=config.random_seed,
        )
        return list(train_idx), list(val_idx), list(test_idx)

    def _build_loader(
        self,
        graph_data_list: list[Any],
        indices: list[int],
        config: TrainingConfig,
        *,
        shuffle: bool,
    ) -> DataLoader:
        dataset = [graph_data_list[index] for index in indices]
        return DataLoader(
            dataset,
            batch_size=config.batch_size,
            shuffle=shuffle,
            num_workers=max(config.num_workers, 0),
            pin_memory=bool(config.pin_memory and str(config.device).startswith("cuda")),
        )

    @staticmethod
    def _resolve_device(device_name: str) -> torch.device:
        if device_name.startswith("cuda") and torch.cuda.is_available():
            return torch.device(device_name)
        return torch.device("cpu")

    @staticmethod
    def _set_seed(seed: int) -> None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    def _evaluate_all_splits(
        self,
        *,
        node_features: int,
        scaler_y: Any,
        config: TrainingConfig,
        device: torch.device,
        loaders: dict[str, DataLoader],
        models_dir: Path,
        reports_dir: Path,
    ) -> dict[str, list[str]]:
        metrics_files: list[str] = []
        detail_files: list[str] = []
        models = []
        for model_index in range(1, config.num_models + 1):
            model = ProbabilisticBridgeTrainGNN(
                node_features=node_features,
                train_features_dim=settings.train_feature_dim,
                hidden_dim=config.hidden_dim,
            ).to(device)
            state_dict = torch.load(models_dir / f"ensemble_model_{model_index}.pth", map_location=device)
            model.load_state_dict(state_dict)
            model.eval()
            models.append(model)

        for split_name, loader in loaders.items():
            metrics_path, split_detail_files = self._evaluate_loader(
                split_name=split_name,
                loader=loader,
                models=models,
                scaler_y=scaler_y,
                reports_dir=reports_dir,
            )
            metrics_files.append(str(metrics_path))
            detail_files.extend([str(path) for path in split_detail_files])
        return {"metrics_files": metrics_files, "detail_files": detail_files}

    def _evaluate_loader(
        self,
        *,
        split_name: str,
        loader: DataLoader,
        models: list[ProbabilisticBridgeTrainGNN],
        scaler_y: Any,
        reports_dir: Path,
    ) -> tuple[Path, list[Path]]:
        all_means = []
        all_epi_vars = []
        all_ale_vars = []
        all_targets = []
        all_positions = []
        all_scenarios = []
        device = next(models[0].parameters()).device

        with torch.no_grad():
            for batch in loader:
                batch = batch.to(device)
                batch_means = []
                batch_vars = []
                for model in models:
                    pred_mean, pred_logvar = model(batch)
                    batch_means.append(pred_mean)
                    batch_vars.append(torch.exp(pred_logvar))
                mean_stack = torch.stack(batch_means)
                var_stack = torch.stack(batch_vars)
                ensemble_mean = mean_stack.mean(dim=0)
                aleatoric_var = var_stack.mean(dim=0)
                if len(models) > 1:
                    epistemic_var = mean_stack.var(dim=0, unbiased=True)
                else:
                    epistemic_var = torch.zeros_like(ensemble_mean)

                all_means.append(ensemble_mean.cpu().numpy())
                all_epi_vars.append(epistemic_var.cpu().numpy())
                all_ale_vars.append(aleatoric_var.cpu().numpy())
                all_targets.append(batch.y.cpu().numpy())
                all_positions.append(batch.positions.cpu().numpy())
                if hasattr(batch, "batch") and batch.batch is not None:
                    scenario_values = batch.scenario_id.cpu().numpy().reshape(-1)
                    node_scenarios = scenario_values[batch.batch.cpu().numpy()]
                else:
                    raw_id = batch.scenario_id.cpu().numpy().reshape(-1)[0]
                    node_scenarios = np.full(ensemble_mean.shape[0], raw_id)
                all_scenarios.append(node_scenarios)

        final_mean = np.vstack(all_means)
        final_epi_var = np.vstack(all_epi_vars)
        final_ale_var = np.vstack(all_ale_vars)
        final_targets = np.vstack(all_targets)
        final_positions = np.concatenate(all_positions)
        final_scenarios = np.concatenate(all_scenarios)

        mean_real = scaler_y.inverse_transform(final_mean)
        targets_real = scaler_y.inverse_transform(final_targets)
        scale_sq = np.asarray(scaler_y.scale_, dtype=float) ** 2
        epi_real = final_epi_var * scale_sq
        ale_real = final_ale_var * scale_sq
        total_real = epi_real + ale_real
        std_real = np.sqrt(np.maximum(total_real, 0.0))
        lower = mean_real - settings.confidence_z * std_real
        upper = mean_real + settings.confidence_z * std_real

        metrics_records = []
        detail_paths: list[Path] = []
        for index, variable in enumerate(settings.model_output_names):
            metric_row = compute_evaluation_metrics(
                targets_real[:, index],
                mean_real[:, index],
                lower[:, index],
                upper[:, index],
            )
            metric_row["Variable"] = variable
            metrics_records.append(metric_row)

            detail_path = reports_dir / f"{split_name}_UQ_{variable}.csv"
            pd.DataFrame(
                {
                    "Scenario_ID": final_scenarios,
                    "Position": final_positions,
                    f"True_{variable}": targets_real[:, index],
                    f"Pred_Mean_{variable}": mean_real[:, index],
                    "Lower_95%": lower[:, index],
                    "Upper_95%": upper[:, index],
                    "Epistemic_Var": epi_real[:, index],
                    "Aleatoric_Var": ale_real[:, index],
                    "Total_Var": total_real[:, index],
                }
            ).to_csv(detail_path, index=False)
            detail_paths.append(detail_path)

        metrics_path = reports_dir / f"{split_name}_Evaluation_Metrics.csv"
        metrics_df = pd.DataFrame(metrics_records)
        metrics_df = metrics_df[["Variable", "R2", "RMSE", "MAE", "PICP", "ACE", "PINAW", "MPIW", "IS", "CWC"]]
        metrics_df.to_csv(metrics_path, index=False)
        return metrics_path, detail_paths

    def _copy_runtime_assets(self, run_root: Path, version: str) -> None:
        config_dir = run_root / "config"
        mapping_dir = run_root / "mapping"
        mapping_dir.mkdir(parents=True, exist_ok=True)
        for candidate in (settings.thresholds_path, settings.feature_columns_path, settings.speed_profiles_path):
            if candidate.exists():
                shutil.copy2(candidate, config_dir / candidate.name)
        if settings.mapping_dir.exists():
            shutil.copytree(settings.mapping_dir, mapping_dir, dirs_exist_ok=True)
        manifest_path = config_dir / "asset_manifest.json"
        manifest_payload = {
            "model_version": version,
            "generated_at": datetime.utcnow().isoformat(),
            "notes": "Desktop training output registered by the workbench.",
        }
        manifest_path.write_text(json.dumps(manifest_payload, ensure_ascii=False, indent=2), encoding="utf-8")

    def _write_manifest(self, record: TrainingRunRecord, config: TrainingConfig) -> None:
        payload = {
            "run": record.as_dict(),
            "config": asdict(config),
        }
        Path(record.manifest_path).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    @staticmethod
    def _emit_status(callbacks: TrainingCallbacks, status: str, progress: int, message: str) -> None:
        if callbacks.on_status:
            callbacks.on_status(status, progress, message)

    @staticmethod
    def _emit_log(callbacks: TrainingCallbacks, message: str) -> None:
        if callbacks.on_log:
            callbacks.on_log(message)

    @staticmethod
    def _emit_metrics(callbacks: TrainingCallbacks, payload: dict[str, Any]) -> None:
        if callbacks.on_metrics:
            callbacks.on_metrics(payload)

    @staticmethod
    def _raise_if_cancelled(callbacks: TrainingCallbacks) -> None:
        if callbacks.check_cancelled and callbacks.check_cancelled():
            raise TrainingCancelledError("Training task cancelled by user")
