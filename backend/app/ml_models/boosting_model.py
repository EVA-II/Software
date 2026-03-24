"""Probabilistic PI-STGNN model and evaluation helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from app.core.python_compat import patch_typing_extensions_self

patch_typing_extensions_self()

from torch_geometric.data import Data
from torch_geometric.nn import GATConv, GCNConv


def create_edge_connections(num_nodes: int) -> torch.Tensor:
    """Create graph edges using physical adjacency, jump, and support links."""
    edges: set[tuple[int, int]] = set()
    interval = 0.5

    for index in range(num_nodes - 1):
        edges.add((index, index + 1))
        edges.add((index + 1, index))

    jump_size = 5
    for index in range(num_nodes - jump_size):
        edges.add((index, index + jump_size))
        edges.add((index + jump_size, index))

    supports = [59.25, 175.25, 1195.25, 1311.25]
    support_range = 10
    for support in supports:
        support_node = int(round(support / interval))
        support_node = min(max(support_node, 0), max(num_nodes - 1, 0))
        for neighbor in range(
            max(0, support_node - support_range),
            min(num_nodes, support_node + support_range + 1),
        ):
            if neighbor == support_node:
                continue
            edges.add((support_node, neighbor))
            edges.add((neighbor, support_node))

    edge_list = list(edges)
    source_nodes = [edge[0] for edge in edge_list]
    target_nodes = [edge[1] for edge in edge_list]
    return torch.tensor([source_nodes, target_nodes], dtype=torch.long)


def build_graph_sample(
    scaled_features: np.ndarray,
    positions: np.ndarray,
    scenario_id: str,
    targets: np.ndarray | None = None,
) -> Data:
    """Wrap tabular node features into the graph format expected by the model."""
    graph = Data(
        x=torch.tensor(scaled_features, dtype=torch.float32),
        edge_index=create_edge_connections(len(positions)),
        scenario_id=scenario_id,
        positions=torch.tensor(positions, dtype=torch.float32),
    )
    if targets is not None:
        graph.y = torch.tensor(targets, dtype=torch.float32)
    return graph


class ProbabilisticBridgeTrainGNN(nn.Module):
    """Probability-aware PI-STGNN used during ensemble inference."""

    def __init__(self, node_features: int, train_features_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.node_features = node_features
        self.train_features_dim = train_features_dim
        self.hidden_dim = hidden_dim

        self.conv1 = GCNConv(node_features, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.conv3 = GCNConv(hidden_dim, hidden_dim)
        self.bn3 = nn.BatchNorm1d(hidden_dim)

        self.gat = GATConv(hidden_dim, hidden_dim, heads=8, concat=True, dropout=0.3)
        self.gat_projection = nn.Linear(hidden_dim * 8, hidden_dim)

        self.train_feat_linear = nn.Linear(train_features_dim, hidden_dim)
        self.fusion = nn.Linear(hidden_dim * 2, hidden_dim)
        self.shared_features = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(0.3)

        self.lstm_unloading = nn.LSTM(
            hidden_dim,
            hidden_dim,
            num_layers=2,
            batch_first=True,
            dropout=0.3,
            bidirectional=True,
        )
        self.lstm_derailment = nn.LSTM(
            hidden_dim,
            hidden_dim,
            num_layers=2,
            batch_first=True,
            dropout=0.3,
            bidirectional=True,
        )

        self.acc_mean_head = nn.Linear(hidden_dim, 1)
        self.acc_logvar_head = nn.Linear(hidden_dim, 1)
        self.force_mean_head = nn.Linear(hidden_dim, 1)
        self.force_logvar_head = nn.Linear(hidden_dim, 1)

        self.unloading_mean_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LeakyReLU(0.1),
            nn.Linear(hidden_dim, 1),
        )
        self.unloading_logvar_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LeakyReLU(0.1),
            nn.Linear(hidden_dim, 1),
        )
        self.derailment_mean_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, 1),
        )
        self.derailment_logvar_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, 1),
        )

    def process_lstm_features(self, shared: torch.Tensor, lstm_layer: nn.LSTM, data: Data) -> torch.Tensor:
        batch_idx = getattr(data, "batch", None)
        if batch_idx is not None:
            if batch_idx.numel() > 0:
                num_graphs = batch_idx.max().item() + 1
                all_features = []
                for index in range(num_graphs):
                    mask = batch_idx == index
                    if mask.sum() > 0:
                        current = shared[mask]
                        if current.size(0) > 0:
                            all_features.append(current.unsqueeze(0))
                lstm_outputs = []
                for feature in all_features:
                    if feature.size(1) > 0:
                        lstm_out, _ = lstm_layer(feature)
                        lstm_outputs.append(lstm_out.squeeze(0))
                if lstm_outputs:
                    return torch.cat(lstm_outputs, dim=0)
            return torch.cat([shared, torch.zeros_like(shared)], dim=1)

        if shared.size(0) > 0:
            lstm_input = shared.unsqueeze(0)
            lstm_out, _ = lstm_layer(lstm_input)
            return lstm_out.squeeze(0)
        return torch.cat([shared, torch.zeros_like(shared)], dim=1)

    def forward(self, data: Data) -> tuple[torch.Tensor, torch.Tensor]:
        x, edge_index = data.x, data.edge_index
        bridge_features = x[:, :-self.train_features_dim] if self.train_features_dim > 0 else x
        train_features = (
            x[:, -self.train_features_dim :]
            if self.train_features_dim > 0
            else torch.tensor([], device=x.device)
        )

        x = F.relu(self.bn1(self.conv1(bridge_features, edge_index)))
        x = self.dropout(x)
        residual = x
        x = F.relu(self.bn2(self.conv2(x, edge_index)))
        x = self.dropout(x)
        x = F.relu(self.bn3(self.conv3(x, edge_index)) + residual)
        x = F.relu(self.gat_projection(self.gat(x, edge_index)))

        if self.train_features_dim > 0:
            train_feat = F.relu(self.train_feat_linear(train_features))
            x = F.relu(self.fusion(torch.cat([x, train_feat], dim=1)))

        shared = F.relu(self.shared_features(x))
        unloading_features = self.process_lstm_features(shared, self.lstm_unloading, data)
        derailment_features = self.process_lstm_features(shared, self.lstm_derailment, data)

        acc_mean = self.acc_mean_head(shared)
        acc_logvar = self.acc_logvar_head(shared)
        force_mean = self.force_mean_head(shared)
        force_logvar = self.force_logvar_head(shared)
        unloading_mean = self.unloading_mean_head(unloading_features)
        unloading_logvar = self.unloading_logvar_head(unloading_features)
        derail_mean = self.derailment_mean_head(derailment_features)
        derail_logvar = self.derailment_logvar_head(derailment_features)

        pred_mean = torch.cat([acc_mean, derail_mean, unloading_mean, force_mean], dim=1)
        pred_logvar = torch.cat([acc_logvar, derail_logvar, unloading_logvar, force_logvar], dim=1)
        return pred_mean, pred_logvar


def compute_evaluation_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_lower: np.ndarray,
    y_upper: np.ndarray,
    confidence_level: float = 0.95,
    eta: int = 50,
) -> dict[str, float]:
    """Compute point and interval prediction metrics."""
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

    r2 = r2_score(y_true, y_pred)
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))

    covered = (y_true >= y_lower) & (y_true <= y_upper)
    picp = float(np.mean(covered))
    ace = float(abs(picp - confidence_level))
    mpiw = float(np.mean(y_upper - y_lower))
    range_y = float(np.max(y_true) - np.min(y_true) + 1e-8)
    pinaw = float(mpiw / range_y)

    alpha = 1.0 - confidence_level
    penalty_low = (2.0 / alpha) * (y_lower - y_true) * (y_true < y_lower)
    penalty_high = (2.0 / alpha) * (y_true - y_upper) * (y_true > y_upper)
    interval_score = float(np.mean((y_upper - y_lower) + penalty_low + penalty_high))

    gamma = 1 if picp < confidence_level else 0
    cwc = float(pinaw + gamma * np.exp(-eta * (picp - confidence_level)))

    return {
        "R2": float(r2),
        "RMSE": rmse,
        "MAE": mae,
        "PICP": picp,
        "ACE": ace,
        "PINAW": pinaw,
        "MPIW": mpiw,
        "IS": interval_score,
        "CWC": cwc,
    }


@dataclass
class EnsemblePrediction:
    scenario_id: str
    positions: list[float]
    mean: dict[str, list[float]]
    aleatoric_var: dict[str, list[float]]
    epistemic_var: dict[str, list[float]]
    total_var: dict[str, list[float]]
    lower_95: dict[str, list[float]]
    upper_95: dict[str, list[float]]
    ground_truth: dict[str, list[float]] | None
    metadata: dict[str, Any]
