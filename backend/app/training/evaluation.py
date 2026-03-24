"""Offline evaluation helpers that reuse the inference metric definitions."""

from __future__ import annotations

import pandas as pd

from app.core.config import settings
from app.ml_models.boosting_model import compute_evaluation_metrics


def build_metrics_report(dataframe: pd.DataFrame) -> pd.DataFrame:
    records = []
    for variable in settings.warning_output_names:
        true_col = f"True_{variable}"
        pred_col = f"Pred_Mean_{variable}"
        lower_col = "Lower_95%"
        upper_col = "Upper_95%"
        if not {true_col, pred_col, lower_col, upper_col}.issubset(dataframe.columns):
            continue
        metrics = compute_evaluation_metrics(
            dataframe[true_col].to_numpy(),
            dataframe[pred_col].to_numpy(),
            dataframe[lower_col].to_numpy(),
            dataframe[upper_col].to_numpy(),
        )
        metrics["Variable"] = variable
        records.append(metrics)
    return pd.DataFrame(records)

