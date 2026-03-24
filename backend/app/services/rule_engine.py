"""Threshold-based multi-level warning rule engine."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from app.core.config import settings


@dataclass
class RuleDecision:
    code: int
    level: str
    label: str
    color: str
    reason: str
    dominant_variable: str | None
    threshold_ratio: float
    epistemic_peak: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "code": self.code,
            "level": self.level,
            "label": self.label,
            "color": self.color,
            "reason": self.reason,
            "dominant_variable": self.dominant_variable,
            "threshold_ratio": self.threshold_ratio,
            "epistemic_peak": self.epistemic_peak,
        }


class RuleEngine:
    """Apply the project warning logic with the README priority order."""

    status_catalog = {
        "I": {"code": 1, "label": "绿色安全", "color": "#32a852"},
        "II": {"code": 2, "label": "黄色预警", "color": "#f5b700"},
        "III": {"code": 3, "label": "红色预警", "color": "#d7263d"},
        "IV": {"code": 4, "label": "紫色退化报警", "color": "#7b2cbf"},
    }

    def evaluate(self, prediction: dict[str, Any]) -> RuleDecision:
        epistemic_peak = 0.0
        dominant_variable = None
        dominant_ratio = 0.0

        for variable_name in settings.warning_output_names:
            variable_epistemic = max(prediction["epistemic_var"][variable_name], default=0.0)
            if variable_epistemic >= epistemic_peak:
                epistemic_peak = variable_epistemic
                dominant_variable = variable_name

        if epistemic_peak > settings.epistemic_tau:
            base = self.status_catalog["IV"]
            return RuleDecision(
                code=base["code"],
                level="IV",
                label=base["label"],
                color=base["color"],
                reason=f"{dominant_variable} 的认知不确定性超过阈值 {settings.epistemic_tau:.3f}",
                dominant_variable=dominant_variable,
                threshold_ratio=0.0,
                epistemic_peak=epistemic_peak,
            )

        for variable_name in settings.warning_output_names:
            thresholds = settings.variable_thresholds.get(variable_name, {})
            upper_limit = float(thresholds.get("safe_upper", 0.0))
            if upper_limit <= 0:
                continue

            max_upper = max(prediction["upper_95"][variable_name], default=0.0)
            max_mean = max(prediction["mean"][variable_name], default=0.0)
            upper_ratio = max_upper / upper_limit
            mean_ratio = max_mean / upper_limit

            if upper_ratio >= dominant_ratio:
                dominant_ratio = upper_ratio
                dominant_variable = variable_name

            if upper_ratio >= settings.default_red_ratio:
                base = self.status_catalog["III"]
                return RuleDecision(
                    code=base["code"],
                    level="III",
                    label=base["label"],
                    color=base["color"],
                    reason=f"{variable_name} 的95%上界达到阈值的 {upper_ratio:.1%}",
                    dominant_variable=variable_name,
                    threshold_ratio=upper_ratio,
                    epistemic_peak=epistemic_peak,
                )

            if mean_ratio >= settings.default_yellow_ratio:
                base = self.status_catalog["II"]
                return RuleDecision(
                    code=base["code"],
                    level="II",
                    label=base["label"],
                    color=base["color"],
                    reason=f"{variable_name} 的预测均值达到阈值的 {mean_ratio:.1%}",
                    dominant_variable=variable_name,
                    threshold_ratio=mean_ratio,
                    epistemic_peak=epistemic_peak,
                )

        base = self.status_catalog["I"]
        return RuleDecision(
            code=base["code"],
            level="I",
            label=base["label"],
            color=base["color"],
            reason="所有指标均处于安全区间内。",
            dominant_variable=dominant_variable,
            threshold_ratio=dominant_ratio,
            epistemic_peak=epistemic_peak,
        )

