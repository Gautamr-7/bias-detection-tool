from __future__ import annotations
from dataclasses import dataclass
from typing import Dict
import numpy as np
import pandas as pd

from config import (
    DEMOGRAPHIC_PARITY_TOLERANCE,
    EQUALIZED_ODDS_TOLERANCE,
    PREDICTIVE_PARITY_TOLERANCE,
    CALIBRATION_TOLERANCE,
    DISPARATE_IMPACT_MIN,
)


@dataclass
class MetricResult:
    name: str
    value: float
    passed: bool
    details: Dict
    description: str
    tolerance: float

    @property
    def status(self) -> str:
        return "pass" if self.passed else "fail"

    @property
    def severity(self) -> str:
        if self.passed:
            return "good"
        if self.value > self.tolerance * 2:
            return "critical"
        return "warning"


class FairnessMetrics:
    def __init__(self, y_true, y_pred, y_proba=None, groups=None):
        self.y_true = pd.Series(y_true).reset_index(drop=True)
        self.y_pred = pd.Series(y_pred).reset_index(drop=True)
        self.y_proba = pd.Series(y_proba).reset_index(drop=True) if y_proba is not None else None
        self.groups = groups or {}
        self.results_: Dict[str, MetricResult] = {}
        self.overall_score_: float = 0.0

    def compute_all(self) -> "FairnessMetrics":
        results = {}

        for attr, group_series in self.groups.items():
            g = pd.Series(group_series).reset_index(drop=True)

            results[f"dp_{attr}"] = self._demographic_parity(g, attr)
            results[f"di_{attr}"] = self._disparate_impact(g, attr)
            results[f"eo_{attr}"] = self._equalized_odds(g, attr)
            results[f"pp_{attr}"] = self._predictive_parity(g, attr)

            if self.y_proba is not None:
                results[f"cal_{attr}"] = self._calibration(g, attr)

        self.results_ = results
        self.overall_score_ = self._score()
        return self

    def summary_df(self) -> pd.DataFrame:
        return pd.DataFrame([{
            "metric": r.name,
            "value": round(r.value, 4),
            "passed": r.passed,
            "severity": r.severity,
            "tolerance": r.tolerance,
            "description": r.description,
        } for r in self.results_.values()])

    def _demographic_parity(self, g, attr):
        rates = {str(grp): float(self.y_pred[g == grp].mean()) for grp in g.unique()}
        gap = max(rates.values()) - min(rates.values()) if len(rates) > 1 else 0.0
        return MetricResult(
            name=f"Demographic Parity ({attr})",
            value=gap,
            passed=gap <= DEMOGRAPHIC_PARITY_TOLERANCE,
            details=rates,
            description="Max approval rate gap across groups.",
            tolerance=DEMOGRAPHIC_PARITY_TOLERANCE,
        )

    def _disparate_impact(self, g, attr):
        rates = [float(self.y_pred[g == grp].mean()) for grp in g.unique()]
        rates = [r for r in rates if not np.isnan(r)]
        if len(rates) < 2 or max(rates) == 0:
            di = 1.0
        else:
            di = min(rates) / max(rates)

        passed = di >= DISPARATE_IMPACT_MIN
        # Convert to distance-from-threshold for severity scoring style
        value = abs(DISPARATE_IMPACT_MIN - di) if not passed else 0.0

        return MetricResult(
            name=f"Disparate Impact ({attr})",
            value=value,
            passed=passed,
            details={"ratio": di},
            description="Min/max approval ratio (4/5ths rule).",
            tolerance=1 - DISPARATE_IMPACT_MIN,
        )

    def _equalized_odds(self, g, attr):
        tpr_r, fpr_r = {}, {}
        for grp in g.unique():
            mask = g == grp
            yt, yp = self.y_true[mask], self.y_pred[mask]

            tp = int(((yt == 1) & (yp == 1)).sum())
            fp = int(((yt == 0) & (yp == 1)).sum())
            tn = int(((yt == 0) & (yp == 0)).sum())
            fn = int(((yt == 1) & (yp == 0)).sum())

            tpr_r[str(grp)] = tp / max(tp + fn, 1)
            fpr_r[str(grp)] = fp / max(fp + tn, 1)

        tpr_gap = max(tpr_r.values()) - min(tpr_r.values()) if tpr_r else 0.0
        fpr_gap = max(fpr_r.values()) - min(fpr_r.values()) if fpr_r else 0.0
        gap = max(tpr_gap, fpr_gap)

        return MetricResult(
            name=f"Equalized Odds ({attr})",
            value=gap,
            passed=gap <= EQUALIZED_ODDS_TOLERANCE,
            details={"tpr": tpr_r, "fpr": fpr_r, "tpr_gap": tpr_gap, "fpr_gap": fpr_gap},
            description="Max TPR/FPR gap across groups.",
            tolerance=EQUALIZED_ODDS_TOLERANCE,
        )

    def _predictive_parity(self, g, attr):
        ppv_r = {}
        for grp in g.unique():
            mask = g == grp
            yt, yp = self.y_true[mask], self.y_pred[mask]
            tp = int(((yt == 1) & (yp == 1)).sum())
            fp = int(((yt == 0) & (yp == 1)).sum())
            ppv_r[str(grp)] = tp / max(tp + fp, 1)

        gap = max(ppv_r.values()) - min(ppv_r.values()) if len(ppv_r) > 1 else 0.0
        return MetricResult(
            name=f"Predictive Parity ({attr})",
            value=gap,
            passed=gap <= PREDICTIVE_PARITY_TOLERANCE,
            details=ppv_r,
            description="Max precision gap across groups.",
            tolerance=PREDICTIVE_PARITY_TOLERANCE,
        )

    def _calibration(self, g, attr):
        cal_e = {}
        for grp in g.unique():
            mask = g == grp
            actual = float(self.y_true[mask].mean())
            pred = float(self.y_proba[mask].mean())
            cal_e[str(grp)] = abs(actual - pred)

        max_e = max(cal_e.values()) if cal_e else 0.0
        return MetricResult(
            name=f"Calibration ({attr})",
            value=max_e,
            passed=max_e <= CALIBRATION_TOLERANCE,
            details=cal_e,
            description="Max gap between predicted probability and actual outcome rate.",
            tolerance=CALIBRATION_TOLERANCE,
        )

    def _score(self) -> float:
        if not self.results_:
            return 100.0

        n_pass = sum(1 for r in self.results_.values() if r.passed)
        score = (n_pass / len(self.results_)) * 100

        for r in self.results_.values():
            if r.severity == "critical":
                score -= 6
            elif r.severity == "warning":
                score -= 2

        return float(max(0.0, min(100.0, score)))