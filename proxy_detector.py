"""
ProxyDetector — identifies suspicious proxy variables that may encode protected attributes.
"""

from __future__ import annotations
import warnings
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, chi2_contingency
from sklearn.metrics import mutual_info_score
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import resample

from config import (
    PROXY_SUSPICION_THRESHOLD,
    PROXY_HIGH_RISK_THRESHOLD,
    TOP_K_PROXIES,
    BOOTSTRAP_ITER,
)

warnings.filterwarnings("ignore")


@dataclass
class ProxyFinding:
    feature: str
    suspicion_score: float
    pearson_r: Optional[float]
    cramers_v: Optional[float]
    mutual_info: float
    approval_by_bin: Dict[str, float]
    is_high_risk: bool
    confidence_interval: Tuple[float, float] = field(default=(0.0, 0.0))

    @property
    def severity(self) -> str:
        if self.suspicion_score >= PROXY_HIGH_RISK_THRESHOLD:
            return "critical"
        if self.suspicion_score >= PROXY_SUSPICION_THRESHOLD:
            return "warning"
        return "ok"

    def to_dict(self) -> dict:
        return {
            "feature": self.feature,
            "suspicion_score": round(self.suspicion_score, 4),
            "pearson_r": round(self.pearson_r, 4) if self.pearson_r is not None else None,
            "cramers_v": round(self.cramers_v, 4) if self.cramers_v is not None else None,
            "mutual_info": round(self.mutual_info, 4),
            "severity": self.severity,
            "approval_by_bin": {k: round(v, 3) for k, v in self.approval_by_bin.items()},
            "ci_low": round(self.confidence_interval[0], 4),
            "ci_high": round(self.confidence_interval[1], 4),
        }


class ProxyDetector:
    def __init__(self, n_bootstrap: int = BOOTSTRAP_ITER):
        self.n_bootstrap = n_bootstrap
        self.findings_: List[ProxyFinding] = []
        self._encoders: Dict[str, LabelEncoder] = {}

    def fit(self, X: pd.DataFrame, y: pd.Series, exclude_cols=None) -> "ProxyDetector":
        exclude = set(exclude_cols or [])
        self.findings_ = []

        for col in X.columns:
            if col in exclude:
                continue
            finding = self._analyse_feature(X[col], y)
            if finding:
                self.findings_.append(finding)

        self.findings_.sort(key=lambda f: f.suspicion_score, reverse=True)
        self.findings_ = self.findings_[:TOP_K_PROXIES]
        return self

    def suspicious_features(self) -> List[str]:
        return [f.feature for f in self.findings_ if f.suspicion_score >= PROXY_SUSPICION_THRESHOLD]

    def high_risk_features(self) -> List[str]:
        return [f.feature for f in self.findings_ if f.is_high_risk]

    def summary_df(self) -> pd.DataFrame:
        return pd.DataFrame([f.to_dict() for f in self.findings_])

    def _analyse_feature(self, series: pd.Series, y: pd.Series) -> Optional[ProxyFinding]:
        series = series.copy()
        is_numeric = pd.api.types.is_numeric_dtype(series)
        if series.nunique(dropna=True) <= 1:
            return None

        if not is_numeric:
            enc = LabelEncoder()
            encoded = enc.fit_transform(series.astype(str))
            self._encoders[series.name] = enc
        else:
            encoded = series.fillna(series.median()).values

        mi = float(mutual_info_score(y.values, encoded))
        pearson = None
        cramers = None

        if is_numeric:
            try:
                r, _ = pearsonr(series.fillna(series.median()), y)
                pearson = abs(float(r))
            except Exception:
                pearson = 0.0
        else:
            cramers = self._cramers_v(series, y)

        # Balanced weighted score (more emphasis on MI)
        stat_values = [v for v in [pearson, cramers] if v is not None]
        corr_component = float(np.mean(stat_values)) if stat_values else 0.0
        suspicion_score = float(np.clip(0.4 * corr_component + 0.6 * min(mi, 1.0), 0.0, 1.0))

        approval_by_bin = self._approval_by_bin(series, y, is_numeric)
        ci = self._bootstrap_ci(series, y, is_numeric)

        return ProxyFinding(
            feature=series.name,
            suspicion_score=suspicion_score,
            pearson_r=pearson,
            cramers_v=cramers,
            mutual_info=mi,
            approval_by_bin=approval_by_bin,
            is_high_risk=suspicion_score >= PROXY_HIGH_RISK_THRESHOLD,
            confidence_interval=ci,
        )

    @staticmethod
    def _cramers_v(series: pd.Series, y: pd.Series) -> float:
        contingency = pd.crosstab(series.astype(str), y)
        try:
            chi2, _, _, _ = chi2_contingency(contingency)
        except Exception:
            return 0.0

        n = contingency.to_numpy().sum()
        k = min(contingency.shape) - 1
        if n == 0 or k <= 0:
            return 0.0
        return float(np.sqrt(chi2 / (n * k)))

    @staticmethod
    def _approval_by_bin(series: pd.Series, y: pd.Series, is_numeric: bool) -> Dict[str, float]:
        if is_numeric and series.nunique(dropna=True) > 6:
            try:
                bins = pd.qcut(series, q=4, duplicates="drop")
            except Exception:
                bins = pd.cut(series, bins=4)
            grouped = y.groupby(bins).mean()
        else:
            grouped = y.groupby(series.astype(str)).mean()
        return {str(k): float(v) for k, v in grouped.items() if pd.notna(v)}

    def _bootstrap_ci(self, series: pd.Series, y: pd.Series, is_numeric: bool, alpha=0.05) -> Tuple[float, float]:
        scores = []
        data = pd.concat([series.rename("feature"), y.rename("target")], axis=1).dropna()

        if len(data) < 30:
            return (0.0, 0.0)

        for _ in range(self.n_bootstrap):
            sample = resample(data)
            s = sample["feature"]
            target = sample["target"]

            if is_numeric:
                s_num = pd.to_numeric(s, errors="coerce").fillna(pd.to_numeric(s, errors="coerce").median())
                try:
                    r, _ = pearsonr(s_num, target)
                    corr = abs(float(r))
                except Exception:
                    corr = 0.0
                mi = float(mutual_info_score(target, pd.qcut(s_num, q=10, duplicates="drop").astype(str)))
                score = 0.4 * corr + 0.6 * min(mi, 1.0)
            else:
                s_cat = s.astype(str)
                mi = float(mutual_info_score(target, s_cat))
                v = self._cramers_v(s_cat, target)
                score = 0.4 * v + 0.6 * min(mi, 1.0)

            scores.append(score)

        arr = np.array(scores)
        return (float(np.percentile(arr, alpha / 2 * 100)), float(np.percentile(arr, (1 - alpha / 2) * 100)))
