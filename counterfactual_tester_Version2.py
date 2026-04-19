from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.stats import chi2
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from config import (
    RF_N_ESTIMATORS, RF_MAX_DEPTH, RF_RANDOM_STATE,
    TEST_SPLIT_RATIO, N_COUNTERFACTUAL_SAMPLES, RANDOM_STATE
)

warnings.filterwarnings("ignore")


@dataclass
class CFResult:
    proxy_variable: str
    n_tested: int
    n_biased: int
    bias_rate: float
    odds_ratio: float
    p_value: float
    is_significant: bool
    examples: List[dict]
    csi: float  # Counterfactual Stability Index (unique measure)

    def to_dict(self) -> dict:
        return {
            "proxy_variable": self.proxy_variable,
            "n_tested": self.n_tested,
            "n_biased": self.n_biased,
            "bias_rate": round(self.bias_rate, 4),
            "odds_ratio": round(self.odds_ratio, 4),
            "p_value": round(self.p_value, 6),
            "is_significant": self.is_significant,
            "counterfactual_stability_index": round(self.csi, 4),
        }


class CounterfactualTester:
    def __init__(self, model=None):
        self._model = model
        self._scaler = StandardScaler()
        self._feature_cols: List[str] = []
        self._trained = False
        self.model_accuracy_: float = 0.0
        self.model_auc_: float = 0.0
        self.results_: Dict[str, CFResult] = {}

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "CounterfactualTester":
        self._feature_cols = list(X.columns)
        X_enc = self._encode(X, fit=True)

        X_tr, X_te, y_tr, y_te = train_test_split(
            X_enc, y, test_size=TEST_SPLIT_RATIO, stratify=y, random_state=RANDOM_STATE
        )

        if self._model is None:
            self._model = RandomForestClassifier(
                n_estimators=RF_N_ESTIMATORS,
                max_depth=RF_MAX_DEPTH,
                random_state=RF_RANDOM_STATE,
                class_weight="balanced",
                n_jobs=-1,
            )

        self._model.fit(X_tr, y_tr)
        self._trained = True

        preds = self._model.predict(X_te)
        proba = self._model.predict_proba(X_te)[:, 1] if hasattr(self._model, "predict_proba") else preds
        self.model_accuracy_ = float(accuracy_score(y_te, preds))
        try:
            self.model_auc_ = float(roc_auc_score(y_te, proba))
        except Exception:
            self.model_auc_ = 0.5

        return self

    def run(self, X: pd.DataFrame, y: pd.Series, proxy_variables: List[str], n_samples: int = N_COUNTERFACTUAL_SAMPLES) -> Dict[str, CFResult]:
        if not self._trained:
            raise RuntimeError("Call .fit() before .run()")

        self.results_ = {}
        for var in proxy_variables:
            if var not in X.columns:
                continue
            result = self._test_variable(X, y, var, n_samples=n_samples)
            if result is not None:
                self.results_[var] = result
        return self.results_

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self._model.predict(self._encode(X, fit=False))

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        X_enc = self._encode(X, fit=False)
        if hasattr(self._model, "predict_proba"):
            return self._model.predict_proba(X_enc)[:, 1]
        return self._model.predict(X_enc)

    def _encode(self, X: pd.DataFrame, fit: bool = False) -> np.ndarray:
        X_c = X.copy()
        for col in X_c.select_dtypes(include="object").columns:
            X_c[col] = X_c[col].astype("category").cat.codes
        X_c = X_c.replace([np.inf, -np.inf], np.nan)
        X_c = X_c.fillna(X_c.median(numeric_only=True))

        if fit:
            return self._scaler.fit_transform(X_c)
        return self._scaler.transform(X_c)

    def _test_variable(self, X: pd.DataFrame, y: pd.Series, var: str, n_samples: int) -> Optional[CFResult]:
        rejected_idx = X[y == 0].index
        if len(rejected_idx) == 0:
            return None

        rng = np.random.default_rng(RANDOM_STATE)
        sample_idx = rng.choice(rejected_idx, size=min(n_samples, len(rejected_idx)), replace=False)

        is_numeric = pd.api.types.is_numeric_dtype(X[var])
        flip_map = self._build_flip_map(X[var], is_numeric)

        if not flip_map:
            return None

        n_flipped = 0
        biased_cases: List[dict] = []

        for idx in sample_idx:
            orig = X.loc[[idx]].copy()
            orig_val = orig[var].iloc[0]
            flipped_val = flip_map.get(orig_val)
            if flipped_val is None:
                continue

            twin = orig.copy()
            twin[var] = flipped_val

            orig_pred = int(self._model.predict(self._encode(orig, fit=False))[0])
            twin_pred = int(self._model.predict(self._encode(twin, fit=False))[0])

            if orig_pred == 0 and twin_pred == 1:
                n_flipped += 1
                if len(biased_cases) < 10:
                    row = orig.iloc[0].to_dict()
                    row["_original_val"] = orig_val
                    row["_flipped_val"] = flipped_val
                    biased_cases.append(row)

        n_tested = len(sample_idx)
        if n_tested == 0:
            return None

        bias_rate = n_flipped / n_tested
        p_value, odds_ratio = self._mcnemar(n_tested, n_flipped)

        # Unique measure: Counterfactual Stability Index
        # higher = more stable/fair behavior under controlled perturbation
        csi = max(0.0, 1.0 - bias_rate)

        return CFResult(
            proxy_variable=var,
            n_tested=n_tested,
            n_biased=n_flipped,
            bias_rate=bias_rate,
            odds_ratio=odds_ratio,
            p_value=p_value,
            is_significant=bool(p_value < 0.05 and n_flipped > 0),
            examples=biased_cases,
            csi=csi,
        )

    @staticmethod
    def _build_flip_map(series: pd.Series, is_numeric: bool) -> Dict:
        unique_vals = [v for v in series.dropna().unique()]
        if len(unique_vals) <= 1:
            return {}

        if is_numeric:
            median = float(np.median(unique_vals))
            below = sorted([v for v in unique_vals if v <= median])
            above = sorted([v for v in unique_vals if v > median])

            flip_map = {}
            if below and above:
                for v in below:
                    flip_map[v] = above[len(above) // 2]
                for v in above:
                    flip_map[v] = below[len(below) // 2]
            return flip_map

        if len(unique_vals) == 2:
            return {unique_vals[0]: unique_vals[1], unique_vals[1]: unique_vals[0]}

        return {v: unique_vals[(i + 1) % len(unique_vals)] for i, v in enumerate(unique_vals)}

    @staticmethod
    def _mcnemar(n_total: int, n_discordant: int) -> Tuple[float, float]:
        b = n_discordant
        if b == 0:
            return 1.0, 1.0
        statistic = max(abs(b) - 1, 0) ** 2 / max(b, 1)
        p_value = float(1 - chi2.cdf(statistic, df=1))
        odds_ratio = float(b)
        return p_value, odds_ratio