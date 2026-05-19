"""
IsolationForest anomaly detector using a sliding window of engineered features.
Unsupervised: no labels required; trained only on normal (baseline) data.
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

from ml_engine.features.feature_engineering import build_features, get_feature_columns
from ml_engine.models.base_model import AnomalyScore, BaseAnomalyModel

logger = logging.getLogger(__name__)


class IsolationForestModel(BaseAnomalyModel):
    """
    Wraps sklearn IsolationForest.
    Converts raw anomaly scores (negative = anomalous) to 0–1 range where
    1.0 means highly anomalous.
    """

    def __init__(
        self,
        contamination: float = 0.05,
        n_estimators: int = 100,
        window_size: int = 30,
        random_state: int = 42,
    ):
        self._contamination = contamination
        self._n_estimators = n_estimators
        self._window_size = window_size
        self._model = IsolationForest(
            contamination=contamination,
            n_estimators=n_estimators,
            random_state=random_state,
        )
        self._scaler = StandardScaler()
        self._fitted = False
        self._feature_cols: list[str] = []

    @property
    def name(self) -> str:
        return "isolation_forest"

    @property
    def is_fitted(self) -> bool:
        return self._fitted

    def fit(self, df: pd.DataFrame) -> None:
        """Train on normal-period data. df: [timestamp, value]"""
        if len(df) < self._window_size * 2:
            logger.warning("Too few points to train IsolationForest (%d < %d)", len(df), self._window_size * 2)
            return

        features = build_features(df).dropna()
        self._feature_cols = get_feature_columns(features)
        X = features[self._feature_cols].values

        X_scaled = self._scaler.fit_transform(X)
        self._model.fit(X_scaled)
        self._fitted = True
        logger.info("IsolationForest fitted on %d points", len(X))

    def score(self, df: pd.DataFrame) -> list[AnomalyScore]:
        """Score each point in df. Returns empty list if model not fitted."""
        if not self._fitted:
            return []

        features = build_features(df)
        available_cols = [c for c in self._feature_cols if c in features.columns]
        X = features[available_cols].fillna(0).values
        X_scaled = self._scaler.transform(X)

        # decision_function returns negative values for anomalies
        raw_scores = self._model.decision_function(X_scaled)

        # normalize to [0, 1]: lower decision_function → higher anomaly score
        min_s, max_s = raw_scores.min(), raw_scores.max()
        if max_s == min_s:
            normalized = np.zeros_like(raw_scores)
        else:
            normalized = 1.0 - (raw_scores - min_s) / (max_s - min_s)

        results = []
        timestamps = df["timestamp"].values if "timestamp" in df.columns else [datetime.now(timezone.utc)] * len(df)
        for i, (ts, score) in enumerate(zip(timestamps, normalized)):
            ts_dt = pd.Timestamp(ts).to_pydatetime() if not isinstance(ts, datetime) else ts
            results.append(
                AnomalyScore(
                    metric_name=df.get("metric_name", pd.Series(["unknown"])).iloc[0] if "metric_name" in df else "unknown",
                    timestamp=ts_dt,
                    score=float(score),
                    model_name=self.name,
                    details={"raw_if_score": float(raw_scores[i])},
                )
            )
        return results
