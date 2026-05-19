"""
Runs ensemble anomaly scoring over the latest N minutes of stored data.
Called by the prediction engine on a schedule to produce fresh AnomalyScores.
"""
from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone

import pandas as pd

from data_pipeline.storage.time_series_store import TimeSeriesStore
from ml_engine.models.isolation_forest import IsolationForestModel
from ml_engine.models.model_registry import ModelRegistry
from ml_engine.models.prophet_model import ProphetModel
from ml_engine.scoring.anomaly_scorer import AnomalyScorer
from ml_engine.models.base_model import AnomalyScore

logger = logging.getLogger(__name__)

SCORE_WINDOW_MINUTES = 60   # how much recent data to score
TRAIN_WINDOW_HOURS = 8      # how much history to feed into models


class BatchScorer:
    """
    For each tracked metric:
      1. Load recent history from storage
      2. Score with all available fitted models
      3. Combine into ensemble score
      4. Return list of AnomalyScore for the prediction engine
    """

    def __init__(self, store: TimeSeriesStore, registry: ModelRegistry):
        self._store = store
        self._registry = registry
        self._scorer = AnomalyScorer()

    async def score_all(self) -> dict[str, list[AnomalyScore]]:
        """
        Score all known metrics. Returns dict: metric_name → [AnomalyScore, ...]
        """
        metrics = await self._store.list_metrics()
        results: dict[str, list[AnomalyScore]] = {}

        for metric_name in metrics:
            scores = await self._score_metric(metric_name)
            if scores:
                results[metric_name] = scores

        return results

    async def score_metric(self, metric_name: str) -> list[AnomalyScore]:
        return await self._score_metric(metric_name)

    async def _score_metric(self, metric_name: str) -> list[AnomalyScore]:
        now = datetime.now(timezone.utc)
        score_start = now - timedelta(minutes=SCORE_WINDOW_MINUTES)

        df = await self._store.query_range(metric_name, start=score_start, end=now)
        if df.empty or len(df) < 5:
            return []

        df_score = df[["timestamp", "value"]].copy()

        # --- IsolationForest ---
        if_model: IsolationForestModel | None = self._registry.get(metric_name, "isolation_forest")
        if_scores = if_model.score(df_score) if if_model and if_model.is_fitted else []

        # --- Prophet ---
        prophet_model: ProphetModel | None = self._registry.get(metric_name, "prophet")
        prophet_scores = prophet_model.score(df_score) if prophet_model and prophet_model.is_fitted else []

        if not if_scores and not prophet_scores:
            logger.debug("No fitted models for %s, skipping scoring", metric_name)
            return []

        # add metric_name column so scorer can stamp it
        df_score["metric_name"] = metric_name

        ensemble = self._scorer.combine(
            if_scores=if_scores,
            prophet_scores=prophet_scores,
            metric_name=metric_name,
        )
        return ensemble

    async def get_latest_score(self, metric_name: str) -> AnomalyScore | None:
        scores = await self._score_metric(metric_name)
        return scores[-1] if scores else None
