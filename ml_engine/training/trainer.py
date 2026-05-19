"""
Pulls historical data from storage and trains all ML models per metric.
Can be run once (bootstrap) or on a schedule (every 6h via asyncio).
"""
from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timedelta, timezone

import pandas as pd

from data_pipeline.storage.time_series_store import TimeSeriesStore
from ml_engine.models.isolation_forest import IsolationForestModel
from ml_engine.models.model_registry import ModelRegistry
from ml_engine.models.prophet_model import ProphetModel

logger = logging.getLogger(__name__)

TRAIN_WINDOW_HOURS = 8
MIN_TRAINING_POINTS = 100
RETRAIN_INTERVAL_HOURS = 6


class ModelTrainer:
    """
    Trains IsolationForest and Prophet for every metric in storage.
    Saves fitted models to ModelRegistry.
    """

    def __init__(
        self,
        store: TimeSeriesStore,
        registry: ModelRegistry,
        train_window_hours: float = TRAIN_WINDOW_HOURS,
        min_points: int = MIN_TRAINING_POINTS,
    ):
        self._store = store
        self._registry = registry
        self._train_window_hours = train_window_hours
        self._min_points = min_points
        self._last_trained: dict[str, datetime] = {}

    async def train_all(self, force: bool = False) -> dict[str, bool]:
        """
        Train models for all metrics. Returns dict: metric_name → success.
        Skips metrics that were recently trained (unless force=True).
        """
        metrics = await self._store.list_metrics()
        results: dict[str, bool] = {}
        for metric_name in metrics:
            if not force and self._recently_trained(metric_name):
                logger.debug("Skipping %s (recently trained)", metric_name)
                results[metric_name] = True
                continue
            results[metric_name] = await self._train_metric(metric_name)
        return results

    async def train_metric(self, metric_name: str) -> bool:
        return await self._train_metric(metric_name)

    async def run_scheduler(self) -> None:
        """Background task: re-train all models every RETRAIN_INTERVAL_HOURS hours."""
        logger.info("Trainer scheduler started (interval=%dh)", RETRAIN_INTERVAL_HOURS)
        while True:
            await self.train_all()
            await asyncio.sleep(RETRAIN_INTERVAL_HOURS * 3600)

    # ------------------------------------------------------------------

    async def _train_metric(self, metric_name: str) -> bool:
        now = datetime.now(timezone.utc)
        start = now - timedelta(hours=self._train_window_hours)

        df = await self._store.query_range(metric_name, start=start, end=now)
        if df.empty or len(df) < self._min_points:
            logger.warning(
                "Not enough data for %s: %d points (need %d)",
                metric_name, len(df), self._min_points,
            )
            return False

        train_df = df[["timestamp", "value"]].copy()
        logger.info("Training models for %s (%d points)", metric_name, len(train_df))

        success = True

        # --- IsolationForest ---
        try:
            if_model = IsolationForestModel()
            if_model.fit(train_df)
            if if_model.is_fitted:
                self._registry.save(metric_name, if_model)
                logger.info("  IsolationForest saved for %s", metric_name)
        except Exception as e:
            logger.error("IsolationForest training failed for %s: %s", metric_name, e)
            success = False

        # --- Prophet ---
        try:
            prophet_model = ProphetModel()
            prophet_model.fit(train_df)
            if prophet_model.is_fitted:
                self._registry.save(metric_name, prophet_model)
                logger.info("  Prophet saved for %s", metric_name)
        except Exception as e:
            logger.warning("Prophet training failed for %s: %s", metric_name, e)
            # Prophet failure is non-fatal — IF alone is sufficient

        self._last_trained[metric_name] = now
        return success

    def _recently_trained(self, metric_name: str) -> bool:
        last = self._last_trained.get(metric_name)
        if last is None:
            return False
        return (datetime.now(timezone.utc) - last).total_seconds() < RETRAIN_INTERVAL_HOURS * 3600
