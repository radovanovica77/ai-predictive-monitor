"""
Demo engine: generates simulated failure scenarios, trains models on the fly,
scores incoming data, and pushes predictions to WebSocket clients every 10s.

Timeline:
  T=0s  : Generate 8h of simulated history, store in SQLite
  T=5s  : Train IsolationForest on first 4h (normal period)
  T=10s : Start replaying remaining 4h at high speed; score & predict every 10s
"""
from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timedelta, timezone
from typing import Any

import pandas as pd

from data_pipeline.pipeline import DataPipeline
from data_pipeline.simulator.metric_simulator import MetricSimulator
from data_pipeline.storage.sqlite_backend import SQLiteStore
from ml_engine.models.isolation_forest import IsolationForestModel
from ml_engine.models.model_registry import ModelRegistry
from ml_engine.scoring.batch_scorer import BatchScorer
from ml_engine.training.trainer import ModelTrainer
from prediction_engine.predictor import Predictor

logger = logging.getLogger(__name__)

HISTORY_HOURS = 8       # total scenario duration
NORMAL_HOURS = 4        # first N hours = "normal" training period
REPLAY_STEP_MINUTES = 2 # minutes of simulated time added per real-time tick
TICK_SECONDS = 10       # real-time interval between ticks


class DemoEngine:
    """
    Encapsulates the full simulation-to-prediction pipeline for the dashboard demo.
    Call start() from the FastAPI lifespan handler.
    """

    def __init__(self, db_path: str = "data/metrics.db"):
        self.store = SQLiteStore(db_path=db_path)
        self.registry = ModelRegistry("data/models")
        self.scorer = BatchScorer(self.store, self.registry)
        self.predictor = Predictor(self.store)
        self._current_predictions: list[dict] = []
        self._metrics_summary: dict[str, Any] = {}
        self._ready = False
        self._broadcast_callback = None

    def set_broadcast_callback(self, fn) -> None:
        self._broadcast_callback = fn

    @property
    def is_ready(self) -> bool:
        return self._ready

    @property
    def current_predictions(self) -> list[dict]:
        return self._current_predictions

    @property
    def metrics_summary(self) -> dict[str, Any]:
        return self._metrics_summary

    async def start(self) -> None:
        """Entry point — runs the full demo lifecycle as a background coroutine."""
        logger.info("Demo engine starting...")
        await self._generate_history()
        await self._train_models()
        self._ready = True
        logger.info("Demo engine ready. Starting replay loop.")
        await self._replay_loop()

    # ------------------------------------------------------------------

    async def _generate_history(self) -> None:
        sim = MetricSimulator(seed=42)
        start = datetime.now(timezone.utc) - timedelta(hours=HISTORY_HOURS)

        logger.info("Generating %dh of simulated history...", HISTORY_HOURS)
        pipeline = DataPipeline(self.store)

        # generate all points, store in batches (no speed throttle)
        all_points = sim.generate_all(start=start, speed_multiplier=1)
        # batch write
        batch_size = 500
        for i in range(0, len(all_points), batch_size):
            await self.store.write(all_points[i : i + batch_size])

        count = len(all_points)
        logger.info("Stored %d metric points across %dh scenario.", count, HISTORY_HOURS)

    async def _train_models(self) -> None:
        trainer = ModelTrainer(
            store=self.store,
            registry=self.registry,
            train_window_hours=NORMAL_HOURS,
            min_points=80,
        )
        logger.info("Training models on first %dh (normal period)...", NORMAL_HOURS)
        results = await trainer.train_all(force=True)
        trained = [m for m, ok in results.items() if ok]
        logger.info("Trained models for: %s", trained)

    async def _replay_loop(self) -> None:
        """
        Continuously generate new data points (simulating time passing),
        score them, produce predictions, and broadcast to WebSocket clients.
        """
        sim = MetricSimulator(seed=99)
        cursor = datetime.now(timezone.utc) - timedelta(hours=1)

        while True:
            try:
                # advance simulated time by REPLAY_STEP_MINUTES
                new_start = cursor
                new_end = cursor + timedelta(minutes=REPLAY_STEP_MINUTES)
                cursor = new_end

                # generate new metric points for this time slice
                new_points = []
                for scenario_gen in [
                    sim.connection_pool_exhaustion(start=new_start, duration_hours=REPLAY_STEP_MINUTES / 60, step_seconds=30),
                    sim.memory_leak(start=new_start, duration_hours=REPLAY_STEP_MINUTES / 60, step_seconds=30),
                    sim.traffic_spike(start=new_start, duration_hours=REPLAY_STEP_MINUTES / 60, step_seconds=15),
                    sim.latency_degradation(start=new_start, duration_hours=REPLAY_STEP_MINUTES / 60, step_seconds=30),
                    sim.disk_fill(start=new_start, duration_hours=REPLAY_STEP_MINUTES / 60, step_seconds=60),
                ]:
                    new_points.extend(scenario_gen)

                if new_points:
                    await self.store.write(new_points)

                # score and predict
                anomaly_scores = await self.scorer.score_all()
                predictions = await self.predictor.predict_all(anomaly_scores)
                metrics = await self._build_metrics_summary()

                self._current_predictions = [p.to_dict() for p in predictions]
                self._metrics_summary = metrics

                if self._broadcast_callback and predictions:
                    await self._broadcast_callback({
                        "type": "predictions_update",
                        "predictions": self._current_predictions,
                        "metrics_summary": metrics,
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                        "active_count": len(predictions),
                    })

            except Exception as e:
                logger.error("Replay loop error: %s", e, exc_info=True)

            await asyncio.sleep(TICK_SECONDS)

    async def _build_metrics_summary(self) -> dict[str, Any]:
        metrics = await self.store.list_metrics()
        summary: dict[str, Any] = {}
        now = datetime.now(timezone.utc)

        for metric_name in metrics:
            latest = await self.store.get_latest(metric_name)
            if latest:
                summary[metric_name] = {
                    "current_value": round(latest.value, 2),
                    "timestamp": latest.timestamp.isoformat(),
                }
        return summary
