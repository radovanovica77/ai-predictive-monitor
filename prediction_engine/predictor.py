"""
Main prediction engine orchestrator.

Flow per metric:
  storage → recent_df → horizon_estimator → confidence_scorer → template → Prediction

The predictor maps metric names to scenarios and thresholds automatically.
"""
from __future__ import annotations

import logging
import uuid
from datetime import datetime, timedelta, timezone
from typing import Any

import pandas as pd

from data_pipeline.storage.time_series_store import TimeSeriesStore
from ml_engine.models.base_model import AnomalyScore
from prediction_engine.confidence_scorer import (
    compute_confidence,
    severity_from_confidence_and_horizon,
)
from prediction_engine.horizon_estimator import estimate_horizon
from prediction_engine.models import (
    Evidence,
    Prediction,
    Scenario,
    Severity,
    TrendInfo,
)
from prediction_engine.templates.prediction_templates import render_message

logger = logging.getLogger(__name__)

# How long a prediction stays active before it's considered stale
PREDICTION_TTL_HOURS = 2

# Minimum anomaly score to generate a prediction (avoid noise)
MIN_ANOMALY_SCORE = 0.45

# How many hours of history to pull for horizon estimation
HISTORY_WINDOW_HOURS = 2


# ------------------------------------------------------------------
# Metric → Scenario + Threshold routing table
# ------------------------------------------------------------------

_METRIC_CONFIG: dict[str, dict[str, Any]] = {
    "db.connections.active": {
        "scenario": Scenario.CONNECTION_POOL_EXHAUSTION,
        "threshold": 100,
        "direction": "above",
        "contributing": ["http.request_rate", "db.query_duration_p95_ms"],
        "extra": {},
    },
    "db.connections.pct": {
        "scenario": Scenario.CONNECTION_POOL_EXHAUSTION,
        "threshold": 90,
        "direction": "above",
        "contributing": ["http.request_rate"],
        "extra": {},
    },
    "process.memory_mb": {
        "scenario": Scenario.MEMORY_LEAK,
        "threshold": 7000,
        "direction": "above",
        "contributing": ["process.memory_pct"],
        "extra": {},
    },
    "process.memory_pct": {
        "scenario": Scenario.MEMORY_PRESSURE,
        "threshold": 85,
        "direction": "above",
        "contributing": [],
        "extra": {},
    },
    "http.request_rate": {
        "scenario": Scenario.TRAFFIC_SPIKE,
        "threshold": 4000,
        "direction": "above",
        "contributing": ["http.latency_p99_ms", "http.error_rate_pct"],
        "extra": {"baseline_rps": 500},
    },
    "http.latency_p99_ms": {
        "scenario": Scenario.LATENCY_DEGRADATION,
        "threshold": 1000,
        "direction": "above",
        "contributing": ["http.request_rate", "db.query_duration_p95_ms"],
        "extra": {"endpoint": "API", "percentile": "p99", "baseline_ms": 50},
    },
    "http.latency_p95_ms": {
        "scenario": Scenario.LATENCY_SLA_BREACH,
        "threshold": 800,
        "direction": "above",
        "contributing": ["http.request_rate"],
        "extra": {"endpoint": "API", "percentile": "p95", "baseline_ms": 40},
    },
    "disk.usage_pct": {
        "scenario": Scenario.DISK_FILL,
        "threshold": 90,
        "direction": "above",
        "contributing": [],
        "extra": {"mount": "/var/data"},
    },
    "system.cpu_pct": {
        "scenario": Scenario.CPU_SATURATION,
        "threshold": 90,
        "direction": "above",
        "contributing": ["http.request_rate"],
        "extra": {},
    },
    "http.error_rate_pct": {
        "scenario": Scenario.ERROR_RATE_SPIKE,
        "threshold": 5,
        "direction": "above",
        "contributing": ["http.request_rate", "http.latency_p99_ms"],
        "extra": {},
    },
    "system.memory_pct": {
        "scenario": Scenario.MEMORY_PRESSURE,
        "threshold": 85,
        "direction": "above",
        "contributing": [],
        "extra": {},
    },
}


class Predictor:
    """
    Generates Prediction objects for all metrics that have anomaly scores
    above the threshold and a detectable trend toward a critical threshold.
    """

    def __init__(
        self,
        store: TimeSeriesStore,
        min_anomaly_score: float = MIN_ANOMALY_SCORE,
        min_confidence: float = 0.45,
    ):
        self._store = store
        self._min_anomaly_score = min_anomaly_score
        self._min_confidence = min_confidence

    async def predict_all(
        self,
        anomaly_scores: dict[str, list[AnomalyScore]],
    ) -> list[Prediction]:
        """
        Generate predictions for all metrics that have both:
          - a recent anomaly score above threshold
          - a detected trend toward a critical threshold
        """
        predictions: list[Prediction] = []

        for metric_name, scores in anomaly_scores.items():
            if not scores:
                continue
            latest_score = scores[-1]
            if latest_score.score < self._min_anomaly_score:
                continue

            if metric_name not in _METRIC_CONFIG:
                logger.debug("No config for metric %s, skipping", metric_name)
                continue

            prediction = await self._predict_metric(metric_name, latest_score)
            if prediction is not None:
                predictions.append(prediction)

        # sort by severity and then confidence (most urgent first)
        severity_order = {
            Severity.CRITICAL: 0,
            Severity.HIGH: 1,
            Severity.MEDIUM: 2,
            Severity.LOW: 3,
        }
        predictions.sort(
            key=lambda p: (severity_order[p.severity], -p.confidence)
        )
        return predictions

    async def predict_metric(
        self,
        metric_name: str,
        anomaly_score: AnomalyScore,
    ) -> Prediction | None:
        return await self._predict_metric(metric_name, anomaly_score)

    async def _predict_metric(
        self,
        metric_name: str,
        anomaly_score: AnomalyScore,
    ) -> Prediction | None:
        config = _METRIC_CONFIG[metric_name]
        threshold = config["threshold"]
        direction = config["direction"]
        scenario = config["scenario"]

        # pull recent history for regression
        now = datetime.now(timezone.utc)
        hist_start = now - timedelta(hours=HISTORY_WINDOW_HOURS)
        df = await self._store.query_range(metric_name, start=hist_start, end=now)

        if df.empty or len(df) < 10:
            logger.debug("Not enough history for %s (%d points)", metric_name, len(df))
            return None

        recent_df = df[["timestamp", "value"]].copy()
        current_value = float(recent_df["value"].iloc[-1])

        # estimate time to threshold
        horizon_hours, trend = estimate_horizon(
            recent_df, threshold=threshold, direction=direction
        )

        if horizon_hours is None:
            logger.debug("%s: metric moving away from threshold, no prediction", metric_name)
            return None

        # compute confidence
        current_pct = (current_value / threshold * 100) if threshold > 0 else 0
        confidence = compute_confidence(
            anomaly_score=anomaly_score.score,
            trend=trend,
            horizon_hours=horizon_hours,
            min_confidence=self._min_confidence,
        )

        if confidence < self._min_confidence:
            return None

        # derive severity
        severity = severity_from_confidence_and_horizon(
            confidence=confidence,
            horizon_hours=horizon_hours,
            current_pct_of_threshold=current_pct,
        )

        # assemble evidence
        free_gb = None
        if metric_name == "disk.usage_pct":
            free_gb = (100 - current_value) / 100 * 500
        extra = dict(config["extra"])
        if free_gb is not None:
            extra["free_gb"] = free_gb

        evidence = Evidence(
            current_value=round(current_value, 2),
            threshold_value=float(threshold),
            current_pct_of_threshold=round(current_pct, 1),
            trend=trend,
            anomaly_score=round(anomaly_score.score, 3),
            contributing_metrics=config["contributing"],
            extra=extra,
        )

        # render specific human-readable message
        message = render_message(
            scenario=scenario,
            evidence=evidence,
            horizon_hours=horizon_hours,
            severity=severity,
        )

        pred_id = f"{metric_name}:{scenario.value}:{now.strftime('%Y%m%dT%H%M%S')}"

        prediction = Prediction(
            id=pred_id,
            metric_name=metric_name,
            scenario=scenario,
            severity=severity,
            horizon_hours=horizon_hours,
            confidence=confidence,
            message=message,
            evidence=evidence,
            created_at=now,
            expires_at=now + timedelta(hours=PREDICTION_TTL_HOURS),
        )

        logger.info(
            "Prediction: [%s] %s (%.0f%% confidence, horizon %s)",
            severity.value,
            metric_name,
            confidence * 100,
            prediction.horizon_label,
        )
        return prediction
