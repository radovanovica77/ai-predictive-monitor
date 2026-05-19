"""
Evaluates anomaly detection quality against labeled synthetic scenarios.
Outputs precision, recall, and F1 score — useful for README and portfolio demonstration.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd

from data_pipeline.simulator.metric_simulator import MetricSimulator
from ml_engine.models.isolation_forest import IsolationForestModel
from ml_engine.models.base_model import AnomalyScore

logger = logging.getLogger(__name__)

ANOMALY_SCORE_THRESHOLD = 0.65


@dataclass
class EvalResult:
    metric_name: str
    model_name: str
    precision: float
    recall: float
    f1: float
    n_points: int
    n_anomalies_true: int
    n_anomalies_detected: int

    def __str__(self) -> str:
        return (
            f"{self.metric_name} | {self.model_name}: "
            f"P={self.precision:.2f} R={self.recall:.2f} F1={self.f1:.2f} "
            f"({self.n_anomalies_detected}/{self.n_anomalies_true} anomalies detected)"
        )


class ModelEvaluator:
    """
    Generates labeled synthetic data (normal + anomalous), trains a model,
    scores it, then computes standard classification metrics.
    """

    def __init__(self, threshold: float = ANOMALY_SCORE_THRESHOLD):
        self._threshold = threshold

    def evaluate_isolation_forest(self) -> list[EvalResult]:
        """
        Evaluate IsolationForest on all five failure scenarios.
        Returns one EvalResult per scenario.
        """
        sim = MetricSimulator(seed=123)
        results = []

        scenarios = [
            ("db.connections.active", list(sim.connection_pool_exhaustion(duration_hours=2, step_seconds=60))),
            ("process.memory_mb", list(sim.memory_leak(duration_hours=4, step_seconds=60))),
            ("http.request_rate", list(sim.traffic_spike(duration_hours=2, step_seconds=30))),
            ("http.latency_p95_ms", list(sim.latency_degradation(duration_hours=2, step_seconds=60))),
            ("disk.usage_pct", list(sim.disk_fill(duration_hours=12, step_seconds=300))),
        ]

        for metric_name, points in scenarios:
            metric_points = [p for p in points if p.metric_name == metric_name]
            if len(metric_points) < 50:
                continue

            df = pd.DataFrame({
                "timestamp": [p.timestamp for p in metric_points],
                "value": [p.value for p in metric_points],
            })

            # label: last 20% of series is "anomalous" (approaching failure threshold)
            n = len(df)
            labels = np.zeros(n, dtype=int)
            anomaly_start = int(n * 0.75)
            labels[anomaly_start:] = 1

            # train on first 60% (normal period only)
            train_end = int(n * 0.6)
            train_df = df.iloc[:train_end]

            model = IsolationForestModel(contamination=0.05)
            model.fit(train_df)

            if not model.is_fitted:
                continue

            scores: list[AnomalyScore] = model.score(df)
            if not scores:
                continue

            predictions = np.array([1 if s.score >= self._threshold else 0 for s in scores])

            tp = int(np.sum((predictions == 1) & (labels == 1)))
            fp = int(np.sum((predictions == 1) & (labels == 0)))
            fn = int(np.sum((predictions == 0) & (labels == 1)))

            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

            result = EvalResult(
                metric_name=metric_name,
                model_name="isolation_forest",
                precision=round(precision, 3),
                recall=round(recall, 3),
                f1=round(f1, 3),
                n_points=n,
                n_anomalies_true=int(labels.sum()),
                n_anomalies_detected=int(predictions.sum()),
            )
            results.append(result)
            logger.info(str(result))

        return results

    def print_report(self, results: list[EvalResult]) -> None:
        print("\n=== Model Evaluation Report ===")
        print(f"{'Metric':<35} {'Precision':>9} {'Recall':>7} {'F1':>6} {'Detected/True':>14}")
        print("-" * 75)
        for r in results:
            print(
                f"{r.metric_name:<35} {r.precision:>9.2f} {r.recall:>7.2f} {r.f1:>6.2f} "
                f"{r.n_anomalies_detected:>6}/{r.n_anomalies_true:<6}"
            )
        if results:
            avg_f1 = np.mean([r.f1 for r in results])
            print(f"\nAverage F1: {avg_f1:.2f}")
