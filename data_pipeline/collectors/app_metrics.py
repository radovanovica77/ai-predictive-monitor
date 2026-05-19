"""
Simulates application-level HTTP metrics (request rate, latency, error rate).
In a real deployment, replace with Prometheus scrape or StatsD listener.
"""
from __future__ import annotations

import random
from datetime import datetime, timezone

import numpy as np

from data_pipeline.simulator.metric_simulator import MetricPoint

_rng = np.random.default_rng()
_baseline_rps = 500.0


def collect_app_metrics() -> list[MetricPoint]:
    now = datetime.now(timezone.utc)
    tags = {"service": "api", "region": "eu-west-1"}

    rps = float(_rng.poisson(_baseline_rps))
    load_ratio = rps / _baseline_rps
    latency_p99 = 50 + (load_ratio ** 2) * 100 + float(_rng.normal(0, 8))
    error_rate = max(0.0, 0.5 + max(load_ratio - 2, 0) * 1.5)

    return [
        MetricPoint(now, "http.request_rate", rps, tags),
        MetricPoint(now, "http.latency_p99_ms", max(latency_p99, 5), tags),
        MetricPoint(now, "http.error_rate_pct", min(error_rate, 20), tags),
    ]
