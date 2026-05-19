"""
Simulates database connection pool metrics.
In a real deployment, replace _get_pool_stats() with actual DB driver calls.
"""
from __future__ import annotations

import random
from datetime import datetime, timezone

from data_pipeline.simulator.metric_simulator import MetricPoint

_MAX_CONNECTIONS = 100
_current_connections = 20


def collect_db_metrics() -> list[MetricPoint]:
    """Returns current DB connection pool stats as MetricPoints."""
    global _current_connections
    now = datetime.now(timezone.utc)
    tags = {"service": "postgres", "pool": "main"}

    # small random walk to simulate live fluctuation
    _current_connections = max(1, min(_MAX_CONNECTIONS, _current_connections + random.gauss(0, 1)))

    return [
        MetricPoint(now, "db.connections.active", _current_connections, tags),
        MetricPoint(now, "db.connections.pct", _current_connections / _MAX_CONNECTIONS * 100, tags),
        MetricPoint(now, "db.query_duration_p95_ms", 50 + (_current_connections / _MAX_CONNECTIONS) * 200, tags),
    ]
