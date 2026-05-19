"""
Generates structured JSON log lines that correlate with metric failure scenarios.
Useful for demonstrating log-based anomaly detection alongside numeric metrics.
"""
from __future__ import annotations

import json
import random
from datetime import datetime, timedelta, timezone
from typing import Iterator


def _log(level: str, message: str, **extra) -> str:
    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "level": level,
        "message": message,
        **extra,
    }
    return json.dumps(record)


class LogGenerator:
    """
    Produces structured log lines for each failure scenario.
    Mirrors the same timeline as MetricSimulator so logs and metrics align.
    """

    def __init__(self, seed: int = 42):
        random.seed(seed)

    def connection_pool_logs(
        self,
        start: datetime | None = None,
        duration_hours: float = 12,
        step_seconds: int = 60,
        max_connections: int = 100,
    ) -> Iterator[str]:
        start = start or datetime.now(timezone.utc)
        steps = int(duration_hours * 3600 / step_seconds)
        for i in range(steps):
            t = start + timedelta(seconds=i * step_seconds)
            pct = (i / steps) * 100
            active = int(20 + (max_connections - 20) * (1 - 2 ** (-4 * i / steps)))

            if pct < 60:
                level, msg = "INFO", f"Connection pool healthy: {active}/{max_connections}"
            elif pct < 80:
                level, msg = "WARN", f"Connection pool at {active}/{max_connections} — monitor closely"
            elif pct < 95:
                level, msg = "ERROR", f"Connection pool critical: {active}/{max_connections} connections used"
            else:
                level, msg = "CRITICAL", f"Connection pool EXHAUSTED ({active}/{max_connections}). New connections rejected."

            record = {
                "timestamp": t.isoformat(),
                "level": level,
                "message": msg,
                "service": "postgres",
                "active_connections": active,
                "max_connections": max_connections,
            }
            yield json.dumps(record)

    def memory_leak_logs(
        self,
        start: datetime | None = None,
        duration_hours: float = 24,
        step_seconds: int = 300,
    ) -> Iterator[str]:
        start = start or datetime.now(timezone.utc)
        steps = int(duration_hours * 3600 / step_seconds)
        floor_mb = 512
        for i in range(steps):
            t = start + timedelta(seconds=i * step_seconds)
            used_mb = floor_mb + (i / steps) * 2048
            pct = used_mb / 8192 * 100

            if pct < 70:
                level, msg = "INFO", f"Memory usage: {used_mb:.0f}MB ({pct:.1f}%)"
            elif pct < 85:
                level, msg = "WARN", f"Memory elevated: {used_mb:.0f}MB ({pct:.1f}%) — possible leak"
            else:
                level, msg = "ERROR", f"Memory leak suspected: {used_mb:.0f}MB ({pct:.1f}%). Consider restart."

            record = {
                "timestamp": t.isoformat(),
                "level": level,
                "message": msg,
                "service": "app",
                "memory_mb": round(used_mb, 1),
                "memory_pct": round(pct, 2),
            }
            yield json.dumps(record)

    def traffic_spike_logs(
        self,
        start: datetime | None = None,
        duration_hours: float = 8,
        step_seconds: int = 30,
        baseline_rps: float = 500,
    ) -> Iterator[str]:
        start = start or datetime.now(timezone.utc)
        steps = int(duration_hours * 3600 / step_seconds)
        spike_at = steps // 3
        spike_end = spike_at + int(600 / step_seconds)

        for i in range(steps):
            t = start + timedelta(seconds=i * step_seconds)
            in_spike = spike_at <= i <= spike_end
            rps = baseline_rps * (6 + random.gauss(0, 0.3)) if in_spike else baseline_rps * (1 + random.gauss(0, 0.05))

            if in_spike:
                level, msg = "WARN", f"Traffic spike detected: {rps:.0f} rps ({rps / baseline_rps:.1f}x baseline)"
            else:
                level, msg = "INFO", f"Traffic nominal: {rps:.0f} rps"

            record = {
                "timestamp": t.isoformat(),
                "level": level,
                "message": msg,
                "service": "api",
                "request_rate": round(rps, 1),
            }
            yield json.dumps(record)
