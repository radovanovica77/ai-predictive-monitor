"""
Collects real CPU, memory, and disk metrics from the host via psutil.
Used in live mode; in simulation mode the MetricSimulator replaces this.
"""
from __future__ import annotations

from datetime import datetime, timezone

import psutil

from data_pipeline.simulator.metric_simulator import MetricPoint


def collect_system_metrics() -> list[MetricPoint]:
    now = datetime.now(timezone.utc)
    tags = {"source": "system"}

    cpu = psutil.cpu_percent(interval=None)
    mem = psutil.virtual_memory()
    disk = psutil.disk_usage("/")

    return [
        MetricPoint(now, "system.cpu_pct", cpu, tags),
        MetricPoint(now, "system.memory_pct", mem.percent, tags),
        MetricPoint(now, "system.memory_used_mb", mem.used / 1024 / 1024, tags),
        MetricPoint(now, "system.disk_pct", disk.percent, tags),
        MetricPoint(now, "system.disk_free_gb", disk.free / 1024 / 1024 / 1024, tags),
    ]
