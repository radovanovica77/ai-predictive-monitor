"""
Generates realistic synthetic time-series metrics that mimic real infrastructure failure patterns.
Each scenario produces a sequence of (timestamp, metric_name, value, tags) tuples.
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Generator, Iterator

import numpy as np


@dataclass
class MetricPoint:
    timestamp: datetime
    metric_name: str
    value: float
    tags: dict = field(default_factory=dict)


class MetricSimulator:
    """
    Produces named metric streams for five canonical failure scenarios.
    All generators are deterministic when seeded, making demos reproducible.
    """

    def __init__(self, seed: int = 42):
        self._rng = np.random.default_rng(seed)

    # ------------------------------------------------------------------
    # Public scenario generators
    # ------------------------------------------------------------------

    def connection_pool_exhaustion(
        self,
        start: datetime | None = None,
        duration_hours: float = 12,
        step_seconds: int = 60,
        max_connections: int = 100,
    ) -> Iterator[MetricPoint]:
        """
        DB connection pool saturates exponentially over duration_hours.
        Mimics a connection leak or traffic surge that is not rate-limited.
        """
        start = start or datetime.now(timezone.utc)
        steps = int(duration_hours * 3600 / step_seconds)
        baseline = 20
        tags = {"service": "postgres", "pool": "main"}

        for i in range(steps):
            t = start + timedelta(seconds=i * step_seconds)
            progress = i / steps

            # exponential saturation toward max_connections
            trend = baseline + (max_connections - baseline) * (1 - math.exp(-4 * progress))
            # add realistic jitter
            jitter = self._rng.normal(0, max_connections * 0.02)
            value = float(np.clip(trend + jitter, 0, max_connections))

            yield MetricPoint(t, "db.connections.active", value, tags)
            yield MetricPoint(t, "db.connections.pct", value / max_connections * 100, tags)

            # correlated: query duration rises as pool fills
            query_ms = 50 + (progress ** 2) * 800 + self._rng.normal(0, 10)
            yield MetricPoint(t, "db.query_duration_p95_ms", float(max(query_ms, 0)), tags)

    def memory_leak(
        self,
        start: datetime | None = None,
        duration_hours: float = 24,
        step_seconds: int = 60,
        baseline_mb: float = 512,
        leak_mb_per_hour: float = 80,
        gc_interval_hours: float = 6,
    ) -> Iterator[MetricPoint]:
        """
        Sawtooth memory pattern: gradual leak, partial GC release, increasing baseline.
        Each GC cycle fails to fully reclaim memory, so the floor creeps upward.
        """
        start = start or datetime.now(timezone.utc)
        steps = int(duration_hours * 3600 / step_seconds)
        tags = {"service": "app", "host": "worker-01"}
        gc_steps = int(gc_interval_hours * 3600 / step_seconds)

        current_floor = baseline_mb
        for i in range(steps):
            t = start + timedelta(seconds=i * step_seconds)
            position_in_cycle = i % gc_steps
            cycle_number = i // gc_steps

            leak = leak_mb_per_hour * (position_in_cycle / (3600 / step_seconds))
            # floor rises 50 MB each GC cycle (memory not fully reclaimed)
            current_floor = baseline_mb + cycle_number * 50
            jitter = self._rng.normal(0, baseline_mb * 0.01)
            value = float(current_floor + leak + jitter)

            yield MetricPoint(t, "process.memory_mb", value, tags)
            yield MetricPoint(t, "process.memory_pct", value / 8192 * 100, tags)

    def traffic_spike(
        self,
        start: datetime | None = None,
        duration_hours: float = 8,
        step_seconds: int = 30,
        baseline_rps: float = 500,
        spike_multiplier: float = 8.0,
        spikes_count: int = 3,
    ) -> Iterator[MetricPoint]:
        """
        Poisson baseline traffic with several sharp spikes.
        During spikes, latency and error rate also increase realistically.
        """
        start = start or datetime.now(timezone.utc)
        steps = int(duration_hours * 3600 / step_seconds)
        tags = {"service": "api", "region": "eu-west-1"}

        spike_centers = sorted(self._rng.integers(steps // 4, steps * 3 // 4, size=spikes_count))
        spike_width = int(600 / step_seconds)  # 10-minute spikes

        for i in range(steps):
            t = start + timedelta(seconds=i * step_seconds)

            # Poisson baseline
            base = self._rng.poisson(baseline_rps)

            # additive spike contribution (Gaussian bump around each spike center)
            spike_contribution = sum(
                spike_multiplier * baseline_rps * math.exp(-((i - c) ** 2) / (2 * spike_width ** 2))
                for c in spike_centers
            )
            rps = float(base + spike_contribution)

            yield MetricPoint(t, "http.request_rate", rps, tags)

            # correlated: latency degrades quadratically during spikes
            load_ratio = rps / baseline_rps
            latency_p99 = 50 + (load_ratio ** 2) * 200 + self._rng.normal(0, 10)
            yield MetricPoint(t, "http.latency_p99_ms", float(max(latency_p99, 5)), tags)

            # error rate increases under load
            error_rate = min(0.5 + max(load_ratio - 3, 0) * 2, 15)
            yield MetricPoint(t, "http.error_rate_pct", float(error_rate), tags)

    def latency_degradation(
        self,
        start: datetime | None = None,
        duration_hours: float = 10,
        step_seconds: int = 60,
        baseline_ms: float = 80,
        degradation_factor: float = 15.0,
    ) -> Iterator[MetricPoint]:
        """
        API latency degrades in a sigmoid pattern as an upstream dependency slows.
        Models cache exhaustion, slow queries, or a saturated downstream service.
        """
        start = start or datetime.now(timezone.utc)
        steps = int(duration_hours * 3600 / step_seconds)
        tags = {"service": "api", "endpoint": "/checkout"}

        for i in range(steps):
            t = start + timedelta(seconds=i * step_seconds)
            progress = i / steps

            # sigmoid: starts slow, accelerates at midpoint, plateaus
            sigmoid = 1 / (1 + math.exp(-10 * (progress - 0.5)))
            trend = baseline_ms + (degradation_factor - 1) * baseline_ms * sigmoid
            jitter = self._rng.normal(0, baseline_ms * 0.08)
            value = float(max(trend + jitter, baseline_ms * 0.5))

            yield MetricPoint(t, "http.latency_p50_ms", value * 0.6, tags)
            yield MetricPoint(t, "http.latency_p95_ms", value, tags)
            yield MetricPoint(t, "http.latency_p99_ms", value * 1.4, tags)

    def disk_fill(
        self,
        start: datetime | None = None,
        duration_hours: float = 48,
        step_seconds: int = 300,
        initial_pct: float = 40.0,
        fill_rate_pct_per_hour: float = 1.2,
    ) -> Iterator[MetricPoint]:
        """
        Disk usage grows linearly with small random fluctuations.
        Models log accumulation or unbounded write workloads.
        """
        start = start or datetime.now(timezone.utc)
        steps = int(duration_hours * 3600 / step_seconds)
        tags = {"host": "worker-01", "mount": "/var/data"}

        current = initial_pct
        for i in range(steps):
            t = start + timedelta(seconds=i * step_seconds)
            increment = fill_rate_pct_per_hour * (step_seconds / 3600)
            jitter = self._rng.normal(0, 0.1)
            current = float(np.clip(current + increment + jitter, 0, 100))

            yield MetricPoint(t, "disk.usage_pct", current, tags)
            yield MetricPoint(t, "disk.free_gb", (100 - current) / 100 * 500, tags)

    # ------------------------------------------------------------------
    # Helper: generate all scenarios mixed together
    # ------------------------------------------------------------------

    def generate_all(
        self,
        start: datetime | None = None,
        speed_multiplier: int = 1,
    ) -> list[MetricPoint]:
        """
        Generate all five scenarios starting from `start`, returning a flat
        sorted list of MetricPoints. speed_multiplier compresses simulated time
        (e.g. 60 means 1 real second = 1 simulated minute).
        """
        start = start or datetime.now(timezone.utc)
        points: list[MetricPoint] = []

        step = max(1, 60 // speed_multiplier)

        for point in self.connection_pool_exhaustion(start=start, step_seconds=step):
            points.append(point)
        for point in self.memory_leak(start=start, step_seconds=step):
            points.append(point)
        for point in self.traffic_spike(start=start, step_seconds=step):
            points.append(point)
        for point in self.latency_degradation(start=start, step_seconds=step):
            points.append(point)
        for point in self.disk_fill(start=start, step_seconds=step):
            points.append(point)

        points.sort(key=lambda p: p.timestamp)
        return points
