"""
Async data pipeline: metrics flow from collectors/simulator → queue → storage.
Uses asyncio.Queue for backpressure; batch-writes every BATCH_INTERVAL seconds.
"""
from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone
from typing import AsyncIterator

from data_pipeline.simulator.metric_simulator import MetricPoint, MetricSimulator
from data_pipeline.storage.time_series_store import TimeSeriesStore

logger = logging.getLogger(__name__)

BATCH_INTERVAL = 5  # seconds between batch writes to storage
QUEUE_MAX = 10_000


class DataPipeline:
    """
    Connects a metric source (simulator or live collectors) to a storage backend.

    Usage:
        pipeline = DataPipeline(store=SQLiteStore(), simulator=MetricSimulator())
        await pipeline.run_simulation(duration_hours=12, speed_multiplier=60)
    """

    def __init__(self, store: TimeSeriesStore):
        self._store = store
        self._queue: asyncio.Queue[MetricPoint] = asyncio.Queue(maxsize=QUEUE_MAX)
        self._running = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def run_simulation(
        self,
        duration_hours: float = 12,
        speed_multiplier: int = 60,
        seed: int = 42,
    ) -> None:
        """
        Run all five failure scenarios through the pipeline.
        speed_multiplier=60 means 1 simulated minute per real second.
        """
        simulator = MetricSimulator(seed=seed)
        start = datetime.now(timezone.utc)

        logger.info(
            "Starting simulation: duration=%sh speed=%sx",
            duration_hours,
            speed_multiplier,
        )

        self._running = True
        writer_task = asyncio.create_task(self._batch_writer())

        step_seconds = max(1, 60 // speed_multiplier)
        real_sleep = step_seconds / speed_multiplier

        scenarios = [
            simulator.connection_pool_exhaustion(start=start, step_seconds=step_seconds, duration_hours=duration_hours),
            simulator.memory_leak(start=start, step_seconds=step_seconds, duration_hours=duration_hours),
            simulator.traffic_spike(start=start, step_seconds=step_seconds, duration_hours=duration_hours),
            simulator.latency_degradation(start=start, step_seconds=step_seconds, duration_hours=duration_hours),
            simulator.disk_fill(start=start, step_seconds=step_seconds, duration_hours=duration_hours),
        ]

        try:
            async for points in self._interleave(scenarios, sleep_per_step=real_sleep):
                for p in points:
                    await self._queue.put(p)
        finally:
            self._running = False
            await writer_task

        logger.info("Simulation complete. Points queued and flushed.")

    async def ingest_points(self, points: list[MetricPoint]) -> None:
        """Push a batch of pre-generated points directly into the queue."""
        for p in points:
            await self._queue.put(p)

    async def flush(self) -> None:
        """Drain the queue and write everything to storage immediately."""
        batch: list[MetricPoint] = []
        while not self._queue.empty():
            batch.append(self._queue.get_nowait())
        if batch:
            await self._store.write(batch)
            logger.debug("Flushed %d points to storage", len(batch))

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    async def _batch_writer(self) -> None:
        """Consumer: drains the queue in batches every BATCH_INTERVAL seconds."""
        while self._running or not self._queue.empty():
            await asyncio.sleep(BATCH_INTERVAL)
            batch: list[MetricPoint] = []
            try:
                while True:
                    batch.append(self._queue.get_nowait())
            except asyncio.QueueEmpty:
                pass

            if batch:
                await self._store.write(batch)
                logger.debug("Wrote %d points to storage", len(batch))

    @staticmethod
    async def _interleave(
        generators,
        sleep_per_step: float = 0.0,
    ) -> AsyncIterator[list[MetricPoint]]:
        """
        Advances all scenario generators one step at a time, yielding the
        points from all active generators at each timestep together.
        """
        iters = [iter(g) for g in generators]
        while iters:
            batch: list[MetricPoint] = []
            alive = []
            for it in iters:
                try:
                    # consume all points from this generator for the current step
                    p = next(it)
                    batch.append(p)
                    alive.append(it)
                except StopIteration:
                    pass
            iters = alive
            if batch:
                yield batch
            if sleep_per_step > 0:
                await asyncio.sleep(sleep_per_step)
