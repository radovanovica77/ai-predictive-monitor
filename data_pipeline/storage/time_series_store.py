"""
Abstract interface for time-series storage backends.
Swap implementations via config without changing callers.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any

import pandas as pd

from data_pipeline.simulator.metric_simulator import MetricPoint


class TimeSeriesStore(ABC):
    """Base class all storage backends must implement."""

    @abstractmethod
    async def write(self, points: list[MetricPoint]) -> None:
        """Persist a batch of metric points."""
        ...

    @abstractmethod
    async def query_range(
        self,
        metric_name: str,
        start: datetime,
        end: datetime,
        tags: dict[str, str] | None = None,
    ) -> pd.DataFrame:
        """
        Return a DataFrame with columns [timestamp, value, tags]
        for the given metric_name between start and end (inclusive).
        """
        ...

    @abstractmethod
    async def get_latest(self, metric_name: str) -> MetricPoint | None:
        """Return the most recent point for a metric, or None."""
        ...

    @abstractmethod
    async def list_metrics(self) -> list[str]:
        """Return distinct metric names present in storage."""
        ...

    @abstractmethod
    async def close(self) -> None:
        """Release any held resources (connections, file handles)."""
        ...
