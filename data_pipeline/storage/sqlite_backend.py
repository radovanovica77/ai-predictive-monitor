"""
SQLite time-series backend. Zero-setup, works everywhere.
Schema: metrics(id, timestamp, metric_name, value, tags_json)
Indexed on (metric_name, timestamp) for fast range queries.
"""
from __future__ import annotations

import asyncio
import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from data_pipeline.simulator.metric_simulator import MetricPoint
from data_pipeline.storage.time_series_store import TimeSeriesStore

_CREATE_TABLE = """
CREATE TABLE IF NOT EXISTS metrics (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp   TEXT    NOT NULL,
    metric_name TEXT    NOT NULL,
    value       REAL    NOT NULL,
    tags_json   TEXT    NOT NULL DEFAULT '{}'
);
"""

_CREATE_INDEX = """
CREATE INDEX IF NOT EXISTS idx_metric_time
    ON metrics (metric_name, timestamp);
"""


class SQLiteStore(TimeSeriesStore):
    """Thread-safe SQLite backend using asyncio executor for blocking I/O."""

    def __init__(self, db_path: str = "data/metrics.db"):
        self._db_path = Path(db_path)
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn: sqlite3.Connection | None = None
        self._lock = asyncio.Lock()

    async def _get_conn(self) -> sqlite3.Connection:
        if self._conn is None:
            self._conn = sqlite3.connect(str(self._db_path), check_same_thread=False)
            self._conn.row_factory = sqlite3.Row
            self._conn.execute(_CREATE_TABLE)
            self._conn.execute(_CREATE_INDEX)
            self._conn.commit()
        return self._conn

    async def write(self, points: list[MetricPoint]) -> None:
        if not points:
            return
        conn = await self._get_conn()
        rows = [
            (
                p.timestamp.isoformat(),
                p.metric_name,
                p.value,
                json.dumps(p.tags),
            )
            for p in points
        ]
        async with self._lock:
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(
                None,
                lambda: conn.executemany(
                    "INSERT INTO metrics (timestamp, metric_name, value, tags_json) VALUES (?,?,?,?)",
                    rows,
                ) or conn.commit(),
            )

    async def query_range(
        self,
        metric_name: str,
        start: datetime,
        end: datetime,
        tags: dict[str, str] | None = None,
    ) -> pd.DataFrame:
        conn = await self._get_conn()
        loop = asyncio.get_event_loop()

        def _query() -> list[dict]:
            cursor = conn.execute(
                """
                SELECT timestamp, value, tags_json
                FROM metrics
                WHERE metric_name = ?
                  AND timestamp >= ?
                  AND timestamp <= ?
                ORDER BY timestamp ASC
                """,
                (metric_name, start.isoformat(), end.isoformat()),
            )
            return [dict(row) for row in cursor.fetchall()]

        rows = await loop.run_in_executor(None, _query)
        if not rows:
            return pd.DataFrame(columns=["timestamp", "value", "tags"])

        df = pd.DataFrame(rows)
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
        df["tags"] = df["tags_json"].apply(json.loads)
        df.drop(columns=["tags_json"], inplace=True)
        return df

    async def get_latest(self, metric_name: str) -> MetricPoint | None:
        conn = await self._get_conn()
        loop = asyncio.get_event_loop()

        def _query() -> dict | None:
            cursor = conn.execute(
                """
                SELECT timestamp, metric_name, value, tags_json
                FROM metrics
                WHERE metric_name = ?
                ORDER BY timestamp DESC
                LIMIT 1
                """,
                (metric_name,),
            )
            row = cursor.fetchone()
            return dict(row) if row else None

        row = await loop.run_in_executor(None, _query)
        if row is None:
            return None
        return MetricPoint(
            timestamp=datetime.fromisoformat(row["timestamp"]),
            metric_name=row["metric_name"],
            value=row["value"],
            tags=json.loads(row["tags_json"]),
        )

    async def list_metrics(self) -> list[str]:
        conn = await self._get_conn()
        loop = asyncio.get_event_loop()

        def _query() -> list[str]:
            cursor = conn.execute("SELECT DISTINCT metric_name FROM metrics ORDER BY metric_name")
            return [r[0] for r in cursor.fetchall()]

        return await loop.run_in_executor(None, _query)

    async def close(self) -> None:
        if self._conn:
            self._conn.close()
            self._conn = None
