"""
REST API endpoints for the dashboard.
All responses are JSON; used by the frontend and optional external tools.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import JSONResponse

router = APIRouter()

# injected by main.py after engine initialises
_engine = None


def set_engine(engine) -> None:
    global _engine
    _engine = engine


@router.get("/health")
async def health() -> dict:
    return {
        "status": "ok",
        "engine_ready": _engine.is_ready if _engine else False,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


@router.get("/predictions")
async def get_predictions() -> dict:
    """Return active predictions sorted by severity (CRITICAL first)."""
    if not _engine or not _engine.is_ready:
        return {"predictions": [], "status": "initializing"}

    return {
        "predictions": _engine.current_predictions,
        "count": len(_engine.current_predictions),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


@router.get("/metrics/summary")
async def get_metrics_summary() -> dict:
    """Return latest value for each tracked metric."""
    if not _engine or not _engine.is_ready:
        return {"metrics": {}, "status": "initializing"}

    return {
        "metrics": _engine.metrics_summary,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


@router.get("/metrics/{metric_name}/history")
async def get_metric_history(
    metric_name: str,
    hours: float = Query(default=2.0, ge=0.1, le=48.0),
) -> dict:
    """Return time-series history for a single metric."""
    if not _engine or not _engine.is_ready:
        raise HTTPException(status_code=503, detail="Engine not ready")

    now = datetime.now(timezone.utc)
    start = now - timedelta(hours=hours)

    df = await _engine.store.query_range(metric_name, start=start, end=now)
    if df.empty:
        raise HTTPException(status_code=404, detail=f"No data for metric: {metric_name}")

    records = [
        {"timestamp": row["timestamp"].isoformat(), "value": round(row["value"], 3)}
        for _, row in df.iterrows()
    ]
    return {
        "metric_name": metric_name,
        "points": records,
        "count": len(records),
        "from": start.isoformat(),
        "to": now.isoformat(),
    }


@router.get("/metrics")
async def list_metrics() -> dict:
    """List all tracked metric names."""
    if not _engine or not _engine.is_ready:
        return {"metrics": []}
    metrics = await _engine.store.list_metrics()
    return {"metrics": metrics}
