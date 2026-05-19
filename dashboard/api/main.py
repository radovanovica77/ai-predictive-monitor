"""
FastAPI application entry point.
Serves the dashboard frontend, REST API, and WebSocket endpoint.
"""
from __future__ import annotations

import asyncio
import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from dashboard.api.demo_engine import DemoEngine
from dashboard.api.routes import router, set_engine
from dashboard.api.websocket import manager

logger = logging.getLogger(__name__)

_FRONTEND_DIR = Path(__file__).parent.parent / "frontend"

engine = DemoEngine(db_path="data/metrics.db")


@asynccontextmanager
async def lifespan(app: FastAPI):
    engine.set_broadcast_callback(manager.broadcast)
    task = asyncio.create_task(engine.start())
    set_engine(engine)
    yield
    task.cancel()
    await engine.store.close()


app = FastAPI(
    title="AI Predictive Monitor",
    description="Real-time infrastructure failure prediction dashboard",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# REST endpoints under /api
app.include_router(router, prefix="/api")

# Serve frontend static files
app.mount("/static", StaticFiles(directory=str(_FRONTEND_DIR)), name="static")


@app.get("/")
async def serve_dashboard():
    return FileResponse(str(_FRONTEND_DIR / "index.html"))


@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await manager.connect(ws)
    try:
        # send current state immediately on connect
        if engine.is_ready and engine.current_predictions:
            import json
            from datetime import timezone
            await ws.send_text(json.dumps({
                "type": "predictions_update",
                "predictions": engine.current_predictions,
                "metrics_summary": engine.metrics_summary,
                "timestamp": __import__("datetime").datetime.now(timezone.utc).isoformat(),
                "active_count": len(engine.current_predictions),
            }, default=str))
        elif not engine.is_ready:
            await ws.send_text('{"type":"status","message":"Initializing simulation..."}')

        # keep connection alive; engine broadcasts updates
        while True:
            await ws.receive_text()
    except WebSocketDisconnect:
        manager.disconnect(ws)
    except Exception as e:
        logger.error("WebSocket error: %s", e)
        manager.disconnect(ws)
