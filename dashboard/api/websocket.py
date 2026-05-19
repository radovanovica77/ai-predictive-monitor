"""
WebSocket connection manager.
Maintains a set of active connections and broadcasts JSON messages to all.
"""
from __future__ import annotations

import json
import logging
from typing import Any

from fastapi import WebSocket

logger = logging.getLogger(__name__)


class ConnectionManager:
    def __init__(self):
        self._active: list[WebSocket] = []

    async def connect(self, ws: WebSocket) -> None:
        await ws.accept()
        self._active.append(ws)
        logger.info("WebSocket client connected. Total: %d", len(self._active))

    def disconnect(self, ws: WebSocket) -> None:
        self._active = [c for c in self._active if c is not ws]
        logger.info("WebSocket client disconnected. Total: %d", len(self._active))

    async def broadcast(self, payload: dict[str, Any]) -> None:
        if not self._active:
            return
        message = json.dumps(payload, default=str)
        dead: list[WebSocket] = []
        for ws in self._active:
            try:
                await ws.send_text(message)
            except Exception:
                dead.append(ws)
        for ws in dead:
            self.disconnect(ws)

    @property
    def client_count(self) -> int:
        return len(self._active)


manager = ConnectionManager()
