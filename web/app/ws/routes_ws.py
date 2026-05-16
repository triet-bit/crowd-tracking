"""WebSocket connection manager and routes."""
import json
import asyncio
import logging
from typing import Dict, List
from fastapi import APIRouter, WebSocket, WebSocketDisconnect
import redis.asyncio as aioredis
from web.app.core.config import settings

logger = logging.getLogger(__name__)
router = APIRouter(tags=["websocket"])


class ConnectionManager:
    def __init__(self):
        self.active_connections: Dict[str, List[WebSocket]] = {}

    async def connect(self, websocket: WebSocket, camera_id: str):
        await websocket.accept()
        if camera_id not in self.active_connections:
            self.active_connections[camera_id] = []
        self.active_connections[camera_id].append(websocket)
        logger.info(f"[WS] Client connected: camera={camera_id}")

    def disconnect(self, websocket: WebSocket, camera_id: str):
        if camera_id in self.active_connections:
            self.active_connections[camera_id].discard(websocket) \
                if hasattr(self.active_connections[camera_id], 'discard') \
                else self.active_connections[camera_id].remove(websocket)
        logger.info(f"[WS] Client disconnected: camera={camera_id}")

    async def broadcast(self, camera_id: str, message: dict):
        conns = self.active_connections.get(camera_id, [])
        dead = []
        for ws in conns:
            try:
                await ws.send_json(message)
            except Exception:
                dead.append(ws)
        for ws in dead:
            conns.remove(ws)


manager = ConnectionManager()


@router.websocket("/ws/cameras/{camera_id}")
async def ws_camera(websocket: WebSocket, camera_id: str):
    await manager.connect(websocket, camera_id)

    # Subscribe Redis pub/sub channel
    redis = aioredis.from_url(settings.REDIS_URL, decode_responses=True)
    pubsub = redis.pubsub()
    await pubsub.subscribe(f"ws:broadcast:{camera_id}")

    async def listen_redis():
        async for msg in pubsub.listen():
            if msg["type"] == "message":
                try:
                    data = json.loads(msg["data"])
                    await manager.broadcast(camera_id, data)
                except Exception as e:
                    logger.error(f"[WS] Redis message error: {e}")

    task = asyncio.create_task(listen_redis())

    try:
        while True:
            # Giữ kết nối, lắng nghe ping từ client
            data = await websocket.receive_text()
            if data == "ping":
                await websocket.send_text("pong")
    except WebSocketDisconnect:
        manager.disconnect(websocket, camera_id)
        task.cancel()
        await pubsub.unsubscribe(f"ws:broadcast:{camera_id}")
        await redis.close()
