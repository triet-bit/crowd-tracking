import redis
import json
from .config import settings

redis_client = redis.from_url(settings.REDIS_URL, decode_responses=True)
# Binary client for storing raw JPEG frames (no decode)
redis_binary = redis.from_url(settings.REDIS_URL, decode_responses=False)


def get_redis():
    return redis_client


def save_camera_metrics(camera_id: str, metrics: dict):
    key = f"camera:{camera_id}:latest_metrics"
    redis_client.setex(key, 300, json.dumps(metrics))  # TTL 5 phút


def get_camera_metrics(camera_id: str) -> dict | None:
    key = f"camera:{camera_id}:latest_metrics"
    data = redis_client.get(key)
    return json.loads(data) if data else None


def set_camera_status(camera_id: str, status: dict):
    key = f"camera:{camera_id}:status"
    redis_client.set(key, json.dumps(status))


def get_camera_status(camera_id: str) -> dict | None:
    key = f"camera:{camera_id}:status"
    data = redis_client.get(key)
    return json.loads(data) if data else None


def check_alert_cooldown(camera_id: str, zone_id: str, alert_type: str) -> bool:
    key = f"alert:cooldown:{camera_id}:{zone_id}:{alert_type}"
    return bool(redis_client.exists(key))


def set_alert_cooldown(camera_id: str, zone_id: str, alert_type: str, ttl: int = None):
    from .config import settings as s
    key = f"alert:cooldown:{camera_id}:{zone_id}:{alert_type}"
    redis_client.setex(key, ttl or s.ALERT_COOLDOWN_SECONDS, "1")


def publish_ws_event(camera_id: str, event: dict):
    channel = f"ws:broadcast:{camera_id}"
    redis_client.publish(channel, json.dumps(event))


def save_camera_frame(camera_id: str, jpeg_bytes: bytes):
    """Lưu frame JPEG đã xử lý vào Redis (binary)."""
    key = f"camera:{camera_id}:frame"
    redis_binary.setex(key, 10, jpeg_bytes)  # TTL 10s


def get_camera_frame(camera_id: str) -> bytes | None:
    """Lấy frame JPEG mới nhất từ Redis."""
    key = f"camera:{camera_id}:frame"
    return redis_binary.get(key)
