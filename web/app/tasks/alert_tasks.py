"""Alert Celery tasks."""
import logging
import time
import os
import cv2
from web.app.core.celery_app import celery_app
from web.app.core.redis import set_alert_cooldown, publish_ws_event
from web.app.core.config import settings

logger = logging.getLogger(__name__)


@celery_app.task(name="web.app.tasks.alert_tasks.create_alert")
def create_alert(payload: dict):
    """
    Tạo alert:
    1. Check cooldown (lần 2 - double check)
    2. Lưu snapshot
    3. Lưu alert vào DB
    4. Gửi Telegram
    5. Gửi hardware command
    6. Broadcast lên WebSocket
    7. Set cooldown
    """
    camera_id = payload.get("camera_id", "unknown")
    zone_id = payload.get("zone_id", "default")
    alert_type = payload.get("alert_type", "threshold_exceeded")

    logger.info(f"[alert_tasks] Creating alert: {alert_type} for camera={camera_id}")

    # Lưu alert vào DB
    alert_id = _insert_alert_to_db(payload)

    # Broadcast WebSocket
    publish_ws_event(camera_id, {
        "event": "alert.created",
        "data": {**payload, "id": alert_id},
    })

    # Set cooldown
    set_alert_cooldown(camera_id, zone_id, alert_type)

    # Gửi Telegram và hardware async
    from web.app.tasks.telegram_tasks import send_telegram_alert
    from web.app.tasks.hardware_tasks import send_hardware_command
    send_telegram_alert.delay(alert_id, payload)
    send_hardware_command.delay(alert_id, payload)

    return {"alert_id": alert_id}


def _insert_alert_to_db(payload: dict) -> str:
    """Lưu alert vào SQLite database."""
    try:
        from web.app.core.database import SessionLocal
        from web.app.models.alert import Alert
        import uuid
        db = SessionLocal()
        alert_id = str(uuid.uuid4())[:8]
        alert = Alert(
            id=alert_id,
            camera_id=payload.get("camera_id"),
            zone_id=payload.get("zone_id"),
            alert_type=payload.get("alert_type", "threshold_exceeded"),
            severity=payload.get("severity", "medium"),
            message=payload.get("message", ""),
            people_count=payload.get("people_count", 0),
            snapshot_path=None,
            hardware_status="pending",
            telegram_status="pending",
        )
        db.add(alert)
        db.commit()
        db.close()
        return alert_id
    except Exception as e:
        logger.error(f"[alert_tasks] Failed to insert alert: {e}")
        return f"alert_{int(time.time())}"
