"""Telegram notification Celery task."""
import asyncio
import logging
import uuid

from web.app.core.celery_app import celery_app
from web.app.adapters.notifier.telegram_notifier import create_notifier

logger = logging.getLogger(__name__)


def _save_telegram_log(alert_id: str, payload: dict, result: dict, send_type: str) -> None:
    try:
        from web.app.core.database import SessionLocal
        from web.app.models.telegram_log import TelegramLog

        db = SessionLocal()
        log = TelegramLog(
            id=str(uuid.uuid4())[:8],
            alert_id=alert_id,
            camera_id=payload.get("camera_id"),
            send_type=send_type,
            status=result.get("status", "failed"),
            response_code=result.get("response_code"),
            error=result.get("error"),
            message_preview=(payload.get("message", "") or "")[:255],
        )
        db.add(log)
        db.commit()
        db.close()
    except Exception as e:
        logger.error(f"[telegram_tasks] Failed to save telegram log for alert {alert_id}: {e}")


def _update_alert_telegram_status(alert_id: str, status: str) -> None:
    try:
        from web.app.core.database import SessionLocal
        from web.app.models.alert import Alert

        db = SessionLocal()
        alert = db.query(Alert).filter(Alert.id == alert_id).first()
        if alert:
            alert.telegram_status = status
            db.commit()
        db.close()
    except Exception as e:
        logger.error(f"[telegram_tasks] Failed to update alert telegram_status for {alert_id}: {e}")


@celery_app.task(name="web.app.tasks.telegram_tasks.send_telegram_alert")
def send_telegram_alert(alert_id: str, payload: dict):
    """Gửi thông báo Telegram khi có alert."""
    notifier = create_notifier()
    camera_id = payload.get("camera_id", "?")
    alert_type = payload.get("alert_type", "alert")
    count = payload.get("people_count", 0)
    severity = payload.get("severity", "medium")
    message = payload.get("message", "")

    text = (
        f"🚨 <b>CROWD ALERT</b>\n"
        f"Camera: <code>{camera_id}</code>\n"
        f"Type: {alert_type}\n"
        f"Severity: {severity.upper()}\n"
        f"People count: {count}\n"
        f"Message: {message}"
    )

    snapshot_path = payload.get("snapshot_path")

    async def _send():
        if snapshot_path:
            return await notifier.send_photo(snapshot_path, caption=text)
        return await notifier.send_message(text)

    try:
        result = asyncio.run(_send())
        send_type = "photo" if snapshot_path else "message"
        status = result.get("status", "failed")
        _save_telegram_log(alert_id, payload, result, send_type)
        _update_alert_telegram_status(alert_id, status)
        logger.info(f"[telegram_tasks] Alert {alert_id} sent: {result}")
    except Exception as e:
        _save_telegram_log(
            alert_id,
            payload,
            {"status": "failed", "error": str(e), "response_code": None},
            "photo" if snapshot_path else "message",
        )
        _update_alert_telegram_status(alert_id, "failed")
        logger.error(f"[telegram_tasks] Failed to send alert {alert_id}: {e}")
