"""Telegram notification Celery task."""
import asyncio
import logging
from web.app.core.celery_app import celery_app
from web.app.adapters.notifier.telegram_notifier import create_notifier

logger = logging.getLogger(__name__)


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
        logger.info(f"[telegram_tasks] Alert {alert_id} sent: {result}")
    except Exception as e:
        logger.error(f"[telegram_tasks] Failed to send alert {alert_id}: {e}")
