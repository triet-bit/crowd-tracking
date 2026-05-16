"""Hardware command Celery task."""
import asyncio
import logging
from web.app.core.celery_app import celery_app
from web.app.adapters.hardware.hardware_client import create_hardware_client

logger = logging.getLogger(__name__)


@celery_app.task(name="web.app.tasks.hardware_tasks.send_hardware_command")
def send_hardware_command(alert_id: str, payload: dict):
    """Gửi lệnh xuống phần cứng khi có alert."""
    hardware = create_hardware_client()
    command = {
        "target": "alarm_device_001",
        "command": "TURN_ON_ALARM",
        "duration_seconds": 10,
        "payload": {
            "alert_id": alert_id,
            "camera_id": payload.get("camera_id"),
            "people_count": payload.get("people_count"),
            "severity": payload.get("severity"),
        }
    }

    async def _send():
        return await hardware.send_command(command)

    try:
        result = asyncio.run(_send())
        logger.info(f"[hardware_tasks] Alert {alert_id} command sent: {result}")
    except Exception as e:
        logger.error(f"[hardware_tasks] Failed to send hardware command: {e}")
