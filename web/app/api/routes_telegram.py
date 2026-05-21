"""Telegram test route."""
from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from web.app.adapters.notifier.telegram_notifier import create_notifier
from web.app.core.database import get_db
from web.app.models.telegram_log import TelegramLog

router = APIRouter(tags=["telegram"])


@router.post("/telegram/test")
async def test_telegram():
    notifier = create_notifier()
    result = await notifier.send_message("🔔 Test message from Crowd Tracking Web!")
    return {"status": "sent", "result": result}


@router.get("/telegram/logs")
def list_telegram_logs(alert_id: str = None, camera_id: str = None, limit: int = 50, db: Session = Depends(get_db)):
    query = db.query(TelegramLog)
    if alert_id:
        query = query.filter(TelegramLog.alert_id == alert_id)
    if camera_id:
        query = query.filter(TelegramLog.camera_id == camera_id)
    logs = query.order_by(TelegramLog.created_at.desc()).limit(limit).all()
    return [
        {
            "id": log.id,
            "alert_id": log.alert_id,
            "camera_id": log.camera_id,
            "send_type": log.send_type,
            "status": log.status,
            "response_code": log.response_code,
            "error": log.error,
            "message_preview": log.message_preview,
            "created_at": str(log.created_at) if log.created_at else None,
        }
        for log in logs
    ]
