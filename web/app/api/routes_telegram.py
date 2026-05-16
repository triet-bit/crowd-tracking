"""Telegram test route."""
import asyncio
from fastapi import APIRouter
from web.app.adapters.notifier.telegram_notifier import create_notifier

router = APIRouter(tags=["telegram"])


@router.post("/telegram/test")
async def test_telegram():
    notifier = create_notifier()
    result = await notifier.send_message("🔔 Test message from Crowd Tracking Web!")
    return {"status": "sent", "result": result}
