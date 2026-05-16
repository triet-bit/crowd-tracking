"""Telegram notifier adapter."""
import httpx
import logging

logger = logging.getLogger(__name__)


class BaseNotifier:
    async def send_message(self, text: str) -> dict:
        raise NotImplementedError

    async def send_photo(self, photo_path: str, caption: str) -> dict:
        raise NotImplementedError


class TelegramNotifier(BaseNotifier):
    """Gửi thông báo qua Telegram Bot API."""

    def __init__(self, bot_token: str, chat_id: str):
        self.bot_token = bot_token
        self.chat_id = chat_id
        self.base_url = f"https://api.telegram.org/bot{bot_token}"

    async def send_message(self, text: str) -> dict:
        if not self.bot_token or not self.chat_id:
            logger.warning("[Telegram] Bot token or chat ID not configured. Skipping.")
            return {"status": "skipped"}
        url = f"{self.base_url}/sendMessage"
        try:
            async with httpx.AsyncClient(timeout=10) as client:
                resp = await client.post(url, json={"chat_id": self.chat_id, "text": text, "parse_mode": "HTML"})
                return {"status": "sent", "response_code": resp.status_code}
        except Exception as e:
            logger.error(f"[Telegram] send_message failed: {e}")
            return {"status": "failed", "error": str(e)}

    async def send_photo(self, photo_path: str, caption: str) -> dict:
        if not self.bot_token or not self.chat_id:
            return {"status": "skipped"}
        url = f"{self.base_url}/sendPhoto"
        try:
            async with httpx.AsyncClient(timeout=30) as client:
                with open(photo_path, "rb") as photo:
                    resp = await client.post(url, data={"chat_id": self.chat_id, "caption": caption}, files={"photo": photo})
                return {"status": "sent", "response_code": resp.status_code}
        except FileNotFoundError:
            logger.warning(f"[Telegram] Photo not found: {photo_path}, falling back to text message.")
            return await self.send_message(caption)
        except Exception as e:
            logger.error(f"[Telegram] send_photo failed: {e}")
            return {"status": "failed", "error": str(e)}


def create_notifier(settings=None) -> BaseNotifier:
    if settings is None:
        from web.app.core.config import settings as _s
        settings = _s
    return TelegramNotifier(settings.TELEGRAM_BOT_TOKEN, settings.TELEGRAM_CHAT_ID)
