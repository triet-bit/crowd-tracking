"""Hardware adapters - gửi lệnh xuống phần cứng."""
import httpx
import logging

logger = logging.getLogger(__name__)


class BaseHardwareClient:
    async def send_command(self, command: dict) -> dict:
        raise NotImplementedError


class MockPublicHardwareClient(BaseHardwareClient):
    """Gửi POST tới public API (httpbin.org) để test."""

    def __init__(self, url: str):
        self.url = url

    async def send_command(self, command: dict) -> dict:
        try:
            async with httpx.AsyncClient(timeout=10) as client:
                resp = await client.post(self.url, json=command)
                logger.info(f"[Hardware Mock] Sent command to {self.url}, status={resp.status_code}")
                return {"status": "sent", "response_code": resp.status_code}
        except Exception as e:
            logger.error(f"[Hardware Mock] Failed: {e}")
            return {"status": "failed", "error": str(e)}


class RealHardwareClient(BaseHardwareClient):
    """Gửi lệnh xuống phần cứng thật (ESP32, Arduino...)."""

    def __init__(self, base_url: str):
        self.base_url = base_url.rstrip("/")

    async def send_command(self, command: dict) -> dict:
        cmd_type = command.get("command", "TURN_ON_ALARM")
        endpoint_map = {
            "TURN_ON_ALARM": "/alarm",
            "LCD_DISPLAY": "/lcd",
            "RED_LIGHT": "/red-light",
        }
        path = endpoint_map.get(cmd_type, "/command")
        url = f"{self.base_url}{path}"
        try:
            async with httpx.AsyncClient(timeout=10) as client:
                resp = await client.post(url, json=command)
                return {"status": "sent", "response_code": resp.status_code}
        except Exception as e:
            logger.error(f"[Hardware Real] Failed to send to {url}: {e}")
            return {"status": "failed", "error": str(e)}


def create_hardware_client(settings=None):
    if settings is None:
        from web.app.core.config import settings as _s
        settings = _s
    if settings.HARDWARE_PROVIDER == "mock_public_api":
        return MockPublicHardwareClient(settings.MOCK_HARDWARE_URL)
    if settings.HARDWARE_PROVIDER == "real_hardware":
        return RealHardwareClient(settings.REAL_HARDWARE_BASE_URL)
    raise ValueError(f"Unknown HARDWARE_PROVIDER: {settings.HARDWARE_PROVIDER}")
