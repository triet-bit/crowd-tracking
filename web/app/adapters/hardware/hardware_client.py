"""Hardware adapters - gửi lệnh xuống phần cứng."""
import httpx
import logging
import time
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

    def __init__(self, base_url: str, username: str, key: str):
        self.base_url = base_url.rstrip("/")
        self.username = username
        self.key = key
        from Adafruit_IO import Client
        self.aio = Client(self.username, self.key)
        
    async def send_command(self, command: dict) -> dict:
        payload = command.get("payload", {})
        camera_id = payload.get("camera_id", "unknown")
        severity = payload.get("severity", "medium")
        people_count = payload.get("people_count", 0)
        msg = f"[{severity}] {people_count} Ps\n{camera_id}"
        print("sending to hardware:"+msg)
        try:
            self.aio.send_data('led-command', 'ON')
            self.aio.send_data('lcd-message', msg)
            # Wait for 3 seconds
            time.sleep(3)
            self.aio.send_data('led-command', 'OFF')
            self.aio.send_data('lcd-message', 'No warning')
            return {"status": "sent", "response_code": 200}
        except Exception as e:
            logger.error(f"[Hardware Real] Failed to send to hardware: {e}")
            return {"status": "failed", "error": str(e)}


def create_hardware_client(settings=None):
    if settings is None:
        from web.app.core.config import settings as _s
        settings = _s
    if settings.HARDWARE_PROVIDER == "mock_public_api":
        return MockPublicHardwareClient(settings.MOCK_HARDWARE_URL)
    if settings.HARDWARE_PROVIDER == "real_hardware":
        return RealHardwareClient(settings.REAL_HARDWARE_BASE_URL, settings.AIO_USERNAME, settings.AIO_KEY)
    raise ValueError(f"Unknown HARDWARE_PROVIDER: {settings.HARDWARE_PROVIDER}")
