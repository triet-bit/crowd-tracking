"""Hardware test route."""
import asyncio
from fastapi import APIRouter
from web.app.adapters.hardware.hardware_client import create_hardware_client

router = APIRouter(tags=["hardware"])


@router.post("/hardware/test-alert")
async def test_hardware(camera_id: str = "cam_test"):
    client = create_hardware_client()
    result = await client.send_command({
        "target": "test_device",
        "command": "TURN_ON_ALARM",
        "duration_seconds": 5,
        "payload": {"camera_id": camera_id, "people_count": 99, "severity": "test"},
    })
    return {"status": "sent", "result": result}
