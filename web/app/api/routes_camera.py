"""API Routes: Camera CRUD + start/stop + MJPEG stream + Zone management."""
import uuid
import json
import time
import logging
import asyncio
from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session

from web.app.core.database import get_db
from web.app.core.redis import (
    get_camera_metrics, get_camera_status, set_camera_status,
    get_camera_frame, clear_camera_runtime, get_camera_visual_mode, set_camera_visual_mode
)
from web.app.models.camera import Camera
from web.app.models.zone import Zone

logger = logging.getLogger(__name__)
router = APIRouter(tags=["cameras"])


# ===================== Camera CRUD =====================

@router.post("/cameras")
def create_camera(payload: dict, db: Session = Depends(get_db)):
    cam_id = f"cam_{str(uuid.uuid4())[:8]}"
    camera = Camera(
        id=cam_id,
        name=payload.get("name", "Camera"),
        source_type=payload.get("source_type", "mock"),
        source_url=payload.get("source_url", ""),
        mode=payload.get("mode", "loitering"),
    )
    db.add(camera)
    db.commit()
    db.refresh(camera)
    set_camera_visual_mode(camera.id, "bytetrack")
    return {"id": camera.id, "name": camera.name, "source_type": camera.source_type,
            "mode": camera.mode, "visual_mode": "bytetrack", "is_active": camera.is_active}


@router.get("/cameras")
def list_cameras(db: Session = Depends(get_db)):
    cameras = db.query(Camera).all()
    result = []
    for c in cameras:
        # Lấy zones liên quan
        zones = db.query(Zone).filter(Zone.camera_id == c.id).all()
        zones_data = [{"id": z.id, "name": z.name, "polygon_json": z.polygon_json,
                       "max_people_threshold": z.max_people_threshold,
                       "loitering_time_threshold": z.loitering_time_threshold} for z in zones]
        result.append({
            "id": c.id, "name": c.name, "source_type": c.source_type,
            "mode": c.mode, "visual_mode": get_camera_visual_mode(c.id), "is_active": c.is_active, "zones": zones_data
        })
    return result


@router.get("/cameras/{camera_id}")
def get_camera(camera_id: str, db: Session = Depends(get_db)):
    cam = db.query(Camera).filter(Camera.id == camera_id).first()
    if not cam:
        raise HTTPException(status_code=404, detail="Camera not found")
    return {"id": cam.id, "name": cam.name, "source_type": cam.source_type,
            "mode": cam.mode, "visual_mode": get_camera_visual_mode(cam.id), "is_active": cam.is_active}


# ===================== Camera Start/Stop =====================

@router.post("/cameras/{camera_id}/start")
def start_camera(camera_id: str, db: Session = Depends(get_db)):
    cam = db.query(Camera).filter(Camera.id == camera_id).first()
    if not cam:
        raise HTTPException(status_code=404, detail="Camera not found")

    # Check đã chạy chưa
    status = get_camera_status(camera_id)
    if status and status.get("status") == "running":
        return {"camera_id": camera_id, "status": "already_running", "task_id": status.get("task_id")}

    # Đặt trạng thái starting
    set_camera_status(camera_id, {"status": "running", "task_id": None, "updated_at": ""})
    set_camera_visual_mode(camera_id, "bytetrack")

    from web.app.tasks.stream_tasks import process_camera_stream
    task = process_camera_stream.delay(camera_id)

    # Cập nhật task_id vào Redis
    set_camera_status(camera_id, {
        "status": "running",
        "task_id": task.id,
        "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    })

    # Cập nhật is_active trong DB
    cam.is_active = True
    db.commit()

    return {"camera_id": camera_id, "status": "starting", "task_id": task.id}


@router.post("/cameras/{camera_id}/stop")
def stop_camera(camera_id: str, db: Session = Depends(get_db)):
    set_camera_status(camera_id, {"status": "stopped", "task_id": None, "updated_at": ""})
    clear_camera_runtime(camera_id)
    cam = db.query(Camera).filter(Camera.id == camera_id).first()
    if cam:
        cam.is_active = False
        db.commit()
    return {"camera_id": camera_id, "status": "stopped"}


@router.get("/cameras/{camera_id}/visual-mode")
def get_visual_mode(camera_id: str, db: Session = Depends(get_db)):
    cam = db.query(Camera).filter(Camera.id == camera_id).first()
    if not cam:
        raise HTTPException(status_code=404, detail="Camera not found")
    return {"camera_id": camera_id, "visual_mode": get_camera_visual_mode(camera_id)}


@router.post("/cameras/{camera_id}/visual-mode")
def set_visual_mode(camera_id: str, payload: dict, db: Session = Depends(get_db)):
    cam = db.query(Camera).filter(Camera.id == camera_id).first()
    if not cam:
        raise HTTPException(status_code=404, detail="Camera not found")

    visual_mode = (payload.get("visual_mode") or "").strip().lower()
    if visual_mode not in {"bytetrack", "heatmap"}:
        raise HTTPException(status_code=400, detail="visual_mode must be 'bytetrack' or 'heatmap'")

    set_camera_visual_mode(camera_id, visual_mode)
    logger.info("[routes_camera] Updated visual mode: camera=%s visual_mode=%s", camera_id, visual_mode)
    return {"camera_id": camera_id, "visual_mode": visual_mode}


@router.get("/cameras/{camera_id}/latest")
def get_latest_metrics(camera_id: str):
    metrics = get_camera_metrics(camera_id)
    if not metrics:
        return {"camera_id": camera_id, "people_count": 0, "total_tracks": 0,
                "enter_count": 0, "exit_count": 0, "timestamp": None}
    return metrics


# ===================== MJPEG Video Stream =====================

async def _mjpeg_generator(camera_id: str):
    """Generator: yield JPEG frames liên tục từ Redis cho MJPEG stream."""
    while True:
        frame_bytes = get_camera_frame(camera_id)
        if frame_bytes:
            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n"
                + frame_bytes +
                b"\r\n"
            )
        await asyncio.sleep(0.05)  # ~20 FPS


@router.get("/cameras/{camera_id}/stream")
async def stream_camera(camera_id: str):
    """MJPEG live stream endpoint. Dùng <img src='.../stream'> trên frontend."""
    return StreamingResponse(
        _mjpeg_generator(camera_id),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )


# ===================== Zone CRUD =====================

@router.post("/cameras/{camera_id}/zones")
def create_zone(camera_id: str, payload: dict, db: Session = Depends(get_db)):
    """Tạo zone mới cho camera (polygon giám sát)."""
    cam = db.query(Camera).filter(Camera.id == camera_id).first()
    if not cam:
        raise HTTPException(status_code=404, detail="Camera not found")

    existing_zones = db.query(Zone).filter(Zone.camera_id == camera_id).all()
    for existing_zone in existing_zones:
        db.delete(existing_zone)
    db.flush()

    zone_id = f"zone_{str(uuid.uuid4())[:8]}"
    polygon_data = payload.get("polygon", [])

    zone = Zone(
        id=zone_id,
        camera_id=camera_id,
        name=payload.get("name", "Zone 1"),
        polygon_json=json.dumps(polygon_data) if isinstance(polygon_data, list) else polygon_data,
        max_people_threshold=payload.get("max_people_threshold", 10),
        loitering_time_threshold=payload.get("loitering_time_threshold", 2),
    )
    db.add(zone)
    db.commit()
    db.refresh(zone)

    return {
        "id": zone.id, "camera_id": camera_id, "name": zone.name,
        "polygon_json": zone.polygon_json,
        "max_people_threshold": zone.max_people_threshold,
        "loitering_time_threshold": zone.loitering_time_threshold,
    }


@router.get("/cameras/{camera_id}/zones")
def list_zones(camera_id: str, db: Session = Depends(get_db)):
    zones = db.query(Zone).filter(Zone.camera_id == camera_id).all()
    return [{"id": z.id, "camera_id": z.camera_id, "name": z.name,
             "polygon_json": z.polygon_json,
             "max_people_threshold": z.max_people_threshold,
             "loitering_time_threshold": z.loitering_time_threshold} for z in zones]


@router.delete("/cameras/{camera_id}/zones/{zone_id}")
def delete_zone(camera_id: str, zone_id: str, db: Session = Depends(get_db)):
    zone = db.query(Zone).filter(Zone.id == zone_id, Zone.camera_id == camera_id).first()
    if not zone:
        raise HTTPException(status_code=404, detail="Zone not found")
    db.delete(zone)
    db.commit()
    return {"status": "deleted", "zone_id": zone_id}

