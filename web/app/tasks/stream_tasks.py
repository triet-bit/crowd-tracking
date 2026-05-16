"""
Celery task: process_camera_stream

Task chính - chạy vòng lặp đọc camera, đưa qua AI engine,
publish metrics realtime, và trigger alert khi cần.
"""
import logging
import time
from web.app.core.celery_app import celery_app
from web.app.core.config import settings
from web.app.core.redis import (
    save_camera_metrics, set_camera_status,
    get_camera_status, publish_ws_event, check_alert_cooldown,
    save_camera_frame
)
from web.app.adapters.camera.video_file_camera import create_camera_source
from web.app.adapters.ai.factory import create_ai_engine

logger = logging.getLogger(__name__)


def _get_camera_config(camera_id: str) -> dict:
    """Lấy camera config từ database."""
    try:
        from web.app.core.database import SessionLocal
        from web.app.models.camera import Camera
        from web.app.models.zone import Zone
        db = SessionLocal()
        cam = db.query(Camera).filter(Camera.id == camera_id).first()
        zones = db.query(Zone).filter(Zone.camera_id == camera_id).all()
        db.close()
        if not cam:
            return {"id": camera_id, "source_type": "mock", "mode": "loitering"}
        config = {
            "id": cam.id,
            "source_type": cam.source_type,
            "source_url": cam.source_url or "",
            "mode": cam.mode or "loitering",
        }
        # Lấy zone đầu tiên làm mặc định
        if zones:
            z = zones[0]
            config["zone_id"] = z.id
            config["polygon_json"] = z.polygon_json
            config["max_people_threshold"] = z.max_people_threshold or 10
            config["loitering_time_threshold"] = z.loitering_time_threshold or 30
        return config
    except Exception as e:
        logger.error(f"[stream_tasks] Error getting camera config: {e}")
        return {"id": camera_id, "source_type": "mock", "mode": "loitering"}


def _camera_is_active(camera_id: str) -> bool:
    """Kiểm tra camera có đang chạy không (từ Redis)."""
    status = get_camera_status(camera_id)
    return status is not None and status.get("status") == "running"


@celery_app.task(name="web.app.tasks.stream_tasks.process_camera_stream", bind=True)
def process_camera_stream(self, camera_id: str):
    """
    Vòng lặp chính xử lý camera stream.

    Flow:
        open camera
        loop:
            read frame
            AI engine.process(frame, ...)
            save metrics to Redis
            publish WebSocket event
            check threshold -> trigger create_alert task
            save CountStat every 30 frames
        release camera
    """
    logger.info(f"[stream_tasks] Starting stream for camera: {camera_id}")

    # Cập nhật status running
    set_camera_status(camera_id, {
        "status": "running",
        "task_id": self.request.id,
        "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    })

    camera_config = _get_camera_config(camera_id)
    camera_source = create_camera_source(camera_config)
    ai_engine = create_ai_engine(settings)

    runtime_state = {
        "frame_id": 0,
        "loiter_dict": {},
        "flow_dict": {},
        "enter_count": 0,
        "exit_count": 0,
    }

    try:
        camera_source.open()
        logger.info(f"[stream_tasks] Camera opened: {camera_id}")

        while _camera_is_active(camera_id):
            success, frame = camera_source.read()
            if not success or frame is None:
                logger.warning(f"[stream_tasks] Failed to read frame from {camera_id}")
                time.sleep(0.1)
                continue

            # AI xử lý frame
            result = ai_engine.process(
                frame=frame,
                camera_config=camera_config,
                runtime_state=runtime_state,
            )

            # Lưu metrics vào Redis
            metrics = result.to_metrics_dict()
            save_camera_metrics(camera_id, metrics)

            # Encode frame đã xử lý thành JPEG và lưu Redis
            import cv2
            if result.processed_frame is not None:
                _, jpeg_buf = cv2.imencode('.jpg', result.processed_frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
                save_camera_frame(camera_id, jpeg_buf.tobytes())

            # Publish WebSocket event
            publish_ws_event(camera_id, {
                "event": "frame.metrics",
                "data": metrics,
            })

            # Kiểm tra alerts từ AI result
            threshold = camera_config.get("max_people_threshold", 10)
            zone_id = camera_config.get("zone_id", "default")

            if result.people_count > threshold:
                if not check_alert_cooldown(camera_id, zone_id, "threshold_exceeded"):
                    from web.app.tasks.alert_tasks import create_alert
                    create_alert.delay(result.to_alert_payload(zone_id=zone_id, threshold=threshold))

            # Loitering alert
            loitering_alerts = [a for a in result.alerts if "loitering" in a]
            if loitering_alerts:
                if not check_alert_cooldown(camera_id, zone_id, "loitering"):
                    from web.app.tasks.alert_tasks import create_alert
                    create_alert.delay(result.to_loitering_payload(zone_id=zone_id))

            # Lưu CountStat mỗi 30 frame
            if result.frame_id % 30 == 0:
                from web.app.tasks.stats_tasks import save_count_stat
                save_count_stat.delay({
                    "camera_id": camera_id,
                    "zone_id": zone_id,
                    "people_count": result.people_count,
                    "enter_count": result.enter_count,
                    "exit_count": result.exit_count,
                    "total_tracks": result.total_tracks,
                })

            runtime_state["frame_id"] += 1

    except Exception as e:
        logger.error(f"[stream_tasks] Error in stream loop for {camera_id}: {e}", exc_info=True)
    finally:
        camera_source.release()
        set_camera_status(camera_id, {
            "status": "stopped",
            "task_id": None,
            "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        })
        logger.info(f"[stream_tasks] Stream stopped for camera: {camera_id}")
