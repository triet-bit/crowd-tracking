"""Stats Celery tasks - lưu CountStat định kỳ."""
import logging
from web.app.core.celery_app import celery_app
from web.app.core.redis import is_camera_running

logger = logging.getLogger(__name__)


@celery_app.task(name="web.app.tasks.stats_tasks.save_count_stat")
def save_count_stat(payload: dict):
    """Lưu CountStat vào database."""
    camera_id = payload.get("camera_id")
    if camera_id and not is_camera_running(camera_id):
        logger.info(f"[stats_tasks] Skip stats for stopped camera={camera_id}")
        return {"status": "skipped", "reason": "camera_stopped"}

    try:
        from web.app.core.database import SessionLocal
        from web.app.models.count_stat import CountStat
        import uuid
        from datetime import datetime

        db = SessionLocal()
        stat = CountStat(
            id=str(uuid.uuid4())[:8],
            camera_id=payload.get("camera_id"),
            zone_id=payload.get("zone_id"),
            people_count=payload.get("people_count", 0),
            enter_count=payload.get("enter_count", 0),
            exit_count=payload.get("exit_count", 0),
            total_tracks=payload.get("total_tracks", 0),
            timestamp=datetime.now(),
        )
        db.add(stat)
        db.commit()
        db.close()
    except Exception as e:
        logger.error(f"[stats_tasks] Failed to save count stat: {e}")
