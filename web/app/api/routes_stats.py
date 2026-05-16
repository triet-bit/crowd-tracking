"""Stats, Hardware, and Telegram test routes."""
from fastapi import APIRouter

router = APIRouter(tags=["stats"])


@router.get("/stats/people-density")
def people_density(camera_id: str = None):
    from web.app.core.database import SessionLocal
    from web.app.models.count_stat import CountStat
    db = SessionLocal()
    query = db.query(CountStat)
    if camera_id:
        query = query.filter(CountStat.camera_id == camera_id)
    stats = query.order_by(CountStat.timestamp.desc()).limit(100).all()
    db.close()
    return {"camera_id": camera_id, "series": [
        {"time": str(s.timestamp), "count": s.people_count, "enter": s.enter_count, "exit": s.exit_count}
        for s in stats
    ]}
