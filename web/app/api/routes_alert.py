"""Alert API routes."""
from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from web.app.core.database import get_db
from web.app.models.alert import Alert

router = APIRouter(tags=["alerts"])


@router.get("/alerts")
def list_alerts(camera_id: str = None, limit: int = 20, db: Session = Depends(get_db)):
    query = db.query(Alert)
    if camera_id:
        query = query.filter(Alert.camera_id == camera_id)
    alerts = query.order_by(Alert.created_at.desc()).limit(limit).all()
    return [{"id": a.id, "camera_id": a.camera_id, "zone_id": a.zone_id,
             "alert_type": a.alert_type, "severity": a.severity,
             "message": a.message, "people_count": a.people_count,
             "hardware_status": a.hardware_status, "telegram_status": a.telegram_status,
             "created_at": str(a.created_at) if a.created_at else None} for a in alerts]


@router.get("/alerts/{alert_id}")
def get_alert(alert_id: str, db: Session = Depends(get_db)):
    alert = db.query(Alert).filter(Alert.id == alert_id).first()
    if not alert:
        return {"error": "Alert not found"}
    return {"id": alert.id, "camera_id": alert.camera_id, "alert_type": alert.alert_type,
            "severity": alert.severity, "message": alert.message,
            "people_count": alert.people_count, "created_at": str(alert.created_at)}
