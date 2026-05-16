from sqlalchemy import Column, String, Integer, ForeignKey, DateTime
from sqlalchemy.sql import func
from web.app.core.database import Base


class Alert(Base):
    __tablename__ = "alerts"

    id = Column(String, primary_key=True, index=True)
    camera_id = Column(String, ForeignKey("cameras.id"), nullable=True)
    zone_id = Column(String, ForeignKey("zones.id"), nullable=True)
    alert_type = Column(String, default="threshold_exceeded")
    severity = Column(String, default="medium")
    message = Column(String, nullable=True)
    people_count = Column(Integer, default=0)
    snapshot_path = Column(String, nullable=True)
    hardware_status = Column(String, default="pending")  # pending, sent, failed, skipped
    telegram_status = Column(String, default="pending")  # pending, sent, failed, skipped
    created_at = Column(DateTime, server_default=func.now())
