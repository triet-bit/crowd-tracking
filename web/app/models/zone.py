from sqlalchemy import Column, String, Integer, Float, ForeignKey, DateTime
from sqlalchemy.sql import func
from web.app.core.database import Base


class Zone(Base):
    __tablename__ = "zones"

    id = Column(String, primary_key=True, index=True)
    camera_id = Column(String, ForeignKey("cameras.id"), nullable=False)
    name = Column(String, nullable=False)
    polygon_json = Column(String, nullable=True)
    max_people_threshold = Column(Integer, default=10)
    loitering_time_threshold = Column(Integer, default=30)
    created_at = Column(DateTime, server_default=func.now())
    updated_at = Column(DateTime, onupdate=func.now())
