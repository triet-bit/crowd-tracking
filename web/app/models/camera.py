from sqlalchemy import Column, String, Boolean, DateTime
from sqlalchemy.sql import func
from web.app.core.database import Base


class Camera(Base):
    __tablename__ = "cameras"

    id = Column(String, primary_key=True, index=True)
    name = Column(String, nullable=False)
    source_type = Column(String, default="mock")  # mock, webcam, video_file, rtsp
    source_url = Column(String, nullable=True)
    mode = Column(String, default="loitering")     # tracking, geofence, loitering
    is_active = Column(Boolean, default=False)
    created_at = Column(DateTime, server_default=func.now())
    updated_at = Column(DateTime, onupdate=func.now())
