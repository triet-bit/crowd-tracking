from sqlalchemy import Column, String, Integer, Float, ForeignKey, DateTime
from sqlalchemy.sql import func
from web.app.core.database import Base


class CountStat(Base):
    __tablename__ = "count_stats"

    id = Column(String, primary_key=True, index=True)
    camera_id = Column(String, ForeignKey("cameras.id"), nullable=True)
    zone_id = Column(String, nullable=True)
    people_count = Column(Integer, default=0)
    enter_count = Column(Integer, default=0)
    exit_count = Column(Integer, default=0)
    total_tracks = Column(Integer, default=0)
    timestamp = Column(DateTime, server_default=func.now())


class HeatmapPoint(Base):
    __tablename__ = "heatmap_points"

    id = Column(String, primary_key=True, index=True)
    camera_id = Column(String, ForeignKey("cameras.id"), nullable=True)
    zone_id = Column(String, nullable=True)
    x = Column(Float, nullable=False)
    y = Column(Float, nullable=False)
    weight = Column(Float, default=1.0)
    timestamp = Column(DateTime, server_default=func.now())
