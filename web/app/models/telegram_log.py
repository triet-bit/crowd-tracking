from sqlalchemy import Column, String, Integer, DateTime, ForeignKey
from sqlalchemy.sql import func

from web.app.core.database import Base


class TelegramLog(Base):
    __tablename__ = "telegram_logs"

    id = Column(String, primary_key=True, index=True)
    alert_id = Column(String, ForeignKey("alerts.id"), nullable=True, index=True)
    camera_id = Column(String, ForeignKey("cameras.id"), nullable=True, index=True)
    send_type = Column(String, default="message")  # message, photo
    status = Column(String, default="pending")  # pending, sent, failed, skipped
    response_code = Column(Integer, nullable=True)
    error = Column(String, nullable=True)
    message_preview = Column(String, nullable=True)
    created_at = Column(DateTime, server_default=func.now())