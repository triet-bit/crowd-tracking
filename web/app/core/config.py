from pydantic_settings import BaseSettings
from typing import Optional


class Settings(BaseSettings):
    APP_ENV: str = "development"
    APP_HOST: str = "0.0.0.0"
    APP_PORT: int = 8000
    LOG_LEVEL: str = "INFO"

    DATABASE_URL: str = "sqlite:///./crowd_tracking.db"

    REDIS_URL: str = "redis://localhost:6379/0"
    CELERY_BROKER_URL: str = "redis://localhost:6379/1"
    CELERY_RESULT_BACKEND: str = "redis://localhost:6379/2"

    # AI Engine: "mock" or "crowd_tracking"
    AI_ENGINE: str = "crowd_tracking"

    DEFAULT_CAMERA_SOURCE_TYPE: str = "mock"
    DEFAULT_CAMERA_SOURCE_URL: str = ""

    # Hardware: "mock_public_api" or "real_hardware"
    HARDWARE_PROVIDER: str = "mock_public_api"
    MOCK_HARDWARE_URL: str = "https://httpbin.org/post"
    REAL_HARDWARE_BASE_URL: str = ""

    TELEGRAM_BOT_TOKEN: str = ""
    TELEGRAM_CHAT_ID: str = ""

    ALERT_COOLDOWN_SECONDS: int = 30
    SNAPSHOT_DIR: str = "web/app/static/snapshots"

    class Config:
        env_file = "web/.env"
        env_file_encoding = "utf-8"


settings = Settings()
