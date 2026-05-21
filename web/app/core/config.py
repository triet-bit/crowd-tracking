from pydantic_settings import BaseSettings
from typing import Optional

import os
from pathlib import Path
from dotenv import load_dotenv
dotenv_path = Path(__file__).resolve().parents[2] / '.env'
load_dotenv(dotenv_path)

class Settings(BaseSettings):
    APP_ENV: str = os.environ.get("APP_ENV", "development")
    APP_HOST: str = os.environ.get("APP_HOST", "0.0.0.0")
    APP_PORT: int = int(os.environ.get("APP_PORT", "8000"))
    LOG_LEVEL: str = os.environ.get("LOG_LEVEL", "INFO")
    
    DATABASE_URL: str = os.environ.get("DATABASE_URL", "sqlite:///./crowd_tracking.db")
    
    REDIS_URL: str = os.environ.get("REDIS_URL", "redis://localhost:6379/0")
    CELERY_BROKER_URL: str = os.environ.get("CELERY_BROKER_URL", "redis://localhost:6379/1")
    CELERY_RESULT_BACKEND: str = os.environ.get("CELERY_RESULT_BACKEND", "redis://localhost:6379/2")

    # AI Engine: "mock" or "crowd_tracking"
    AI_ENGINE: str = os.environ.get("AI_ENGINE", "crowd_tracking")
    
    DEFAULT_CAMERA_SOURCE_TYPE: str = os.environ.get("DEFAULT_CAMERA_SOURCE_TYPE", "mock")
    DEFAULT_CAMERA_SOURCE_URL: str = os.environ.get("DEFAULT_CAMERA_SOURCE_URL", "")

    # Hardware: "mock_public_api" or "real_hardware"
    HARDWARE_PROVIDER: str = os.environ.get("HARDWARE_PROVIDER", "mock_public_api")
    MOCK_HARDWARE_URL: str = os.environ.get("MOCK_HARDWARE_URL", "https://httpbin.org/post")
    REAL_HARDWARE_BASE_URL: str = os.environ.get("REAL_HARDWARE_BASE_URL", "io.adafruit.com")
    AIO_USERNAME: str = os.environ.get("AIO_USERNAME", "")
    AIO_KEY: str = os.environ.get("AIO_KEY", "")

    TELEGRAM_BOT_TOKEN: str = os.environ.get("TELEGRAM_BOT_TOKEN", "")
    TELEGRAM_CHAT_ID: str = os.environ.get("TELEGRAM_CHAT_ID", "")
    
    ALERT_COOLDOWN_SECONDS: int = int(os.environ.get("ALERT_COOLDOWN_SECONDS", "30"))
    SNAPSHOT_DIR: str = os.environ.get("SNAPSHOT_DIR", "web/app/static/snapshots")

    class Config:
        env_file = "web/.env"
        env_file_encoding = "utf-8"


settings = Settings()
