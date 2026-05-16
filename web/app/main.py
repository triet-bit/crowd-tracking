"""FastAPI main application."""
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from web.app.core.config import settings
from web.app.core.database import create_tables

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup/shutdown events."""
    logger.info("Starting Crowd Tracking Web API...")
    create_tables()
    logger.info(f"AI Engine: {settings.AI_ENGINE}")
    logger.info(f"Hardware Provider: {settings.HARDWARE_PROVIDER}")
    yield
    logger.info("Shutting down...")


app = FastAPI(
    title="Crowd Tracking Web API",
    description="Realtime crowd monitoring with ByteTrack + YOLO",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Static files
import os
os.makedirs("web/app/static/snapshots", exist_ok=True)
app.mount("/static", StaticFiles(directory="web/app/static"), name="static")

# API Routes
from web.app.api.routes_health import router as health_router
from web.app.api.routes_camera import router as camera_router
from web.app.api.routes_alert import router as alert_router
from web.app.api.routes_stats import router as stats_router
from web.app.api.routes_hardware import router as hardware_router
from web.app.api.routes_telegram import router as telegram_router
from web.app.ws.routes_ws import router as ws_router

app.include_router(health_router)
app.include_router(camera_router, prefix="/api")
app.include_router(alert_router, prefix="/api")
app.include_router(stats_router, prefix="/api")
app.include_router(hardware_router, prefix="/api")
app.include_router(telegram_router, prefix="/api")
app.include_router(ws_router)

# Serve dashboard frontend
from fastapi.responses import FileResponse

@app.get("/")
async def serve_dashboard():
    return FileResponse("web/frontend/index.html")
