"""Health check routes."""
from fastapi import APIRouter
from web.app.core.redis import redis_client

router = APIRouter(tags=["health"])


@router.get("/health")
def health():
    return {"status": "ok", "service": "crowd-tracking-web"}


@router.get("/health/redis")
def health_redis():
    try:
        redis_client.ping()
        return {"redis": "connected"}
    except Exception as e:
        return {"redis": "error", "detail": str(e)}


@router.get("/health/worker")
def health_worker():
    try:
        from web.app.core.celery_app import celery_app
        inspect = celery_app.control.inspect()
        active = inspect.active()
        if active:
            return {"worker": "online"}
        return {"worker": "offline"}
    except Exception as e:
        return {"worker": "error", "detail": str(e)}
