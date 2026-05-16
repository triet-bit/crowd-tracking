from celery import Celery
from .config import settings

celery_app = Celery(
    "crowd_tracking",
    broker=settings.CELERY_BROKER_URL,
    backend=settings.CELERY_RESULT_BACKEND,
    include=[
        "web.app.tasks.stream_tasks",
        "web.app.tasks.alert_tasks",
        "web.app.tasks.telegram_tasks",
        "web.app.tasks.hardware_tasks",
        "web.app.tasks.stats_tasks",
    ]
)

celery_app.conf.update(
    task_routes={
        "web.app.tasks.stream_tasks.*": {"queue": "stream_queue"},
        "web.app.tasks.alert_tasks.*": {"queue": "alert_queue"},
        "web.app.tasks.telegram_tasks.*": {"queue": "notification_queue"},
        "web.app.tasks.hardware_tasks.*": {"queue": "hardware_queue"},
        "web.app.tasks.stats_tasks.*": {"queue": "stats_queue"},
    },
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="Asia/Ho_Chi_Minh",
    enable_utc=True,
)
