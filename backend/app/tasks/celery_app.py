from celery import Celery
from celery.schedules import crontab  # noqa: F401

from app.config import settings

celery_app = Celery(
    "tasks",
    broker=f"redis://{settings.REDIS_HOST}:{settings.REDIS_PORT}",
    include=[
        "app.tasks.scheduler",
    ]
)

celery_app.conf.timezone = 'Europe/Moscow'

# celery_app.conf.beat_schedule = {
#     "name_": {
#         "task": "gen_recs",
#         "schedule": crontab(minute="30", hour="18")
#     }
# }