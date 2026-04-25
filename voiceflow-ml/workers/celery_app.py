"""Celery application factory.

The broker / result backend point at the same Redis instance used by the API.
"""
from __future__ import annotations

from celery import Celery

from core.config import settings


def _build_celery() -> Celery:
    app = Celery(
        "voiceflow_workers",
        broker=settings.redis_url,
        backend=settings.redis_url,
        include=["workers.inference_tasks"],
    )
    app.conf.task_serializer = "json"
    app.conf.result_serializer = "json"
    app.conf.accept_content = ["json"]
    app.conf.task_acks_late = True
    app.conf.worker_prefetch_multiplier = 1
    app.conf.task_time_limit = 30 * 60
    app.conf.task_soft_time_limit = 25 * 60
    app.conf.broker_connection_retry_on_startup = True
    return app


celery_app = _build_celery()


class CeleryTaskQueue:
    """Adapter that satisfies ``services._TaskQueueProtocol`` using Celery."""

    def __init__(self, app: Celery = celery_app) -> None:
        self._app = app

    def enqueue_inference(self, job_id: str) -> None:
        self._app.send_task(
            "voiceflow.inference.process_batch",
            args=[job_id],
            queue="inference",
        )
