"""Celery tasks: drive batch inference end-to-end.

The task is intentionally thin — it transitions the job through its lifecycle and
delegates the heavy lifting to the Rust ``/infer`` service.
"""
from __future__ import annotations

from typing import Any

import httpx
import structlog

from core.config import settings
from core.database import SessionLocal
from repositories.job_repository import JobRepository
from services.inference_service import InferenceService
from workers.celery_app import CeleryTaskQueue, celery_app

logger = structlog.get_logger(__name__)


def _call_rust_inference(audio_path: str) -> dict[str, Any]:
    """Invoke the Rust inference service synchronously inside a Celery worker."""
    url = f"{settings.rust_service_url.rstrip('/')}/infer"
    payload = {"audio_path": audio_path, "sample_rate": settings.sample_rate}
    timeout = httpx.Timeout(settings.rust_service_timeout)
    with httpx.Client(timeout=timeout) as client:
        response = client.post(url, json=payload)
        response.raise_for_status()
        return response.json()


def _post_callback(callback_url: str, body: dict[str, Any]) -> None:
    try:
        with httpx.Client(timeout=httpx.Timeout(10.0)) as client:
            client.post(callback_url, json=body).raise_for_status()
    except httpx.HTTPError as exc:
        logger.warning("callback_post_failed", url=callback_url, error=str(exc))


@celery_app.task(
    bind=True,
    name="voiceflow.inference.process_batch",
    max_retries=3,
    default_retry_delay=30,
)
def process_batch_inference(self, job_id: str) -> dict[str, Any]:  # type: ignore[override]
    """Run a single batch inference job.

    Lifecycle: PENDING -> PROCESSING -> COMPLETED|FAILED.
    """
    session = SessionLocal()
    try:
        repo = JobRepository(session)
        service = InferenceService(repo, CeleryTaskQueue())
        job = service.mark_processing(job_id)
        try:
            result = _call_rust_inference(job.audio_path)
        except httpx.HTTPError as exc:
            logger.error("rust_inference_failed", job_id=job_id, error=str(exc))
            service.mark_failed(job_id, f"rust_inference_error: {exc}")
            raise self.retry(exc=exc)

        completed = service.mark_completed(job_id, result)
        if completed.callback_url:
            _post_callback(
                completed.callback_url,
                {"job_id": job_id, "status": completed.status.value, "result": result},
            )
        return {"job_id": job_id, "status": completed.status.value}
    finally:
        session.close()
