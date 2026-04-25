"""FastAPI dependency providers — wire repositories / services to the request scope."""
from __future__ import annotations

from typing import Generator

from fastapi import Depends
from sqlalchemy.orm import Session

from core import get_db
from repositories.job_repository import JobRepository
from services.inference_service import InferenceService
from workers.celery_app import CeleryTaskQueue


def get_job_repository(db: Session = Depends(get_db)) -> Generator[JobRepository, None, None]:
    yield JobRepository(db)


def get_inference_service(
    job_repo: JobRepository = Depends(get_job_repository),
) -> InferenceService:
    return InferenceService(job_repo, CeleryTaskQueue())
