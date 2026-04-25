"""Repository for ``BatchJob`` aggregate.

This module is the only place outside of ``core/`` allowed to import SQLAlchemy.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Optional, Sequence

from sqlalchemy import Column, DateTime, String, Text
from sqlalchemy.orm import Session

from core.database import Base
from domain.job import BatchJob, JobStatus


class BatchJobModel(Base):
    __tablename__ = "batch_jobs"

    job_id = Column(String(36), primary_key=True)
    audio_path = Column(String(2048), nullable=False)
    status = Column(String(32), nullable=False, default=JobStatus.PENDING.value)
    created_at = Column(DateTime(timezone=True), nullable=False)
    updated_at = Column(DateTime(timezone=True), nullable=False)
    callback_url = Column(String(2048), nullable=True)
    model_version = Column(String(64), nullable=True)
    result_json = Column(Text, nullable=True)
    error = Column(Text, nullable=True)


def _to_domain(row: BatchJobModel) -> BatchJob:
    return BatchJob(
        job_id=row.job_id,
        audio_path=row.audio_path,
        status=JobStatus(row.status),
        created_at=row.created_at,
        updated_at=row.updated_at,
        callback_url=row.callback_url,
        model_version=row.model_version,
        result=json.loads(row.result_json) if row.result_json else None,
        error=row.error,
    )


def _apply_to_row(row: BatchJobModel, job: BatchJob) -> None:
    row.job_id = job.job_id
    row.audio_path = job.audio_path
    row.status = job.status.value
    row.created_at = job.created_at
    row.updated_at = job.updated_at
    row.callback_url = job.callback_url
    row.model_version = job.model_version
    row.result_json = json.dumps(job.result) if job.result is not None else None
    row.error = job.error


class JobRepository:
    """SQLAlchemy-backed repository for ``BatchJob``."""

    def __init__(self, session: Session) -> None:
        self._session = session

    def get_by_id(self, entity_id: str) -> Optional[BatchJob]:
        row = self._session.get(BatchJobModel, entity_id)
        return _to_domain(row) if row else None

    def create(self, entity: BatchJob) -> BatchJob:
        row = BatchJobModel()
        _apply_to_row(row, entity)
        self._session.add(row)
        self._session.commit()
        self._session.refresh(row)
        return _to_domain(row)

    def update(self, entity: BatchJob) -> BatchJob:
        row = self._session.get(BatchJobModel, entity.job_id)
        if row is None:
            raise LookupError(f"Job {entity.job_id} not found for update")
        entity.updated_at = datetime.now(timezone.utc)
        _apply_to_row(row, entity)
        self._session.commit()
        self._session.refresh(row)
        return _to_domain(row)

    def list(self, limit: int = 50, offset: int = 0) -> Sequence[BatchJob]:
        rows = (
            self._session.query(BatchJobModel)
            .order_by(BatchJobModel.created_at.desc())
            .offset(offset)
            .limit(limit)
            .all()
        )
        return [_to_domain(r) for r in rows]
