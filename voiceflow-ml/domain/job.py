"""Pure domain entity for a batch inference job.

No SQLAlchemy / FastAPI / Redis imports allowed in this module (ADR-001).
"""
from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Optional


class JobStatus(str, Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"

    @classmethod
    def terminal(cls) -> set["JobStatus"]:
        return {cls.COMPLETED, cls.FAILED}


def _now_utc() -> datetime:
    return datetime.now(timezone.utc)


@dataclass
class BatchJob:
    job_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    audio_path: str = ""
    status: JobStatus = JobStatus.PENDING
    created_at: datetime = field(default_factory=_now_utc)
    updated_at: datetime = field(default_factory=_now_utc)
    result: Optional[dict[str, Any]] = None
    error: Optional[str] = None
    callback_url: Optional[str] = None
    model_version: Optional[str] = None

    _ALLOWED_TRANSITIONS = {
        JobStatus.PENDING: {JobStatus.PROCESSING, JobStatus.FAILED},
        JobStatus.PROCESSING: {JobStatus.COMPLETED, JobStatus.FAILED},
        JobStatus.COMPLETED: set(),
        JobStatus.FAILED: set(),
    }

    def transition_to(self, new_status: JobStatus) -> None:
        allowed = BatchJob._ALLOWED_TRANSITIONS[self.status]
        if new_status not in allowed:
            raise ValueError(
                f"Illegal status transition: {self.status.value} -> {new_status.value}"
            )
        self.status = new_status
        self.updated_at = _now_utc()

    def mark_completed(self, result: dict[str, Any]) -> None:
        self.transition_to(JobStatus.COMPLETED)
        self.result = result

    def mark_failed(self, error: str) -> None:
        self.transition_to(JobStatus.FAILED)
        self.error = error

    @property
    def is_terminal(self) -> bool:
        return self.status in JobStatus.terminal()
