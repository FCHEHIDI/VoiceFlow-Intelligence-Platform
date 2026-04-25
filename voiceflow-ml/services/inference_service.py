"""Inference service — orchestrates batch jobs across the repository and the worker queue.

This module is intentionally framework-agnostic. The route layer wires it into
FastAPI through Depends; the worker process consumes its outputs.
"""
from __future__ import annotations

from typing import Optional, Protocol

from domain.exceptions import InvalidJobInputError, JobNotFoundError
from domain.job import BatchJob, JobStatus


class _JobRepoProtocol(Protocol):
    def get_by_id(self, entity_id: str) -> Optional[BatchJob]: ...
    def create(self, entity: BatchJob) -> BatchJob: ...
    def update(self, entity: BatchJob) -> BatchJob: ...


class _TaskQueueProtocol(Protocol):
    def enqueue_inference(self, job_id: str) -> None: ...


class InferenceService:
    """Application service for batch inference jobs."""

    def __init__(
        self,
        job_repo: _JobRepoProtocol,
        task_queue: _TaskQueueProtocol,
    ) -> None:
        self._job_repo = job_repo
        self._task_queue = task_queue

    def submit_batch_job(
        self,
        audio_path: str,
        callback_url: Optional[str] = None,
    ) -> BatchJob:
        """Persist a new ``BatchJob`` in PENDING and enqueue it for processing."""
        if not audio_path or not audio_path.strip():
            raise InvalidJobInputError("audio_path must be a non-empty string")

        job = BatchJob(audio_path=audio_path.strip(), callback_url=callback_url)
        stored = self._job_repo.create(job)
        self._task_queue.enqueue_inference(stored.job_id)
        return stored

    def get_job_status(self, job_id: str) -> BatchJob:
        """Return a job by id or raise ``JobNotFoundError``."""
        if not job_id or not job_id.strip():
            raise InvalidJobInputError("job_id must be a non-empty string")

        job = self._job_repo.get_by_id(job_id)
        if job is None:
            raise JobNotFoundError(job_id)
        return job

    def mark_processing(self, job_id: str) -> BatchJob:
        job = self.get_job_status(job_id)
        job.transition_to(JobStatus.PROCESSING)
        return self._job_repo.update(job)

    def mark_completed(self, job_id: str, result: dict) -> BatchJob:
        job = self.get_job_status(job_id)
        job.mark_completed(result)
        return self._job_repo.update(job)

    def mark_failed(self, job_id: str, error: str) -> BatchJob:
        job = self.get_job_status(job_id)
        job.mark_failed(error)
        return self._job_repo.update(job)
