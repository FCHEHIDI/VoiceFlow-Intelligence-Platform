"""Unit tests for ``services.inference_service.InferenceService``.

We cover the framework-agnostic contract: the service must persist a job in
PENDING, dispatch a task via the provided queue adapter, and surface domain
errors for missing jobs or invalid inputs. No FastAPI / SQLAlchemy / Redis
is touched here — that's the integration layer's job.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from domain.exceptions import InvalidJobInputError, JobNotFoundError
from domain.job import BatchJob, JobStatus
from services.inference_service import InferenceService

pytestmark = pytest.mark.unit


@pytest.fixture
def service(mock_job_repo, mock_task_queue) -> InferenceService:
    return InferenceService(mock_job_repo, mock_task_queue)


class TestSubmitBatchJob:
    def test_happy_path_returns_pending_persisted_dispatched(
        self, service: InferenceService, mock_job_repo, mock_task_queue: MagicMock
    ) -> None:
        job = service.submit_batch_job("/data/audio.wav", callback_url="http://cb/x")

        # Domain assertions
        assert isinstance(job, BatchJob)
        assert job.status is JobStatus.PENDING
        assert job.audio_path == "/data/audio.wav"
        assert job.callback_url == "http://cb/x"

        # Persistence assertions
        persisted = mock_job_repo.get(job.job_id)
        assert persisted is not None
        assert persisted.status is JobStatus.PENDING

        # Dispatch assertions: the task queue's enqueue_inference must be called once
        mock_task_queue.enqueue_inference.assert_called_once_with(job.job_id)

    def test_strips_audio_path_whitespace(
        self, service: InferenceService, mock_job_repo
    ) -> None:
        job = service.submit_batch_job("   /data/audio.wav   ")
        assert job.audio_path == "/data/audio.wav"
        persisted = mock_job_repo.get(job.job_id)
        assert persisted is not None
        assert persisted.audio_path == "/data/audio.wav"

    @pytest.mark.parametrize("bad", ["", "   ", "\t\n"])
    def test_empty_audio_path_raises_invalid_job_input(
        self, service: InferenceService, mock_task_queue: MagicMock, bad: str
    ) -> None:
        with pytest.raises(InvalidJobInputError):
            service.submit_batch_job(bad)
        # Must not enqueue anything when the input was rejected.
        mock_task_queue.enqueue_inference.assert_not_called()

    def test_callback_url_optional(self, service: InferenceService) -> None:
        job = service.submit_batch_job("/data/audio.wav")
        assert job.callback_url is None


class TestGetJobStatus:
    def test_returns_persisted_job(
        self, service: InferenceService, mock_job_repo
    ) -> None:
        created = service.submit_batch_job("/data/audio.wav")
        fetched = service.get_job_status(created.job_id)
        assert fetched.job_id == created.job_id
        assert fetched.status is JobStatus.PENDING

    def test_unknown_id_raises_not_found(self, service: InferenceService) -> None:
        with pytest.raises(JobNotFoundError) as exc_info:
            service.get_job_status("does-not-exist")
        assert exc_info.value.job_id == "does-not-exist"

    @pytest.mark.parametrize("bad", ["", "  "])
    def test_empty_job_id_raises_invalid_input(
        self, service: InferenceService, bad: str
    ) -> None:
        with pytest.raises(InvalidJobInputError):
            service.get_job_status(bad)


class TestStateTransitions:
    def test_mark_processing_then_completed(
        self, service: InferenceService, mock_job_repo
    ) -> None:
        created = service.submit_batch_job("/data/audio.wav")
        service.mark_processing(created.job_id)
        assert mock_job_repo.get(created.job_id).status is JobStatus.PROCESSING

        service.mark_completed(created.job_id, {"speakers": 3})
        final = mock_job_repo.get(created.job_id)
        assert final.status is JobStatus.COMPLETED
        assert final.result == {"speakers": 3}

    def test_mark_failed_records_error(
        self, service: InferenceService, mock_job_repo
    ) -> None:
        created = service.submit_batch_job("/data/audio.wav")
        service.mark_failed(created.job_id, "boom")
        final = mock_job_repo.get(created.job_id)
        assert final.status is JobStatus.FAILED
        assert final.error == "boom"
