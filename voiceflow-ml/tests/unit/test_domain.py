"""Pure-domain tests — no FastAPI / SQLAlchemy / Redis touched."""

from __future__ import annotations

import pytest

from domain.exceptions import DomainError, InvalidJobInputError, JobNotFoundError
from domain.job import BatchJob, JobStatus
from domain.model_version import ModelVersion
from domain.segment import Segment, Speaker

pytestmark = pytest.mark.unit


class TestBatchJob:
    def test_initial_state_is_pending(self) -> None:
        job = BatchJob(audio_path="s3://bucket/audio.wav")
        assert job.status is JobStatus.PENDING
        assert job.result is None
        assert job.error is None
        assert job.is_terminal is False
        # job_id is a uuid-like string
        assert isinstance(job.job_id, str) and len(job.job_id) >= 32

    def test_pending_to_processing_to_completed(self) -> None:
        job = BatchJob(audio_path="x")
        job.transition_to(JobStatus.PROCESSING)
        assert job.status is JobStatus.PROCESSING
        job.mark_completed({"speakers": 2})
        assert job.status is JobStatus.COMPLETED
        assert job.result == {"speakers": 2}
        assert job.is_terminal is True

    def test_pending_to_failed(self) -> None:
        job = BatchJob(audio_path="x")
        job.mark_failed("boom")
        assert job.status is JobStatus.FAILED
        assert job.error == "boom"
        assert job.is_terminal is True

    def test_invalid_transition_completed_to_processing(self) -> None:
        job = BatchJob(audio_path="x")
        job.transition_to(JobStatus.PROCESSING)
        job.mark_completed({"ok": True})
        with pytest.raises(ValueError, match="Illegal status transition"):
            job.transition_to(JobStatus.PROCESSING)

    def test_invalid_transition_pending_to_completed_directly(self) -> None:
        # PENDING → COMPLETED is not in the allowed transitions table.
        job = BatchJob(audio_path="x")
        with pytest.raises(ValueError):
            job.transition_to(JobStatus.COMPLETED)

    def test_invalid_transition_from_failed(self) -> None:
        job = BatchJob(audio_path="x")
        job.mark_failed("e")
        with pytest.raises(ValueError):
            job.transition_to(JobStatus.PROCESSING)

    def test_terminal_set_helper(self) -> None:
        assert JobStatus.terminal() == {JobStatus.COMPLETED, JobStatus.FAILED}


class TestSegment:
    def test_valid_segment(self) -> None:
        seg = Segment(start=0.0, end=1.5, speaker_id="spk_0", confidence=0.9)
        assert seg.duration == pytest.approx(1.5)

    def test_end_must_be_greater_than_start(self) -> None:
        with pytest.raises(ValueError, match="end must be greater than start"):
            Segment(start=2.0, end=1.0, speaker_id="spk_0")

    def test_end_equal_to_start_rejected(self) -> None:
        with pytest.raises(ValueError):
            Segment(start=1.0, end=1.0, speaker_id="spk_0")

    def test_negative_timestamps_rejected(self) -> None:
        with pytest.raises(ValueError, match="non-negative"):
            Segment(start=-0.1, end=1.0, speaker_id="spk_0")

    def test_confidence_out_of_range(self) -> None:
        with pytest.raises(ValueError):
            Segment(start=0.0, end=1.0, speaker_id="spk_0", confidence=1.5)

    def test_speaker_id_required_on_segment(self) -> None:
        # Empty string speaker_id is allowed by the dataclass but Speaker
        # explicitly rejects empty IDs — assert via the Speaker entity.
        with pytest.raises(ValueError, match="speaker_id"):
            Speaker(speaker_id="")

    def test_segment_is_frozen(self) -> None:
        seg = Segment(start=0.0, end=1.0, speaker_id="spk_0")
        with pytest.raises(Exception):
            seg.start = 5.0  # type: ignore[misc]


class TestModelVersion:
    def test_default_construction(self) -> None:
        mv = ModelVersion(version="1.2.3", onnx_path="models/x.onnx")
        assert mv.version == "1.2.3"
        assert mv.is_production is False
        assert isinstance(mv.metrics, dict)
        assert mv.version_id  # auto-generated uuid

    def test_promote_to_production(self) -> None:
        mv = ModelVersion(version="1.0.0")
        mv.promote_to_production()
        assert mv.is_production is True

    def test_two_versions_have_distinct_ids(self) -> None:
        a = ModelVersion(version="1.0.0")
        b = ModelVersion(version="1.0.0")
        assert a.version_id != b.version_id

    def test_metrics_dict_independent_per_instance(self) -> None:
        a = ModelVersion(version="1.0.0")
        b = ModelVersion(version="1.0.1")
        a.metrics["der"] = 0.1
        assert "der" not in b.metrics


class TestExceptions:
    def test_job_not_found_carries_id(self) -> None:
        err = JobNotFoundError("abc")
        assert err.job_id == "abc"
        assert "abc" in str(err)
        assert isinstance(err, DomainError)

    def test_invalid_job_input_is_domain_error(self) -> None:
        assert issubclass(InvalidJobInputError, DomainError)
