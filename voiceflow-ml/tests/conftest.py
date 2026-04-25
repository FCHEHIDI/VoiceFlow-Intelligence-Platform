"""Pytest fixtures for the voiceflow-ml service.

Sets required env vars before the app's Settings class is instantiated, then
exposes a small set of additive fixtures used by both the unit and integration
test suites:

* ``mock_job_repo``   — in-memory fake for ``services._JobRepoProtocol``
* ``mock_task_queue`` — Mock that satisfies ``services._TaskQueueProtocol``
* ``mock_redis``      — MagicMock that quacks like ``redis.Redis``
* ``test_client``     — FastAPI ``TestClient`` with the production
                         ``get_job_repository`` / ``get_inference_service``
                         dependencies overridden to use the fakes above
* ``sample_wav_bytes``  — 5 s of silence, 16 kHz, mono, 16-bit PCM
* ``fake_zip_bytes``    — minimal ZIP magic header bytes
* ``oversized_wav_path`` — path to a ``> 100 MB`` RIFF blob streamed to disk

The streaming temp-file pattern for the oversized fixture keeps memory usage
bounded; tests that only need to assert on ``Content-Length`` can use
``oversized_content_length`` instead.
"""

from __future__ import annotations

import io
import os
import wave
from pathlib import Path
from typing import Dict, Iterator, List, Optional
from unittest.mock import MagicMock

import pytest

# Required by core.config — must be set BEFORE any app module is imported.
os.environ.setdefault(
    "JWT_SECRET_KEY",
    "00000000000000000000000000000000",
)
os.environ.setdefault("POSTGRES_PASSWORD", "pytest-test-db-password-32chars!")


# ---------------------------------------------------------------------------
# In-memory fakes
# ---------------------------------------------------------------------------


class InMemoryJobRepo:
    """Minimal in-memory repo satisfying ``services._JobRepoProtocol``.

    The protocol used by ``InferenceService`` exposes ``get_by_id``,
    ``create`` and ``update``. We expose those plus convenience helpers
    (``add``, ``get``, ``update_status``) for direct use from unit tests.
    """

    def __init__(self) -> None:
        from domain.job import BatchJob, JobStatus  # local import keeps domain layer pure

        self._BatchJob = BatchJob
        self._JobStatus = JobStatus
        self._store: Dict[str, BatchJob] = {}

    # _JobRepoProtocol surface --------------------------------------------------
    def get_by_id(self, entity_id):
        job = self._store.get(entity_id)
        if job is None:
            return None
        return self._BatchJob(
            job_id=job.job_id,
            audio_path=job.audio_path,
            status=job.status,
            created_at=job.created_at,
            updated_at=job.updated_at,
            result=job.result,
            error=job.error,
            callback_url=job.callback_url,
            model_version=job.model_version,
        )

    def create(self, entity):
        self._store[entity.job_id] = entity
        return entity

    def update(self, entity):
        if entity.job_id not in self._store:
            raise LookupError(f"Job {entity.job_id} not found for update")
        self._store[entity.job_id] = entity
        return entity

    # Helpers used directly by unit tests --------------------------------------
    def add(self, job) -> None:
        self._store[job.job_id] = job

    def get(self, job_id):
        return self._store.get(job_id)

    def update_status(self, job_id: str, status) -> None:
        job = self._store[job_id]
        job.transition_to(status)

    def list(self, limit: int = 50, offset: int = 0) -> List:
        items = list(self._store.values())
        return items[offset : offset + limit]


@pytest.fixture
def mock_job_repo() -> InMemoryJobRepo:
    """Return a fresh in-memory repo for each test."""
    return InMemoryJobRepo()


@pytest.fixture
def mock_task_queue() -> MagicMock:
    """A MagicMock satisfying ``services._TaskQueueProtocol``."""
    queue = MagicMock()
    queue.send = MagicMock()
    queue.enqueue_inference = MagicMock()
    return queue


@pytest.fixture
def mock_redis() -> MagicMock:
    """A MagicMock that quacks like a synchronous ``redis.Redis`` client."""
    client = MagicMock()
    client.ping.return_value = True
    client.get.return_value = None
    client.set.return_value = True
    return client


# ---------------------------------------------------------------------------
# Audio fixtures
# ---------------------------------------------------------------------------


def _make_wav_bytes(duration_seconds: float = 5.0, sample_rate: int = 16_000) -> bytes:
    """Build a valid 16-bit PCM mono WAV in memory using only the stdlib."""
    buffer = io.BytesIO()
    n_samples = int(duration_seconds * sample_rate)
    with wave.open(buffer, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)  # 16-bit PCM
        wav.setframerate(sample_rate)
        wav.writeframes(b"\x00\x00" * n_samples)
    return buffer.getvalue()


@pytest.fixture
def sample_wav_bytes() -> bytes:
    """5 seconds of silence, 16 kHz mono, 16-bit PCM (valid RIFF/WAVE)."""
    return _make_wav_bytes(5.0, 16_000)


@pytest.fixture
def fake_zip_bytes() -> bytes:
    """A minimal ZIP-magic header used to test ZIP-disguised-as-WAV rejection."""
    return b"PK\x03\x04" + b"\x00" * 64


@pytest.fixture
def oversized_wav_path(tmp_path: Path) -> Path:
    """A 101 MB RIFF blob written to a temp file.

    The audio cap is ``settings.audio_max_size_mb`` (100 MB by default), so
    this file is guaranteed to exceed it. Streamed to disk in 1 MiB chunks
    so we never hold the full payload in RAM at once.
    """
    target = tmp_path / "oversized.wav"
    one_mib = b"\x00" * (1024 * 1024)
    with target.open("wb") as f:
        f.write(b"RIFF")
        for _ in range(101):
            f.write(one_mib)
    return target


@pytest.fixture
def oversized_content_length() -> int:
    """``Content-Length`` value just above the 100 MB cap, for header tests."""
    return 101 * 1024 * 1024


# ---------------------------------------------------------------------------
# FastAPI test client with overridden dependencies
# ---------------------------------------------------------------------------


@pytest.fixture
def test_client(
    mock_job_repo: InMemoryJobRepo,
    mock_task_queue: MagicMock,
) -> Iterator:
    """A ``TestClient`` whose repository / service deps are in-memory fakes."""
    from fastapi.testclient import TestClient

    from api.dependencies import get_inference_service, get_job_repository
    from api.main import app
    from services.inference_service import InferenceService

    service = InferenceService(mock_job_repo, mock_task_queue)

    def _override_repo():
        yield mock_job_repo

    def _override_service():
        return service

    app.dependency_overrides[get_job_repository] = _override_repo
    app.dependency_overrides[get_inference_service] = _override_service

    with TestClient(app) as client:
        # Expose the in-memory state so tests can introspect / pre-seed.
        client.job_repo = mock_job_repo  # type: ignore[attr-defined]
        client.task_queue = mock_task_queue  # type: ignore[attr-defined]
        client.inference_service = service  # type: ignore[attr-defined]
        yield client

    app.dependency_overrides.clear()


# ---------------------------------------------------------------------------
# Skip marker: libmagic detection (Windows commonly lacks it)
# ---------------------------------------------------------------------------


def _libmagic_available() -> bool:
    try:
        import magic  # type: ignore[import-untyped]
        # python-magic on Windows requires the libmagic DLL; calling it once
        # surfaces ImportError-by-side-effect early.
        magic.from_buffer(b"\x00\x00\x00\x00", mime=True)
        return True
    except Exception:
        return False


_LIBMAGIC_AVAILABLE = _libmagic_available()


def pytest_configure(config: "pytest.Config") -> None:
    """Expose ``libmagic_available`` to ``skipif`` decorators."""
    config.addinivalue_line("markers", "needs_libmagic: requires libmagic on host")


@pytest.fixture(scope="session")
def libmagic_available() -> bool:
    return _LIBMAGIC_AVAILABLE
