"""Integration tests for the inference / health HTTP surface.

These exercise the full FastAPI app via ``TestClient``; the database and
Celery dependencies are replaced by in-memory fakes provided by the
``test_client`` fixture (see ``conftest.py``).

The inference batch endpoint accepts a JSON body of the form
``{"audio_path": "...", "callback_url": "..."}`` — see
``api/routes/inference.py::BatchInferenceRequest``. The "WAV upload" half of
the request flow lives behind ``/api/inference/sync`` and is covered by the
audio-validation unit tests.
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.integration


# ---------------------------------------------------------------------------
# Health & root
# ---------------------------------------------------------------------------


class TestHealthEndpoints:
    def test_health_returns_200_with_service_and_status(self, test_client) -> None:
        resp = test_client.get("/health")
        assert resp.status_code == 200
        body = resp.json()
        assert "service" in body
        assert "status" in body
        assert body["status"] == "healthy"
        assert body["service"] == "voiceflow-ml"

    def test_root_returns_service_metadata(self, test_client) -> None:
        resp = test_client.get("/")
        assert resp.status_code == 200
        body = resp.json()
        assert body.get("status") == "running"


# ---------------------------------------------------------------------------
# /api/inference/batch — submit & lookup
# ---------------------------------------------------------------------------


class TestBatchInferenceEndpoint:
    def test_submit_returns_202_with_pending_job(self, test_client) -> None:
        payload = {
            "audio_path": "s3://bucket/clip.wav",
            "callback_url": "https://example.test/cb",
        }
        resp = test_client.post("/api/inference/batch", json=payload)
        assert resp.status_code == 202
        body = resp.json()
        assert body["status"] == "pending"
        assert body["audio_path"] == "s3://bucket/clip.wav"
        assert body["callback_url"] == "https://example.test/cb"
        assert body["job_id"]
        assert body["result"] is None
        assert body["error"] is None

        # Persisted in the in-memory repo & dispatched on the queue
        assert test_client.job_repo.get(body["job_id"]) is not None
        test_client.task_queue.enqueue_inference.assert_called_once_with(body["job_id"])

    def test_get_existing_job_returns_200(self, test_client) -> None:
        created = test_client.post(
            "/api/inference/batch", json={"audio_path": "s3://b/a.wav"}
        ).json()
        job_id = created["job_id"]

        resp = test_client.get(f"/api/inference/batch/{job_id}")
        assert resp.status_code == 200
        body = resp.json()
        assert body["job_id"] == job_id
        assert body["status"] == "pending"

    def test_get_unknown_job_returns_404(self, test_client) -> None:
        resp = test_client.get("/api/inference/batch/00000000-0000-0000-0000-000000000000")
        assert resp.status_code == 404
        body = resp.json()
        assert "detail" in body
        assert "00000000-0000-0000-0000-000000000000" in body["detail"]

    def test_missing_audio_path_returns_422(self, test_client) -> None:
        # No "audio_path" field in the JSON body -> Pydantic validation 422
        resp = test_client.post("/api/inference/batch", json={})
        assert resp.status_code == 422

    def test_empty_audio_path_returns_422(self, test_client) -> None:
        # Pydantic min_length=1 catches the empty string at the boundary.
        resp = test_client.post("/api/inference/batch", json={"audio_path": ""})
        assert resp.status_code == 422


# ---------------------------------------------------------------------------
# /api/inference/sync — audio upload validation surface
# ---------------------------------------------------------------------------


class TestSyncInferenceUpload:
    def test_sync_accepts_valid_wav_upload(
        self, test_client, sample_wav_bytes: bytes
    ) -> None:
        files = {"file": ("clip.wav", sample_wav_bytes, "audio/wav")}
        resp = test_client.post("/api/inference/sync", files=files)
        assert resp.status_code == 200
        # The route returns an empty list of segments for now (Rust does the work).
        assert resp.json() == []

    def test_sync_rejects_zip_disguised_as_wav(
        self, test_client, fake_zip_bytes: bytes
    ) -> None:
        files = {"file": ("clip.wav", fake_zip_bytes, "audio/wav")}
        resp = test_client.post("/api/inference/sync", files=files)
        assert resp.status_code == 400

    def test_sync_413_when_content_length_exceeds_cap(
        self, test_client, oversized_content_length: int
    ) -> None:
        # Drive the content_length_middleware via headers without ever
        # allocating a 100 MB buffer in the test process.
        resp = test_client.post(
            "/api/inference/sync",
            content=b"\x00\x00\x00\x00",
            headers={"content-length": str(oversized_content_length)},
        )
        assert resp.status_code == 413
