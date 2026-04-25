"""Unit tests for the audio upload validation middleware.

The validation pipeline has two halves:

1. ``validate_audio_upload`` — async per-upload check (content-type, magic
   bytes, no zip/pdf/exe, total size <= max). Returns ``None`` on success;
   raises ``HTTPException`` otherwise.
2. ``content_length_middleware`` / ``build_content_length_middleware`` — an
   ASGI middleware that 413s requests whose ``Content-Length`` header
   exceeds the cap, before the body is even read.

These tests exercise both, while staying lightweight on memory by relying
on header-driven 413 paths instead of materialising a 100 MB body.
"""

from __future__ import annotations

import asyncio
from io import BytesIO
from unittest.mock import AsyncMock

import pytest
from fastapi import HTTPException, UploadFile
from starlette.datastructures import Headers
from starlette.requests import Request

from api.middleware.input_validation import (
    InputValidationConfig,
    build_content_length_middleware,
    content_length_middleware,
    validate_audio_upload,
)
from core.config import settings

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_upload(payload: bytes, *, content_type: str = "audio/wav", filename: str = "x.wav") -> UploadFile:
    headers = Headers({"content-type": content_type})
    return UploadFile(file=BytesIO(payload), filename=filename, headers=headers)


def _run(coro):
    return asyncio.run(coro)


# ---------------------------------------------------------------------------
# validate_audio_upload — async per-upload check
# ---------------------------------------------------------------------------


class TestValidateAudioUpload:
    def test_accepts_valid_wav(self, sample_wav_bytes: bytes) -> None:
        upload = _make_upload(sample_wav_bytes, content_type="audio/wav")
        # Returns None on success.
        result = _run(validate_audio_upload(upload))
        assert result is None

    def test_accepts_valid_wav_with_octet_stream_content_type(
        self, sample_wav_bytes: bytes
    ) -> None:
        # Browsers / curl sometimes send application/octet-stream for files;
        # the validator should fall back to magic-byte detection.
        upload = _make_upload(
            sample_wav_bytes, content_type="application/octet-stream", filename="x.wav"
        )
        assert _run(validate_audio_upload(upload)) is None

    def test_rejects_zip_disguised_as_wav(self, fake_zip_bytes: bytes) -> None:
        upload = _make_upload(fake_zip_bytes, content_type="audio/wav", filename="x.wav")
        with pytest.raises(HTTPException) as exc_info:
            _run(validate_audio_upload(upload))
        assert exc_info.value.status_code == 400
        assert "ZIP" in exc_info.value.detail or "PDF" in exc_info.value.detail

    def test_rejects_too_small_file(self) -> None:
        # < 4 bytes is rejected upfront.
        upload = _make_upload(b"AB", content_type="audio/wav")
        with pytest.raises(HTTPException) as exc_info:
            _run(validate_audio_upload(upload))
        assert exc_info.value.status_code == 400
        assert "too small" in exc_info.value.detail.lower()

    def test_rejects_non_audio_content_type(self, sample_wav_bytes: bytes) -> None:
        upload = _make_upload(sample_wav_bytes, content_type="text/plain")
        with pytest.raises(HTTPException) as exc_info:
            _run(validate_audio_upload(upload))
        assert exc_info.value.status_code == 400

    def test_rejects_oversized_payload(self, sample_wav_bytes: bytes) -> None:
        # Avoid allocating 100 MB in the test process by monkey-patching the
        # cap down to ``0 MB``. Any valid WAV body then trips the size guard.
        original = settings.audio_max_size_mb
        settings.audio_max_size_mb = 0
        try:
            upload = _make_upload(sample_wav_bytes, content_type="audio/wav")
            with pytest.raises(HTTPException) as exc_info:
                _run(validate_audio_upload(upload))
            assert exc_info.value.status_code == 413
        finally:
            settings.audio_max_size_mb = original


# ---------------------------------------------------------------------------
# content_length_middleware — ASGI guard
# ---------------------------------------------------------------------------


def _build_middleware(max_bytes: int):
    cls = build_content_length_middleware(
        InputValidationConfig(
            max_size_bytes=max_bytes,
            target_path_prefix="/api/inference",
        )
    )
    return cls(app=AsyncMock())


def _make_request(method: str, path: str, *, content_length: str | None = None) -> Request:
    headers = []
    if content_length is not None:
        headers.append((b"content-length", content_length.encode("ascii")))
    scope = {
        "type": "http",
        "method": method,
        "path": path,
        "headers": headers,
        "query_string": b"",
        "scheme": "http",
        "server": ("testserver", 80),
        "client": ("test", 0),
        "root_path": "",
    }

    async def _receive():  # pragma: no cover - unused
        return {"type": "http.disconnect"}

    return Request(scope, _receive)


class TestContentLengthMiddleware:
    def test_413_when_content_length_exceeds_cap(
        self, oversized_content_length: int
    ) -> None:
        mw = _build_middleware(max_bytes=100 * 1024 * 1024)
        request = _make_request(
            "POST",
            "/api/inference/batch",
            content_length=str(oversized_content_length),
        )

        async def call_next(_req):  # pragma: no cover - should not run
            raise AssertionError("downstream handler must not be invoked when 413")

        response = _run(mw.dispatch(request, call_next))
        assert response.status_code == 413

    def test_passes_through_when_within_limit(self) -> None:
        mw = _build_middleware(max_bytes=100 * 1024 * 1024)
        request = _make_request(
            "POST", "/api/inference/batch", content_length="1024"
        )
        sentinel = object()

        async def call_next(_req):
            return sentinel

        result = _run(mw.dispatch(request, call_next))
        assert result is sentinel

    def test_passes_through_when_path_outside_prefix(
        self, oversized_content_length: int
    ) -> None:
        mw = _build_middleware(max_bytes=100 * 1024 * 1024)
        request = _make_request(
            "POST", "/api/models/upload", content_length=str(oversized_content_length)
        )
        sentinel = object()

        async def call_next(_req):
            return sentinel

        # Even for an oversized body, paths outside the inference prefix are
        # let through (they have their own validation rules).
        result = _run(mw.dispatch(request, call_next))
        assert result is sentinel

    def test_invalid_content_length_returns_400(self) -> None:
        mw = _build_middleware(max_bytes=100 * 1024 * 1024)
        request = _make_request(
            "POST", "/api/inference/batch", content_length="not-a-number"
        )

        async def call_next(_req):  # pragma: no cover
            raise AssertionError("downstream handler must not be invoked on 400")

        response = _run(mw.dispatch(request, call_next))
        assert response.status_code == 400

    def test_factory_yields_class_with_default_settings(self) -> None:
        cls = content_length_middleware()
        assert isinstance(cls, type)
