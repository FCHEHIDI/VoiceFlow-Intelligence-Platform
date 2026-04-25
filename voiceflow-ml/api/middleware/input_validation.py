"""
Audio upload validation: size (413), magic bytes and MIME (400).
Uses python-magic when available; falls back to raw signature checks.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Awaitable, Callable

from fastapi import Request, Response, UploadFile, status
from fastapi import HTTPException
from starlette.middleware.base import BaseHTTPMiddleware

from core import settings

try:
    import magic  # type: ignore[import-untyped]
except ImportError:  # pragma: no cover
    magic = None  # type: ignore[assignment]


# Dangerous or non-audio families we reject if detected from magic or bytes
_REJECTED_MIME_PREFIXES = (
    "application/",
    "text/",
    "image/",
    "video/",
    "font/",
    "message/",
    "model/",
    "multipart/",
    "chemical/",
)
_ALLOWED_AUDIO_MIME_PREFIXES = ("audio/",)
_ALLOWED_MIME_STRICT = frozenset(
    {
        "audio/wav",
        "audio/wave",
        "audio/x-wav",
        "audio/mpeg",
        "audio/flac",
        "audio/ogg",
        "audio/vorbis",
        "application/ogg",  # some browsers
    }
)


@dataclass(frozen=True)
class InputValidationConfig:
    max_size_bytes: int
    target_path_prefix: str  # e.g. "/api/v1/inference" — must be prefix of batch/sync POST


def _mime_from_buffer(buf: bytes) -> str:
    if magic is not None:
        try:
            return str(magic.from_buffer(buf, mime=True))
        except Exception:
            pass
    return _mime_from_buffer_fallback(buf)


def _mime_from_buffer_fallback(buf: bytes) -> str:
    if len(buf) >= 4 and buf[0:4] == b"RIFF" and b"WAVE" in buf[:20]:
        return "audio/wav"
    if buf[0:4] == b"fLaC":
        return "audio/flac"
    if buf[0:4] == b"OggS":
        return "audio/ogg"
    if len(buf) >= 2 and (buf[0:2] in (b"\xff\xfb", b"\xff\xf3", b"\xff\xf2")):
        return "audio/mpeg"
    if len(buf) >= 4 and buf[0:4] == b"PK\x03\x04":
        return "application/zip"
    if buf[0:4] == b"%PDF":
        return "application/pdf"
    if len(buf) >= 2 and buf[0:2] == b"MZ":
        return "application/x-msdownload"
    return "application/octet-stream"


def _riff_wav_looks_valid(buf: bytes) -> bool:
    return len(buf) >= 12 and buf[0:4] == b"RIFF" and buf[8:12] == b"WAVE"


def _is_disallowed_mime(mime: str) -> bool:
    m = (mime or "").lower().strip()
    if not m:
        return True
    if m in _ALLOWED_MIME_STRICT:
        return False
    if m.startswith(_ALLOWED_AUDIO_MIME_PREFIXES):
        return False
    if m in ("application/ogg", "application/x-ogg", "application/octet-stream", "unknown"):
        return False
    if m.startswith(_REJECTED_MIME_PREFIXES):
        return True
    return not m.startswith("audio/")


def _refine_mime_by_signature(buf: bytes) -> str:
    """Map known good audio signatures to an audio/* MIME for validation."""
    if _riff_wav_looks_valid(buf):
        return "audio/wav"
    if len(buf) >= 4 and buf[0:4] == b"fLaC":
        return "audio/flac"
    if len(buf) >= 4 and buf[0:4] == b"OggS":
        return "audio/ogg"
    if len(buf) >= 2 and (buf[0:2] in (b"\xff\xfb", b"\xff\xf3", b"\xff\xf2")):
        return "audio/mpeg"
    return _mime_from_buffer(buf)


def _is_zip_pdf_exe(buf: bytes) -> bool:
    if len(buf) >= 4 and buf[0:4] == b"PK\x03\x04":
        return True
    if len(buf) >= 4 and buf[0:4] == b"%PDF":
        return True
    if len(buf) >= 2 and buf[0:2] == b"MZ":
        return True
    return False


async def validate_audio_upload(file: UploadFile) -> None:
    """
    Validate a single upload: content-type, magic bytes, no zip/pdf/exe, size <= max.
    Re-reads the file into memory up to max+1; resets stream to start on success.
    """
    max_bytes = settings.audio_max_size_mb * 1024 * 1024
    ct = (file.content_type or "").lower().strip()
    if ct and ct not in (
        "application/octet-stream",
        "binary/octet-stream",
        "application/ogg",
        "application/x-ogg",
    ):
        if not (ct.startswith("audio/") or "wav" in ct or "webm" in ct):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid or non-audio Content-Type. Upload an audio file (e.g. WAV, MP3, FLAC, OGG).",
            )

    first = await file.read(4096)
    if len(first) < 4:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="File too small to be a valid audio upload.",
        )

    if _is_zip_pdf_exe(first):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Refused: file signature looks like a ZIP, PDF, or executable, not audio.",
        )

    detected = _refine_mime_by_signature(first)
    if _is_disallowed_mime(detected):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid file content (detected: {detected}). Expected a supported audio format.",
        )
    if detected == "audio/wav" and not _riff_wav_looks_valid(first):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid WAV: expected RIFF/WAVE header.",
        )

    total = len(first)
    while total <= max_bytes:
        chunk = await file.read(65536)
        if not chunk:
            break
        total += len(chunk)
    if total > max_bytes:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"File exceeds maximum size of {settings.audio_max_size_mb} MB.",
        )
    # Reset for downstream handler
    await file.seek(0)


def build_content_length_middleware(
    config: InputValidationConfig,
) -> type[BaseHTTPMiddleware]:
    class _CLMiddleware(BaseHTTPMiddleware):
        async def dispatch(
            self, request: Request, call_next: Callable[[Request], Awaitable[Response]]
        ) -> Response:
            path = request.url.path
            if request.method == "POST" and path.startswith(config.target_path_prefix):
                cl = request.headers.get("content-length")
                if cl is not None:
                    try:
                        n = int(cl)
                    except ValueError:
                        return Response(status_code=400, content="Invalid Content-Length")
                    if n > config.max_size_bytes:
                        return Response(
                            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                            content=f"Request body exceeds {settings.audio_max_size_mb} MB",
                        )
            return await call_next(request)

    return _CLMiddleware


def content_length_middleware() -> type[BaseHTTPMiddleware]:
    p = f"{settings.api_v1_prefix}/inference"
    return build_content_length_middleware(
        InputValidationConfig(
            max_size_bytes=settings.audio_max_size_mb * 1024 * 1024,
            target_path_prefix=p,
        )
    )
