"""Health / readiness endpoints — drives ECS, Kubernetes and ALB probes."""
from __future__ import annotations

import asyncio
from typing import Any

import structlog
from fastapi import APIRouter
from fastapi.responses import JSONResponse

from core.config import settings
from core.database import check_db_connection
from core.redis_client import check_redis_connection

router = APIRouter()
logger = structlog.get_logger(__name__)


@router.get("/health")
async def health_check() -> dict[str, str]:
    """Liveness probe — returns 200 as long as the process is alive."""
    return {"status": "healthy", "service": "voiceflow-ml"}


async def _ping_rust(url: str | None) -> dict[str, Any]:
    if not url:
        return {"status": "skipped", "reason": "rust_service_url unset"}
    try:
        import httpx

        async with httpx.AsyncClient(timeout=httpx.Timeout(2.0)) as client:
            resp = await client.get(f"{url.rstrip('/')}/health")
            return {"status": "ok" if resp.status_code == 200 else "down", "code": resp.status_code}
    except Exception as exc:  # pragma: no cover - network specific
        return {"status": "down", "error": str(exc)}


@router.get("/ready")
async def readiness_check() -> JSONResponse:
    """Readiness probe — verifies DB, Redis and downstream Rust service."""
    db_ok = check_db_connection()
    redis_ok, rust_status = await asyncio.gather(
        check_redis_connection(),
        _ping_rust(getattr(settings, "rust_service_url", None)),
    )

    checks = {
        "database": {"status": "ok" if db_ok else "down"},
        "redis": {"status": "ok" if redis_ok else "down"},
        "rust_service": rust_status,
    }
    all_ok = all(c["status"] in {"ok", "skipped"} for c in checks.values())

    if not all_ok:
        logger.warning("service_not_ready", checks=checks)

    return JSONResponse(
        status_code=200 if all_ok else 503,
        content={"status": "ready" if all_ok else "degraded", "checks": checks},
    )
