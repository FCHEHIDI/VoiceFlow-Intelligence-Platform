"""
FastAPI application entry point.
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import structlog

from api.middleware.input_validation import content_length_middleware
from core import settings, configure_logging, init_db, check_db_connection, close_redis
from api.routes import health, models as models_routes, inference

# Ensure ORM models are registered with `Base.metadata` before `init_db()` is
# called during the lifespan startup; otherwise tables would not be created.
from repositories import job_repository as _register_job_models  # noqa: F401

# Configure logging — JSON in production, console renderer otherwise.
configure_logging(env=settings.app_env, debug=settings.debug)
logger = structlog.get_logger()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan events."""
    # Startup
    logger.info("Starting VoiceFlow ML Service", version=settings.app_version)
    
    # Initialize database
    if check_db_connection():
        init_db()
        logger.info("Database initialized")
    else:
        logger.warning("Database connection failed")
    
    yield
    
    # Shutdown
    logger.info("Shutting down VoiceFlow ML Service")
    await close_redis()


# Create FastAPI app
app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    description="Speaker Diarization ML Training & Management Service",
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
    lifespan=lifespan
)

# CORS: explicit origins only; never use wildcard in production
_cors = settings.cors_allowed_origins
if settings.app_env == "production" and (not _cors or any(o.strip() == "*" for o in _cors)):
    raise RuntimeError("CORS: set CORS_ORIGINS in production to explicit origins (no wildcard).")

# Content-Length guard for large uploads (inference) — add before CORS (outer)
app.add_middleware(content_length_middleware())

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(health.router, tags=["Health"])
app.include_router(
    models_routes.router,
    prefix=f"{settings.api_v1_prefix}/models",
    tags=["Models"]
)
app.include_router(
    inference.router,
    prefix=f"{settings.api_v1_prefix}/inference",
    tags=["Inference"]
)

# AWS X-Ray distributed tracing — production only.
if settings.app_env == "production":
    try:
        from aws_xray_sdk.core import patch_all, xray_recorder
        from aws_xray_sdk.ext.fastapi.middleware import XRayMiddleware

        xray_recorder.configure(service="voiceflow-ml", sampling=True)
        patch_all()
        app.add_middleware(XRayMiddleware, recorder=xray_recorder)
        logger.info("xray_initialised", service="voiceflow-ml")
    except ImportError:  # pragma: no cover - optional in dev
        logger.warning("aws_xray_sdk not installed — distributed tracing disabled")


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "service": settings.app_name,
        "version": settings.app_version,
        "status": "running"
    }


if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "api.main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.debug,
        log_level="debug" if settings.debug else "info"
    )
