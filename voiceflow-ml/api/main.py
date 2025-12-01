"""
FastAPI application entry point.
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import structlog

from core import settings, configure_logging, init_db, check_db_connection, close_redis
from api.routes import health, models as models_routes, inference

# Configure logging
configure_logging(debug=settings.debug)
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

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
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
