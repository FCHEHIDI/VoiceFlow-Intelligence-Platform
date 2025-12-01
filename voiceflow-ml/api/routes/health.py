"""
Health check endpoints.
"""

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
import structlog

from core import get_db, check_db_connection
from core.redis_client import check_redis_connection

router = APIRouter()
logger = structlog.get_logger()


@router.get("/health")
async def health_check():
    """
    Basic health check endpoint.
    Returns 200 if service is running.
    """
    return {"status": "healthy"}


@router.get("/ready")
async def readiness_check():
    """
    Readiness probe - checks if service is ready to accept traffic.
    Validates database and Redis connections.
    """
    db_healthy = check_db_connection()
    redis_healthy = await check_redis_connection()
    
    is_ready = db_healthy and redis_healthy
    
    status = {
        "ready": is_ready,
        "checks": {
            "database": "up" if db_healthy else "down",
            "redis": "up" if redis_healthy else "down"
        }
    }
    
    if not is_ready:
        logger.warning("Service not ready", checks=status["checks"])
        return status, 503
    
    return status
