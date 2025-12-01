"""
Redis client for caching and rate limiting.
"""

import redis.asyncio as aioredis
from redis.asyncio import Redis
from typing import Optional
import structlog

from core.config import settings

logger = structlog.get_logger()

# Global Redis client instance
_redis_client: Optional[Redis] = None


async def get_redis() -> Redis:
    """Get Redis client instance."""
    global _redis_client
    
    if _redis_client is None:
        _redis_client = await aioredis.from_url(
            settings.redis_url,
            encoding="utf-8",
            decode_responses=True,
            max_connections=50
        )
        logger.info("Redis client initialized", url=settings.redis_url)
    
    return _redis_client


async def close_redis() -> None:
    """Close Redis connection."""
    global _redis_client
    
    if _redis_client:
        await _redis_client.close()
        _redis_client = None
        logger.info("Redis connection closed")


async def check_redis_connection() -> bool:
    """Check if Redis connection is alive."""
    try:
        redis = await get_redis()
        await redis.ping()
        return True
    except Exception as e:
        logger.error("redis_connection_failed", error=str(e))
        return False
