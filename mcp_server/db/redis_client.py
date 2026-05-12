"""
Async Redis connection pool — one pool for the entire process.
All modules import `get_redis()` to get the shared client.
"""

import redis.asyncio as aioredis
from redis.asyncio import ConnectionPool
from core.config import settings
from core.logging import get_logger

logger = get_logger(__name__)

_pool: ConnectionPool | None = None


async def init_redis_pool() -> None:
    """Create the connection pool. Called once during app startup."""
    global _pool
    _pool = ConnectionPool(
        host=settings.redis_host,
        port=settings.redis_port,
        db=settings.redis_db,
        password=settings.redis_password,
        max_connections=settings.redis_max_connections,
        decode_responses=True,          # Always work with strings, not bytes
        socket_connect_timeout=5,
        socket_timeout=5,
        retry_on_timeout=True,
        health_check_interval=30,
    )
    # Verify connectivity
    client = aioredis.Redis(connection_pool=_pool)
    await client.ping()
    logger.info("redis_pool_initialized", host=settings.redis_host, port=settings.redis_port)


async def close_redis_pool() -> None:
    """Gracefully close all connections. Called on shutdown."""
    global _pool
    if _pool:
        await _pool.aclose()
        _pool = None
        logger.info("redis_pool_closed")


def get_redis() -> aioredis.Redis:
    """Return a Redis client backed by the shared pool."""
    if _pool is None:
        raise RuntimeError("Redis pool not initialised — call init_redis_pool() first")
    return aioredis.Redis(connection_pool=_pool)