"""
Shared database connection pools for the backend.
Exposes init_* and close_* functions called at FastAPI startup/shutdown,
and get_* functions used everywhere else.
"""

from contextlib import asynccontextmanager
from typing import AsyncGenerator

import redis.asyncio as aioredis
from redis.asyncio import ConnectionPool
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.orm import DeclarativeBase
from sqlalchemy import text

from core.config import settings
from core.logging import get_logger

logger = get_logger(__name__)

# ── SQLAlchemy base ───────────────────────────────────────────────────────────

class Base(DeclarativeBase):
    pass


# ── PostgreSQL ────────────────────────────────────────────────────────────────

_pg_engine: AsyncEngine | None = None
_pg_session_factory: async_sessionmaker | None = None


async def init_postgres() -> None:
    global _pg_engine, _pg_session_factory
    _pg_engine = create_async_engine(
        settings.postgres_dsn,
        pool_size=settings.postgres_pool_size,
        max_overflow=settings.postgres_max_overflow,
        pool_pre_ping=True,
        pool_recycle=3600,
        echo=settings.debug,
    )
    _pg_session_factory = async_sessionmaker(
        bind=_pg_engine,
        class_=AsyncSession,
        expire_on_commit=False,
        autoflush=False,
    )
    async with _pg_engine.connect() as conn:
        await conn.execute(text("SELECT 1"))
    logger.info("postgres_connected", host=settings.postgres_host, db=settings.postgres_db)


async def close_postgres() -> None:
    global _pg_engine
    if _pg_engine:
        await _pg_engine.dispose()
        _pg_engine = None
        logger.info("postgres_closed")


@asynccontextmanager
async def get_db() -> AsyncGenerator[AsyncSession, None]:
    if _pg_session_factory is None:
        raise RuntimeError("PostgreSQL not initialised")
    session: AsyncSession = _pg_session_factory()
    try:
        yield session
        await session.commit()
    except Exception:
        await session.rollback()
        raise
    finally:
        await session.close()


def get_engine() -> AsyncEngine:
    if _pg_engine is None:
        raise RuntimeError("PostgreSQL not initialised")
    return _pg_engine


# ── Redis ─────────────────────────────────────────────────────────────────────

_redis_pool: ConnectionPool | None = None


async def init_redis() -> None:
    global _redis_pool
    _redis_pool = ConnectionPool(
        host=settings.redis_host,
        port=settings.redis_port,
        db=settings.redis_db,
        password=settings.redis_password,
        max_connections=settings.redis_max_connections,
        decode_responses=True,
        socket_connect_timeout=5,
        socket_timeout=5,
        retry_on_timeout=True,
        health_check_interval=30,
    )
    client = aioredis.Redis(connection_pool=_redis_pool)
    await client.ping()
    logger.info("redis_connected", host=settings.redis_host)


async def close_redis() -> None:
    global _redis_pool
    if _redis_pool:
        await _redis_pool.aclose()
        _redis_pool = None
        logger.info("redis_closed")


def get_redis() -> aioredis.Redis:
    if _redis_pool is None:
        raise RuntimeError("Redis not initialised")
    return aioredis.Redis(connection_pool=_redis_pool)