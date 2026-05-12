"""
Async SQLAlchemy engine + session factory.
Provides get_db() dependency and raw async connection helper.
"""

from contextlib import asynccontextmanager
from typing import AsyncGenerator

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

_engine: AsyncEngine | None = None
_session_factory: async_sessionmaker | None = None


class Base(DeclarativeBase):
    """Base class for all SQLAlchemy ORM models."""
    pass


async def init_postgres() -> None:
    """Create engine and session factory. Called once at startup."""
    global _engine, _session_factory

    _engine = create_async_engine(
        settings.postgres_dsn,
        pool_size=settings.postgres_pool_size,
        max_overflow=settings.postgres_max_overflow,
        pool_pre_ping=True,          # Detect stale connections
        pool_recycle=3600,           # Recycle connections every 1 hour
        echo=settings.debug,         # Log SQL in debug mode only
        connect_args={
            "server_settings": {
                "application_name": settings.app_name,
            }
        },
    )

    _session_factory = async_sessionmaker(
        bind=_engine,
        class_=AsyncSession,
        expire_on_commit=False,      # Prevent lazy-load errors after commit
        autoflush=False,
    )

    # Verify connectivity
    async with _engine.connect() as conn:
        await conn.execute(text("SELECT 1"))

    logger.info(
        "postgres_initialized",
        host=settings.postgres_host,
        db=settings.postgres_db,
        pool_size=settings.postgres_pool_size,
    )


async def close_postgres() -> None:
    """Dispose engine. Called on shutdown."""
    global _engine
    if _engine:
        await _engine.dispose()
        _engine = None
        logger.info("postgres_pool_closed")


@asynccontextmanager
async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """
    Async context manager that provides a session and handles
    commit/rollback automatically.

    Usage:
        async with get_db() as db:
            result = await db.execute(...)
    """
    if _session_factory is None:
        raise RuntimeError("PostgreSQL not initialised — call init_postgres() first")

    session: AsyncSession = _session_factory()
    try:
        yield session
        await session.commit()
    except Exception:
        await session.rollback()
        raise
    finally:
        await session.close()


def get_engine() -> AsyncEngine:
    if _engine is None:
        raise RuntimeError("PostgreSQL not initialised")
    return _engine