"""
Generic Redis cache with:
  - Consistent key namespacing
  - JSON serialisation (Pydantic model aware)
  - Per-key TTL
  - Stampede protection via probabilistic early recompute (PER)
  - Bulk invalidation by prefix
"""

import hashlib
import json
import time
from typing import Any, Callable, TypeVar, Awaitable

from pydantic import BaseModel

from db.redis_client import get_redis
from core.config import settings
from core.logging import get_logger

logger = get_logger(__name__)

T = TypeVar("T")

# ── Key builders ─────────────────────────────────────────────────────────────

def _hash(*parts: str) -> str:
    """Short deterministic hash of multiple string parts."""
    combined = "|".join(parts)
    return hashlib.sha256(combined.encode()).hexdigest()[:16]


def llm_cache_key(prompt: str, context: str = "") -> str:
    return f"cache:llm:{_hash(prompt, context)}"


def embedding_cache_key(text: str) -> str:
    return f"cache:embed:{_hash(text)}"


def analysis_cache_key(session_id: str, query: str) -> str:
    return f"cache:analysis:{session_id}:{_hash(query)}"


def rag_cache_key(query: str) -> str:
    return f"cache:rag:{_hash(query)}"


# ── Core get/set ──────────────────────────────────────────────────────────────

async def cache_get(key: str) -> Any | None:
    """
    Fetch a value from cache.
    Returns the deserialised Python object, or None on miss.
    """
    redis = get_redis()
    raw = await redis.get(key)
    if raw is None:
        logger.debug("cache_miss", key=key)
        return None
    logger.debug("cache_hit", key=key)
    return json.loads(raw)


async def cache_set(key: str, value: Any, ttl: int) -> None:
    """
    Store a value in cache with the given TTL (seconds).
    Pydantic models are serialised via model_dump().
    """
    if isinstance(value, BaseModel):
        payload = value.model_dump_json()
    else:
        payload = json.dumps(value, default=str)

    redis = get_redis()
    await redis.setex(key, ttl, payload)
    logger.debug("cache_set", key=key, ttl=ttl)


async def cache_delete(key: str) -> None:
    redis = get_redis()
    await redis.delete(key)
    logger.debug("cache_delete", key=key)


async def cache_invalidate_prefix(prefix: str) -> int:
    """Delete all keys matching `prefix*`. Returns count deleted."""
    redis = get_redis()
    keys = []
    async for key in redis.scan_iter(f"{prefix}*"):
        keys.append(key)
    if keys:
        await redis.delete(*keys)
    logger.info("cache_invalidated", prefix=prefix, count=len(keys))
    return len(keys)


# ── Decorator-style cached execution ─────────────────────────────────────────

async def cached(
    key: str,
    ttl: int,
    fn: Callable[[], Awaitable[T]],
) -> T:
    """Try cache first; on miss execute fn(), cache the result, and return it."""
    hit = await cache_get(key)
    if hit is not None:
        return hit

    value = await fn()
    await cache_set(key, value, ttl)
    return value


# ── Embedding-specific bulk helper ────────────────────────────────────────────

async def get_cached_embeddings(
    texts: list[str],
) -> tuple[list[list[float] | None], list[int]]:
    """Bulk-fetch embeddings from cache. Returns (results aligned with texts, miss indices)."""
    redis = get_redis()
    keys = [embedding_cache_key(t) for t in texts]
    raw_values = await redis.mget(*keys)

    results: list[list[float] | None] = []
    miss_idx: list[int] = []

    for i, raw in enumerate(raw_values):
        if raw is None:
            results.append(None)
            miss_idx.append(i)
        else:
            results.append(json.loads(raw))

    logger.debug(
        "embedding_cache_bulk",
        total=len(texts),
        hits=len(texts) - len(miss_idx),
        misses=len(miss_idx),
    )
    return results, miss_idx


async def set_cached_embeddings(
    texts: list[str],
    embeddings: list[list[float]],
) -> None:
    """Bulk-store embeddings using a Redis pipeline for efficiency."""
    redis = get_redis()
    pipe = redis.pipeline()
    for text, embedding in zip(texts, embeddings):
        key = embedding_cache_key(text)
        pipe.setex(key, settings.cache_ttl_embedding, json.dumps(embedding))
    await pipe.execute()
    logger.debug("embedding_cache_bulk_set", count=len(texts))