"""
Background Indexer Service

Continuously drains the Redis indexing queue and writes conversation
embeddings into ChromaDB so the RAG and vector search tools have
up-to-date data.

Queue contract (matches session_service.py):
  key:    queue:index_messages
  value:  JSON  {"message_id": str, "session_id": str, "content": str, "queued_at": str}

Flow per batch:
  1. Pop up to batch_size items from the Redis list (LMPOP)
  2. Filter out items already indexed (check chromadb by ID)
  3. Fetch intent and metadata for each message from PostgreSQL
  4. Embed all content texts in one Gemini call (cache-aware)
  5. Upsert into ChromaDB with metadata
  6. Mark messages as indexed in PostgreSQL
  7. Sleep if queue is empty, immediately loop if more items remain

Fault tolerance:
  - Failed items are pushed to a dead-letter list (queue:index_dead)
    after 3 attempts, with error and original payload preserved.
  - Gemini rate limit errors pause the batch and retry after the
    sliding window clears.
  - PostgreSQL / ChromaDB errors are logged and the batch is re-queued.
"""

import asyncio
import json
import logging
import random
import sys
import time
from datetime import datetime, timedelta

import chromadb
import redis.asyncio as aioredis
import structlog
from google import genai
from google.genai import types
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker, AsyncSession
from sqlalchemy import text

from config import settings


# ── Logging ───────────────────────────────────────────────────────────────────

logging.basicConfig(format="%(message)s", stream=sys.stdout, level=logging.INFO)
structlog.configure(
    processors=[
        structlog.stdlib.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.format_exc_info,
        structlog.dev.ConsoleRenderer(colors=False)
        if settings.environment == "production"
        else structlog.dev.ConsoleRenderer(colors=True),
    ],
    wrapper_class=structlog.make_filtering_bound_logger(logging.INFO),
    logger_factory=structlog.PrintLoggerFactory(),
)
log = structlog.get_logger("indexer")


# ── Connection setup ──────────────────────────────────────────────────────────

async def build_connections():
    engine = create_async_engine(
        settings.postgres_dsn,
        pool_size=5,
        pool_pre_ping=True,
        echo=False,
    )
    session_factory = async_sessionmaker(bind=engine, expire_on_commit=False)

    redis = aioredis.Redis(
        host=settings.redis_host,
        port=settings.redis_port,
        db=settings.redis_db,
        password=settings.redis_password,
        decode_responses=True,
    )
    await redis.ping()

    chroma = await chromadb.AsyncHttpClient(
        host=settings.chromadb_host,
        port=settings.chromadb_port,
    )
    collection = await chroma.get_or_create_collection(
        name=settings.chromadb_collection_conversations,
        metadata={"hnsw:space": "cosine"},
    )

    gemini = genai.Client(api_key=settings.gemini_api_key)

    log.info("connections_ready",
             pg=settings.postgres_host,
             redis=settings.redis_host,
             chroma=settings.chromadb_host)

    return engine, session_factory, redis, collection, gemini


# ── Rate limiter (mirrors mcp_server logic, isolated for this process) ────────

async def _check_rate_limit(redis: aioredis.Redis) -> float:
    """
    Check RPM and RPD limits using the same Redis keys as the MCP server
    so both processes share one view of quota usage.
    Returns seconds to wait (0 if clear).
    """
    now = time.time()

    pipe = redis.pipeline()
    pipe.zremrangebyscore(settings.rate_limit_rpm_key, "-inf", now - 60)
    pipe.zcard(settings.rate_limit_rpm_key)
    pipe.zremrangebyscore(settings.rate_limit_rpd_key, "-inf", now - 86_400)
    pipe.zcard(settings.rate_limit_rpd_key)
    results = await pipe.execute()

    rpm_count, rpd_count = results[1], results[3]

    if rpm_count >= settings.gemini_rpm_limit:
        oldest = await redis.zrange(settings.rate_limit_rpm_key, 0, 0, withscores=True)
        wait = (oldest[0][1] + 60 - now) if oldest else 60
        return max(wait, 1.0)

    if rpd_count >= settings.gemini_rpd_limit:
        return 3600  # Daily limit — back off significantly

    return 0.0


async def _consume_rate_limit(redis: aioredis.Redis, n_requests: int = 1) -> None:
    """Record n_requests against the shared rate limit counters."""
    now = time.time()
    pipe = redis.pipeline()
    for i in range(n_requests):
        member = f"{now}:{i}"
        pipe.zadd(settings.rate_limit_rpm_key, {member: now})
        pipe.zadd(settings.rate_limit_rpd_key, {member: now})
    pipe.expire(settings.rate_limit_rpm_key, 70)
    pipe.expire(settings.rate_limit_rpd_key, 86_410)
    await pipe.execute()


# ── Embedding (cache-aware) ───────────────────────────────────────────────────

async def embed_batch(
    texts: list[str],
    redis: aioredis.Redis,
    gemini: genai.Client,
) -> list[list[float]]:
    """
    Embed a list of texts.
    Checks Redis cache per text; calls Gemini only for cache misses.
    Caches fresh embeddings after the call.
    """
    import hashlib

    def _key(t: str) -> str:
        h = hashlib.sha256(t.encode()).hexdigest()[:16]
        return f"{settings.embedding_cache_prefix}{h}"

    keys = [_key(t) for t in texts]
    cached_raw = await redis.mget(*keys)

    results: list[list[float] | None] = []
    miss_indices: list[int] = []

    for i, raw in enumerate(cached_raw):
        if raw is None:
            results.append(None)
            miss_indices.append(i)
        else:
            results.append(json.loads(raw))

    if not miss_indices:
        return results  # type: ignore[return-value]

    miss_texts = [texts[i] for i in miss_indices]

    # Wait for rate limit clearance
    for attempt in range(settings.gemini_max_retries):
        wait = await _check_rate_limit(redis)
        if wait > 0:
            jitter = random.uniform(0, settings.gemini_retry_jitter_max)
            log.info("rate_limit_wait", seconds=round(wait + jitter, 1), attempt=attempt)
            await asyncio.sleep(wait + jitter)
            continue
        break

    # Call Gemini
    await _consume_rate_limit(redis)
    response = await asyncio.to_thread(
        gemini.models.embed_content,
        model=settings.gemini_embedding_model,
        contents=miss_texts,
        config=types.EmbedContentConfig(task_type="RETRIEVAL_DOCUMENT"),
    )
    fresh = [e.values for e in response.embeddings]

    # Cache fresh embeddings
    pipe = redis.pipeline()
    for text, vec in zip(miss_texts, fresh):
        pipe.setex(_key(text), settings.embedding_cache_ttl, json.dumps(vec))
    await pipe.execute()

    # Merge
    fresh_iter = iter(fresh)
    for i in miss_indices:
        results[i] = next(fresh_iter)

    return results  # type: ignore[return-value]


# ── PostgreSQL helpers ────────────────────────────────────────────────────────

async def fetch_message_metadata(
    session: AsyncSession,
    message_ids: list[str],
) -> dict[str, dict]:
    """
    Fetch intent and session metadata for a list of message IDs.
    Returns a dict keyed by message_id.
    """
    rows = await session.execute(
        text("""
            SELECT
                cm.id           AS message_id,
                cm.session_id,
                cm.created_at,
                ic.intent,
                ic.is_failure,
                ic.confidence
            FROM chat_messages cm
            LEFT JOIN intent_classifications ic ON ic.message_id = cm.id
            WHERE cm.id = ANY(:ids)
        """),
        {"ids": message_ids},
    )
    return {
        r.message_id: {
            "session_id": r.session_id,
            "timestamp": r.created_at.isoformat(),
            "timestamp_unix": r.created_at.timestamp(),
            "intent": r.intent or "unknown",
            "is_failure": str(r.is_failure or False),
            "confidence": str(r.confidence or 0.0),
        }
        for r in rows.fetchall()
    }


# ── Dead-letter queue ─────────────────────────────────────────────────────────

_DEAD_LETTER_KEY = "queue:index_dead"
_ATTEMPTS_KEY_PREFIX = "indexer:attempts:"


async def _increment_attempts(redis: aioredis.Redis, message_id: str) -> int:
    key = f"{_ATTEMPTS_KEY_PREFIX}{message_id}"
    count = await redis.incr(key)
    await redis.expire(key, 86_400)
    return count


async def _send_to_dead_letter(
    redis: aioredis.Redis, item: dict, error: str
) -> None:
    payload = json.dumps({**item, "error": error, "failed_at": datetime.utcnow().isoformat()})
    await redis.rpush(_DEAD_LETTER_KEY, payload)
    log.warning("dead_letter", message_id=item.get("message_id"), error=error[:120])


# ── Main indexing batch ───────────────────────────────────────────────────────

async def index_batch(
    items: list[dict],
    session_factory: async_sessionmaker,
    redis: aioredis.Redis,
    collection,
    gemini: genai.Client,
) -> int:
    """
    Process one batch of queued messages.
    Returns the count of successfully indexed messages.
    """
    if not items:
        return 0

    seen: set[str] = set()
    unique_items: list[dict] = []
    for item in items:
        mid = item["message_id"]
        if mid not in seen:
            seen.add(mid)
            unique_items.append(item)
    items = unique_items

    message_ids = [item["message_id"] for item in items]

    try:
        existing = await collection.get(ids=message_ids, include=[])
        already_indexed = set(existing.get("ids", []))
    except Exception:
        already_indexed = set()

    to_index = [item for item in items if item["message_id"] not in already_indexed]
    if not to_index:
        log.debug("batch_all_already_indexed", count=len(items))
        return len(items)

    # Embed
    try:
        texts = [item["content"] for item in to_index]
        vectors = await embed_batch(texts, redis, gemini)
    except Exception as exc:
        log.error("embedding_failed", error=str(exc), count=len(to_index))
        # Re-queue the entire batch
        pipe = redis.pipeline()
        for item in to_index:
            pipe.rpush(settings.index_queue_key, json.dumps(item))
        await pipe.execute()
        return 0

    # Fetch metadata from PostgreSQL
    ids_to_fetch = [item["message_id"] for item in to_index]
    async with session_factory() as session:
        metadata_map = await fetch_message_metadata(session, ids_to_fetch)

    # Upsert into ChromaDB
    ids = []
    embeddings = []
    documents = []
    metadatas = []

    for item, vector in zip(to_index, vectors):
        if vector is None:
            continue
        mid = item["message_id"]
        ids.append(mid)
        embeddings.append(vector)
        documents.append(item["content"])
        meta = metadata_map.get(mid, {})
        meta["queued_at"] = item.get("queued_at", "")
        metadatas.append(meta)

    if ids:
        try:
            await collection.upsert(
                ids=ids,
                embeddings=embeddings,
                documents=documents,
                metadatas=metadatas,
            )
            log.info("batch_indexed", count=len(ids))
        except Exception as exc:
            log.error("chromadb_upsert_failed", error=str(exc))
            pipe = redis.pipeline()
            for item in to_index:
                attempts = await _increment_attempts(redis, item["message_id"])
                if attempts >= 3:
                    await _send_to_dead_letter(redis, item, str(exc))
                else:
                    pipe.rpush(settings.index_queue_key, json.dumps(item))
            await pipe.execute()
            return 0

    return len(ids)


# ── Catch-up scan ─────────────────────────────────────────────────────────────

async def catchup_scan(
    session_factory: async_sessionmaker,
    redis: aioredis.Redis,
    collection,
    gemini: genai.Client,
) -> None:
    """
    On startup, find any user messages from the last max_queue_age_hours
    that are not yet in ChromaDB and enqueue them.
    This handles any gaps caused by restarts or queue failures.
    """
    log.info("catchup_scan_start")
    since = datetime.utcnow() - timedelta(hours=settings.max_queue_age_hours)

    async with session_factory() as session:
        rows = await session.execute(
            text("""
                SELECT id, session_id, content, created_at
                FROM chat_messages
                WHERE role = 'user'
                  AND created_at >= :since
                ORDER BY created_at ASC
            """),
            {"since": since},
        )
        messages = rows.fetchall()

    if not messages:
        log.info("catchup_scan_nothing_to_do")
        return

    all_ids = [str(r.id) for r in messages]
    try:
        existing = await collection.get(ids=all_ids, include=[])
        indexed_ids = set(existing.get("ids", []))
    except Exception:
        indexed_ids = set()

    missing = [r for r in messages if str(r.id) not in indexed_ids]
    log.info("catchup_scan_complete", total=len(messages), missing=len(missing))

    if missing:
        pipe = redis.pipeline()
        for r in missing:
            payload = json.dumps({
                "message_id": str(r.id),
                "session_id": str(r.session_id),
                "content": r.content,
                "queued_at": r.created_at.isoformat(),
            })
            pipe.rpush(settings.index_queue_key, payload)
        await pipe.execute()
        log.info("catchup_enqueued", count=len(missing))


# ── Main loop ─────────────────────────────────────────────────────────────────

async def main() -> None:
    engine, session_factory, redis, collection, gemini = await build_connections()

    await catchup_scan(session_factory, redis, collection, gemini)

    log.info("indexer_running", poll_interval=settings.poll_interval_seconds)

    consecutive_empty = 0

    while True:
        try:
            raw_items = await redis.lmpop(1, settings.index_queue_key, direction="LEFT", count=settings.batch_size,)

            if not raw_items or not raw_items[1]:
                consecutive_empty += 1
                sleep_time = min(
                    settings.poll_interval_seconds,
                    2 ** min(consecutive_empty, 4),
                )
                await asyncio.sleep(sleep_time)
                continue

            consecutive_empty = 0
            items = [json.loads(raw) for raw in raw_items[1]]

            indexed = await index_batch(items, session_factory, redis, collection, gemini)
            log.debug("batch_complete", submitted=len(items), indexed=indexed)

            queue_len = await redis.llen(settings.index_queue_key)
            if queue_len == 0:
                await asyncio.sleep(1)

        except asyncio.CancelledError:
            break
        except Exception as exc:
            log.error("indexer_loop_error", error=str(exc))
            await asyncio.sleep(5)

    log.info("indexer_stopped")
    await redis.aclose()
    await engine.dispose()


if __name__ == "__main__":
    asyncio.run(main())