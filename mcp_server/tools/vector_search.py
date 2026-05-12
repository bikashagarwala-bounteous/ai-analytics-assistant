"""
MCP Tool: search_similar_conversations

Performs semantic similarity search over indexed chatbot conversations.
The LLM calls this tool when it needs to find conversations related to
a specific topic, pattern, or failure type.

Flow:
  query text → Gemini embedding → ChromaDB ANN search → ranked results
"""

import time
from datetime import datetime
from typing import Any

from db.chromadb_client import query_collection
from core.gemini_client import embed_query
from core.cache import rag_cache_key, cache_get, cache_set
from core.config import settings
from core.logging import get_logger
from schemas import (
    VectorSearchInput,
    VectorSearchOutput,
    ConversationMatch,
)

logger = get_logger(__name__)


async def search_similar_conversations(params: VectorSearchInput) -> VectorSearchOutput:
    """
    Find conversations semantically similar to the given query.

    The LLM uses this to:
    - Find examples of a specific failure pattern
    - Retrieve context for "why did X happen" questions
    - Identify conversations with similar user complaints
    """
    t0 = time.perf_counter()

    # ── Cache check ───────────────────────────────────────────────────────────
    cache_key = rag_cache_key(
        f"{params.query}:{params.limit}:{params.intent_filter}:{params.time_range}"
    )
    cached = await cache_get(cache_key)
    if cached:
        logger.debug("vector_search_cache_hit", query=params.query[:50])
        return VectorSearchOutput(**cached)

    # ── Embed the query ───────────────────────────────────────────────────────
    query_vector = await embed_query(params.query)

    # ── Build ChromaDB filters ────────────────────────────────────────────────
    where: dict[str, Any] | None = None
    conditions = []

    if params.intent_filter:
        conditions.append({"intent": {"$eq": params.intent_filter}})

    if len(conditions) == 1:
        where = conditions[0]
    elif len(conditions) > 1:
        where = {"$and": conditions}

    fetch_n = params.limit * 4 if params.time_range else params.limit
    raw_results = await query_collection(
        collection_name=settings.chromadb_collection_conversations,
        query_embeddings=[query_vector],
        n_results=fetch_n,
        where=where,
    )

    # ── Parse results ─────────────────────────────────────────────────────────
    matches: list[ConversationMatch] = []

    if raw_results and raw_results.get("ids"):
        ids = raw_results["ids"][0]
        documents = raw_results["documents"][0]
        metadatas = raw_results["metadatas"][0]
        distances = raw_results["distances"][0]

        for doc_id, doc, meta, dist in zip(ids, documents, metadatas, distances):
            similarity = 1.0 - (dist / 2.0)
            if similarity < params.min_similarity:
                continue

            raw_ts = meta.get("timestamp") or meta.get("timestamp_unix")
            try:
                ts = (datetime.fromtimestamp(float(raw_ts))
                      if isinstance(raw_ts, (int, float))
                      else datetime.fromisoformat(str(raw_ts)))
            except Exception:
                ts = datetime.now()

            if params.time_range and not (params.time_range.start <= ts <= params.time_range.end):
                continue

            if len(matches) >= params.limit:
                break

            matches.append(ConversationMatch(
                message_id=doc_id,
                session_id=meta.get("session_id", ""),
                content=doc,
                intent=meta.get("intent"),
                similarity=round(similarity, 4),
                timestamp=ts,
                metadata={k: v for k, v in meta.items()
                          if k not in ("session_id", "intent", "timestamp", "timestamp_unix")},
            ))

    # Sort by similarity descending
    matches.sort(key=lambda m: m.similarity, reverse=True)

    duration_ms = (time.perf_counter() - t0) * 1000

    output = VectorSearchOutput(
        query=params.query,
        matches=matches,
        total_found=len(matches),
        search_duration_ms=round(duration_ms, 2),
    )

    logger.info(
        "vector_search_complete",
        query=params.query[:60],
        matches=len(matches),
        duration_ms=round(duration_ms, 1),
    )

    # ── Cache the result ──────────────────────────────────────────────────────
    await cache_set(cache_key, output.model_dump(mode="json"), settings.cache_ttl_rag_retrieval)

    return output