"""
Async-friendly ChromaDB HTTP client.
ChromaDB's Python client is synchronous — we run it in a thread pool
to avoid blocking the event loop.
"""
from __future__ import annotations

import asyncio
from functools import partial
from typing import Any, Optional

import chromadb
from chromadb import AsyncHttpClient
from chromadb.config import Settings as ChromaSettings

from core.config import settings
from core.logging import get_logger

logger = get_logger(__name__)

_client: Optional[chromadb.AsyncHttpClient] = None


async def init_chromadb() -> None:
    """Create the ChromaDB async HTTP client. Called once at startup."""
    global _client

    _client = await chromadb.AsyncHttpClient(
        host=settings.chromadb_host,
        port=settings.chromadb_port,
        settings=ChromaSettings(anonymized_telemetry=False),
    )

    # Ensure required collections exist
    await _ensure_collection(settings.chromadb_collection_conversations)
    await _ensure_collection(settings.chromadb_collection_intents)

    logger.info(
        "chromadb_initialized",
        host=settings.chromadb_host,
        port=settings.chromadb_port,
    )


async def _ensure_collection(name: str) -> None:
    """Create collection if it doesn't already exist."""
    await _client.get_or_create_collection(
        name=name,
        metadata={"hnsw:space": "cosine"},     # Cosine distance for semantic search
    )
    logger.debug("chromadb_collection_ready", collection=name)


async def close_chromadb() -> None:
    global _client
    _client = None
    logger.info("chromadb_client_closed")


def get_chromadb() -> chromadb.AsyncHttpClient:
    if _client is None:
        raise RuntimeError("ChromaDB not initialised — call init_chromadb() first")
    return _client


async def get_collection(name: str):
    """Get a ChromaDB collection by name."""
    client = get_chromadb()
    return await client.get_collection(name=name)


async def upsert_embeddings(
    collection_name: str,
    ids: list[str],
    embeddings: list[list[float]],
    documents: list[str],
    metadatas: list[dict[str, Any]],
) -> None:
    """Upsert documents + embeddings into a collection."""
    collection = await get_collection(collection_name)
    await collection.upsert(
        ids=ids,
        embeddings=embeddings,
        documents=documents,
        metadatas=metadatas,
    )
    logger.debug(
        "chromadb_upsert",
        collection=collection_name,
        count=len(ids),
    )


async def query_collection(
    collection_name: str,
    query_embeddings: list[list[float]],
    n_results: int = 10,
    where: Optional[dict] = None,
) -> dict[str, Any]:
    """Semantic similarity search in a collection."""
    collection = await get_collection(collection_name)
    results = await collection.query(
        query_embeddings=query_embeddings,
        n_results=n_results,
        where=where,
        include=["documents", "metadatas", "distances"],
    )
    return results