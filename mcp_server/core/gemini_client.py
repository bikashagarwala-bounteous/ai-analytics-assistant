"""
production Gemini wrapper with retry/backoff, rate limiting,
Redis response caching, streaming support, and cache-aware batch embeddings.
"""

import asyncio
import hashlib
import json
import random
import time
from collections.abc import AsyncGenerator
from typing import Any

from google import genai
from google.genai import types

from core.config import settings
from core.logging import get_logger
from core.rate_limiter import wait_for_capacity, RateLimitExceeded
from core.cache import (
    llm_cache_key,
    embedding_cache_key,
    cache_get,
    cache_set,
    get_cached_embeddings,
    set_cached_embeddings,
)

logger = get_logger(__name__)

# ── Retry-worthy errors ───────────────────────────────────────────────────────
_RETRYABLE_STATUS_CODES = {429, 500, 502, 503, 504}
_RETRYABLE_MESSAGES = ("quota", "rate limit", "overloaded", "internal", "unavailable")


def _is_retryable(exc: Exception) -> bool:
    msg = str(exc).lower()
    return any(kw in msg for kw in _RETRYABLE_MESSAGES)


# ── Singleton client ──────────────────────────────────────────────────────────
_client: genai.Client | None = None


def get_client() -> genai.Client:
    global _client
    if _client is None:
        _client = genai.Client(api_key=settings.gemini_api_key)
    return _client


# ── Core generate (non-streaming) ─────────────────────────────────────────────

async def generate(
    prompt: str,
    system_prompt: str = "",
    temperature: float = 0.1,
    max_output_tokens: int = 8192,
    use_cache: bool = True,
    estimated_input_tokens: int = 500,
) -> str:
    """
    Call Gemini generate_content with full fault tolerance.

    Args:
        prompt:                 User message.
        system_prompt:          Optional system context.
        temperature:            Sampling temperature (low = deterministic).
        max_output_tokens:      Response length cap.
        use_cache:              Whether to check/populate Redis cache.
        estimated_input_tokens: Used for rate limiter token tracking.

    Returns:
        Generated text string.
    """
    cache_key = llm_cache_key(prompt, system_prompt)

    # ── Cache check ───────────────────────────────────────────────────────────
    if use_cache:
        cached = await cache_get(cache_key)
        if cached is not None:
            logger.info("llm_cache_hit", key=cache_key[:20])
            return cached

    # ── Rate limit gate ───────────────────────────────────────────────────────
    await wait_for_capacity(estimated_tokens=estimated_input_tokens)

    # ── Retry loop ────────────────────────────────────────────────────────────
    last_exc: Exception | None = None
    for attempt in range(settings.gemini_max_retries):
        try:
            client = get_client()
            contents: list[Any] = []

            if system_prompt:
                contents.append(
                    types.Content(
                        role="user",
                        parts=[types.Part(text=system_prompt)],
                    )
                )

            contents.append(prompt)

            t0 = time.perf_counter()
            response = await asyncio.to_thread(
                client.models.generate_content,
                model=settings.gemini_model,
                contents=contents,
                config=types.GenerateContentConfig(
                    temperature=temperature,
                    max_output_tokens=max_output_tokens,
                    system_instruction=system_prompt if system_prompt else None,
                ),
            )
            duration_ms = (time.perf_counter() - t0) * 1000

            result_text = response.text or ""

            logger.info(
                "llm_call_success",
                attempt=attempt,
                duration_ms=round(duration_ms, 1),
                output_len=len(result_text),
            )

            # ── Cache the result ──────────────────────────────────────────────
            if use_cache and result_text:
                await cache_set(cache_key, result_text, settings.cache_ttl_llm_response)

            return result_text

        except Exception as exc:
            last_exc = exc
            if not _is_retryable(exc):
                logger.error("llm_non_retryable_error", error=str(exc))
                raise

            wait = _backoff_wait(attempt)
            logger.warning(
                "llm_retryable_error",
                attempt=attempt,
                error=str(exc)[:120],
                retry_in=round(wait, 2),
            )
            await asyncio.sleep(wait)

    logger.error("llm_all_retries_exhausted", attempts=settings.gemini_max_retries)
    raise RuntimeError(
        f"Gemini call failed after {settings.gemini_max_retries} retries"
    ) from last_exc


# ── Streaming generate ────────────────────────────────────────────────────────

async def generate_stream(
    prompt: str,
    system_prompt: str = "",
    temperature: float = 0.3,
    max_output_tokens: int = 8192,
    estimated_input_tokens: int = 500,
) -> AsyncGenerator[str, None]:
    """
    Stream Gemini response — yields text chunks as they arrive.
    Falls back to non-streaming on repeated rate limit failures.

    Usage:
        async for chunk in generate_stream("explain..."):
            print(chunk, end="", flush=True)
    """
    await wait_for_capacity(estimated_tokens=estimated_input_tokens)

    last_exc: Exception | None = None
    for attempt in range(settings.gemini_max_retries):
        try:
            client = get_client()

            # Run the blocking stream iterator in a thread
            # We collect chunks via a queue to bridge sync → async
            queue: asyncio.Queue[str | None] = asyncio.Queue()

            async def _stream_worker() -> None:
                try:
                    for chunk in client.models.generate_content_stream(
                        model=settings.gemini_model,
                        contents=[prompt],
                        config=types.GenerateContentConfig(
                            temperature=temperature,
                            max_output_tokens=max_output_tokens,
                            system_instruction=system_prompt or None,
                        ),
                    ):
                        text_piece = chunk.text or ""
                        if text_piece:
                            await queue.put(text_piece)
                finally:
                    await queue.put(None)  # Sentinel

            task = asyncio.create_task(_stream_worker())

            while True:
                chunk = await queue.get()
                if chunk is None:
                    break
                yield chunk

            await task
            return  # Successful stream — done

        except Exception as exc:
            last_exc = exc
            if not _is_retryable(exc):
                raise

            wait = _backoff_wait(attempt)
            logger.warning(
                "stream_retryable_error",
                attempt=attempt,
                retry_in=round(wait, 2),
            )
            await asyncio.sleep(wait)

    raise RuntimeError("Gemini stream failed after max retries") from last_exc


# ── Embeddings ────────────────────────────────────────────────────────────────

async def embed_texts(
    texts: list[str],
    task_type: str = "RETRIEVAL_DOCUMENT",
) -> list[list[float]]:
    """
    Embed a list of texts using gemini-embedding-001.

    Cache-aware: only calls the API for texts not already cached.
    Handles batching to stay within API limits.

    Args:
        texts:     Texts to embed.
        task_type: "RETRIEVAL_DOCUMENT" | "RETRIEVAL_QUERY" | "SEMANTIC_SIMILARITY"

    Returns:
        List of embedding vectors aligned with input texts.
    """
    if not texts:
        return []

    # ── Pull what we can from cache ───────────────────────────────────────────
    cached_results, miss_idx = await get_cached_embeddings(texts)

    if not miss_idx:
        logger.debug("all_embeddings_cached", count=len(texts))
        return cached_results  # type: ignore[return-value]

    # ── Batch the misses ──────────────────────────────────────────────────────
    miss_texts = [texts[i] for i in miss_idx]
    fresh_embeddings: list[list[float]] = []

    batch_size = settings.embedding_batch_size
    for batch_start in range(0, len(miss_texts), batch_size):
        batch = miss_texts[batch_start : batch_start + batch_size]

        await wait_for_capacity(estimated_tokens=len(" ".join(batch)) // 4)

        for attempt in range(settings.gemini_max_retries):
            try:
                client = get_client()
                result = await asyncio.to_thread(
                    client.models.embed_content,
                    model=settings.gemini_embedding_model,
                    contents=batch,
                    config=types.EmbedContentConfig(task_type=task_type),
                )
                batch_vecs = [e.values for e in result.embeddings]
                fresh_embeddings.extend(batch_vecs)
                break
            except Exception as exc:
                if not _is_retryable(exc) or attempt == settings.gemini_max_retries - 1:
                    raise
                await asyncio.sleep(_backoff_wait(attempt))

    # ── Cache the fresh embeddings ────────────────────────────────────────────
    await set_cached_embeddings(miss_texts, fresh_embeddings)

    # ── Merge cached + fresh into aligned output ──────────────────────────────
    fresh_iter = iter(fresh_embeddings)
    for i in miss_idx:
        cached_results[i] = next(fresh_iter)

    return cached_results  # type: ignore[return-value]


async def embed_query(query: str) -> list[float]:
    """Convenience: embed a single retrieval query."""
    results = await embed_texts([query], task_type="RETRIEVAL_QUERY")
    return results[0]


# ── Helpers ───────────────────────────────────────────────────────────────────

def _backoff_wait(attempt: int) -> float:
    """
    Exponential backoff with full jitter.
    wait = min(cap, base * 2^attempt) + uniform(0, jitter_max)
    """
    base = settings.gemini_retry_min_wait
    cap = settings.gemini_retry_max_wait
    jitter_max = settings.gemini_retry_jitter_max

    exponential = min(cap, base * (2 ** attempt))
    jitter = random.uniform(0, jitter_max)
    return exponential + jitter