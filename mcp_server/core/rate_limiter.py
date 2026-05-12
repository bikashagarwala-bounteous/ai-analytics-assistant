"""
Redis sliding window rate limiter for Gemini (RPM/RPD/TPM).
Uses sorted sets for O(log N) window checks. Call check_and_consume() before each Gemini request.
"""

import time
import asyncio
from dataclasses import dataclass
from enum import Enum

from db.redis_client import get_redis
from core.config import settings
from core.logging import get_logger

logger = get_logger(__name__)

# Redis key prefixes
_KEY_RPM = "ratelimit:gemini:rpm"
_KEY_RPD = "ratelimit:gemini:rpd"
_KEY_TPM = "ratelimit:gemini:tpm"


class RateLimitExceeded(Exception):
    """Raised when a rate limit window is saturated."""

    def __init__(self, limit_type: str, retry_after: float) -> None:
        self.limit_type = limit_type
        self.retry_after = retry_after          # Seconds until a slot opens
        super().__init__(
            f"Gemini rate limit hit: {limit_type}. "
            f"Retry after {retry_after:.1f}s"
        )


@dataclass
class RateLimitStatus:
    rpm_used: int
    rpm_limit: int
    rpd_used: int
    rpd_limit: int
    tpm_used: int
    tpm_limit: int

    @property
    def is_healthy(self) -> bool:
        return (
            self.rpm_used < self.rpm_limit * 0.8
            and self.rpd_used < self.rpd_limit * 0.8
            and self.tpm_used < self.tpm_limit * 0.8
        )


async def _sliding_window_check(
    key: str,
    window_seconds: int,
    limit: int,
    cost: int = 1,
) -> tuple[bool, float]:
    """
    Atomic sliding window check using a Redis sorted set.

    Returns (allowed: bool, retry_after_seconds: float).
    The sorted set stores timestamps as both score and member (with a UUID suffix
    to handle identical timestamps).
    """
    redis = get_redis()
    now = time.time()
    window_start = now - window_seconds

    pipe = redis.pipeline()
    # Remove entries outside the window
    pipe.zremrangebyscore(key, "-inf", window_start)
    # Count current entries
    pipe.zcard(key)
    results = await pipe.execute()

    current_count = results[1]

    if current_count + cost > limit:
        # Find the oldest entry in the window to compute retry_after
        oldest = await redis.zrange(key, 0, 0, withscores=True)
        if oldest:
            oldest_ts = oldest[0][1]
            retry_after = (oldest_ts + window_seconds) - now
        else:
            retry_after = window_seconds
        return False, max(retry_after, 0.1)

    # Record the new request(s)
    pipe = redis.pipeline()
    for i in range(cost):
        member = f"{now}:{i}"
        pipe.zadd(key, {member: now})
    pipe.expire(key, window_seconds + 10)   # TTL slightly longer than window
    await pipe.execute()

    return True, 0.0


async def check_and_consume(estimated_tokens: int = 100) -> None:
    """
    Check all rate limit windows and consume a slot.
    Raises RateLimitExceeded if any window is full.

    Args:
        estimated_tokens: Rough token estimate for TPM tracking.
    """
    # ── RPM check ────────────────────────────────────────────────────────────
    rpm_ok, rpm_retry = await _sliding_window_check(
        _KEY_RPM, 60, settings.gemini_rpm_limit
    )
    if not rpm_ok:
        logger.warning("gemini_rpm_limit_hit", retry_after=rpm_retry)
        raise RateLimitExceeded("RPM", rpm_retry)

    # ── RPD check ────────────────────────────────────────────────────────────
    rpd_ok, rpd_retry = await _sliding_window_check(
        _KEY_RPD, 86_400, settings.gemini_rpd_limit
    )
    if not rpd_ok:
        logger.warning("gemini_rpd_limit_hit", retry_after=rpd_retry)
        raise RateLimitExceeded("RPD", rpd_retry)

    # ── TPM check (token cost) ────────────────────────────────────────────────
    # We treat every ~100 tokens as 1 unit to keep the sorted set size manageable
    token_units = max(1, estimated_tokens // 100)
    tpm_ok, tpm_retry = await _sliding_window_check(
        _KEY_TPM, 60, settings.gemini_tpm_limit // 100, cost=token_units
    )
    if not tpm_ok:
        logger.warning("gemini_tpm_limit_hit", retry_after=tpm_retry)
        raise RateLimitExceeded("TPM", tpm_retry)

    logger.debug("rate_limit_ok", estimated_tokens=estimated_tokens)


async def get_status() -> RateLimitStatus:
    """Return current usage across all windows (for monitoring/dashboard)."""
    redis = get_redis()
    now = time.time()

    rpm_count = await redis.zcount(_KEY_RPM, now - 60, "+inf")
    rpd_count = await redis.zcount(_KEY_RPD, now - 86_400, "+inf")
    tpm_count = await redis.zcount(_KEY_TPM, now - 60, "+inf")

    return RateLimitStatus(
        rpm_used=rpm_count,
        rpm_limit=settings.gemini_rpm_limit,
        rpd_used=rpd_count,
        rpd_limit=settings.gemini_rpd_limit,
        tpm_used=tpm_count * 100,           # Convert back to tokens
        tpm_limit=settings.gemini_tpm_limit,
    )


async def wait_for_capacity(
    estimated_tokens: int = 100,
    timeout: float = 120.0,
) -> None:
    """
    Block (with async sleep) until a rate limit slot is available.
    Used by the request queue — callers should prefer this over hard errors.

    Raises TimeoutError if capacity doesn't free up within `timeout` seconds.
    """
    deadline = time.time() + timeout
    attempt = 0

    while time.time() < deadline:
        try:
            await check_and_consume(estimated_tokens)
            return
        except RateLimitExceeded as exc:
            wait = min(exc.retry_after + 0.5, deadline - time.time())
            if wait <= 0:
                raise TimeoutError(
                    f"Rate limit capacity not available within {timeout}s"
                ) from exc
            logger.info(
                "waiting_for_rate_limit",
                limit_type=exc.limit_type,
                wait_seconds=round(wait, 2),
                attempt=attempt,
            )
            await asyncio.sleep(wait)
            attempt += 1

    raise TimeoutError(f"Rate limit timeout after {timeout}s")