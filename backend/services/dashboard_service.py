"""
Computes and caches the metrics snapshot consumed by the Streamlit dashboard.

The dashboard polls /dashboard every 30 seconds.
Results are cached in Redis for 5 minutes to avoid hammering PostgreSQL.
"""

import json
from datetime import datetime, timedelta

from sqlalchemy import text

from db.connections import get_db, get_redis
from core.config import settings
from core.logging import get_logger

logger = get_logger(__name__)

_CACHE_PREFIX = "cache:dashboard:metrics"


def _cache_key(days: int) -> str:
    return f"{_CACHE_PREFIX}:{days}"


async def get_dashboard_metrics(days: int = 7, force: bool = False) -> dict:
    """
    Return aggregated metrics for the dashboard.
    Serves from Redis cache when available, otherwise queries PostgreSQL.
    Pass force=True to bypass the cache (used by the Refresh button).
    """
    redis = get_redis()
    key = _cache_key(days)

    if not force:
        cached = await redis.get(key)
        if cached:
            logger.debug("dashboard_cache_hit", days=days)
            return json.loads(cached)

    metrics = await _compute_metrics(days)
    await redis.setex(key, settings.cache_ttl_dashboard, json.dumps(metrics, default=str))
    return metrics


async def invalidate_dashboard_cache() -> None:
    """Call this after storing new feedback or completing an agent run."""
    redis = get_redis()
    # Invalidate all day-range variants
    async for key in redis.scan_iter(f"{_CACHE_PREFIX}:*"):
        await redis.delete(key)


async def _compute_metrics(days: int) -> dict:
    now = datetime.utcnow()
    start = now - timedelta(days=days)
    prev_start = start - timedelta(days=days)

    async with get_db() as db:

        # ── Session totals ────────────────────────────────────────────────────
        row = await db.execute(text("""
            SELECT
                COUNT(*)                                        AS total_sessions,
                COUNT(*) FILTER (WHERE created_at >= :prev)     AS prev_sessions
            FROM chat_sessions
            WHERE created_at >= :prev_start
        """), {"prev": start, "prev_start": prev_start})
        totals = row.fetchone()

        # ── Message volume ────────────────────────────────────────────────────
        row = await db.execute(text("""
            SELECT COUNT(*) AS total_messages
            FROM chat_messages
            WHERE created_at >= :start AND role = 'user'
        """), {"start": start})
        msg_row = row.fetchone()

        # ── Engagement rate ───────────────────────────────────────────────────
        row = await db.execute(text("""
            SELECT ROUND(
                100.0 * COUNT(*) FILTER (WHERE msg_count >= 2)
                       / NULLIF(COUNT(*), 0), 1
            ) AS engagement_rate
            FROM (
                SELECT cs.session_id,
                       COUNT(cm.id) FILTER (WHERE cm.role = 'user') AS msg_count
                FROM chat_sessions cs
                LEFT JOIN chat_messages cm ON cm.session_id = cs.session_id
                WHERE cs.created_at >= :start
                GROUP BY cs.session_id
            ) s
        """), {"start": start})
        eng_row = row.fetchone()

        # ── Failure rate ──────────────────────────────────────────────────────
        row = await db.execute(text("""
            SELECT ROUND(
                100.0 * COUNT(*) FILTER (WHERE ic.is_failure)
                       / NULLIF(COUNT(*), 0), 1
            ) AS failure_rate
            FROM chat_messages cm
            LEFT JOIN intent_classifications ic ON ic.message_id = cm.id
            WHERE cm.created_at >= :start AND cm.role = 'user'
        """), {"start": start})
        fail_row = row.fetchone()

        # ── Average rating ────────────────────────────────────────────────────
        row = await db.execute(text("""
            SELECT ROUND(AVG(rating)::NUMERIC, 2) AS avg_rating
            FROM user_feedback WHERE created_at >= :start
        """), {"start": start})
        rating_row = row.fetchone()

        # ── Top intents ───────────────────────────────────────────────────────
        rows = await db.execute(text("""
            SELECT intent, COUNT(*) AS count
            FROM intent_classifications ic
            JOIN chat_messages cm ON cm.id = ic.message_id
            WHERE cm.created_at >= :start
            GROUP BY intent
            ORDER BY count DESC
            LIMIT 8
        """), {"start": start})
        top_intents = [{"intent": r.intent, "count": r.count} for r in rows.fetchall()]

        # ── Daily trends ──────────────────────────────────────────────────────
        rows = await db.execute(text("""
            SELECT
                DATE_TRUNC('day', cs.created_at)::DATE::TEXT AS date,
                ROUND(
                    100.0 * COUNT(*) FILTER (WHERE msgs.cnt >= 2)
                           / NULLIF(COUNT(*), 0), 1
                ) AS value
            FROM chat_sessions cs
            LEFT JOIN (
                SELECT session_id, COUNT(*) FILTER (WHERE role='user') AS cnt
                FROM chat_messages GROUP BY session_id
            ) msgs ON msgs.session_id = cs.session_id
            WHERE cs.created_at >= :start
            GROUP BY 1 ORDER BY 1
        """), {"start": start})
        engagement_trend = [{"date": r.date, "value": float(r.value or 0)} for r in rows.fetchall()]

        rows = await db.execute(text("""
            SELECT
                DATE_TRUNC('day', cm.created_at)::DATE::TEXT AS date,
                ROUND(
                    100.0 * COUNT(*) FILTER (WHERE ic.is_failure)
                           / NULLIF(COUNT(*), 0), 1
                ) AS value
            FROM chat_messages cm
            LEFT JOIN intent_classifications ic ON ic.message_id = cm.id
            WHERE cm.created_at >= :start AND cm.role = 'user'
            GROUP BY 1 ORDER BY 1
        """), {"start": start})
        failure_trend = [{"date": r.date, "value": float(r.value or 0)} for r in rows.fetchall()]

        rows = await db.execute(text("""
            SELECT
                DATE_TRUNC('day', created_at)::DATE::TEXT AS date,
                COUNT(*) AS value
            FROM chat_messages
            WHERE created_at >= :start AND role = 'user'
            GROUP BY 1 ORDER BY 1
        """), {"start": start})
        volume_trend = [{"date": r.date, "value": r.value} for r in rows.fetchall()]

        # ── Top failure intents ───────────────────────────────────────────────
        rows = await db.execute(text("""
            SELECT
                ic.intent,
                COUNT(*) FILTER (WHERE ic.is_failure) AS failures,
                COUNT(*) AS total,
                ROUND(
                    100.0 * COUNT(*) FILTER (WHERE ic.is_failure)
                           / NULLIF(COUNT(*), 0), 1
                ) AS rate
            FROM intent_classifications ic
            JOIN chat_messages cm ON cm.id = ic.message_id
            WHERE cm.created_at >= :start
            GROUP BY ic.intent
            HAVING COUNT(*) FILTER (WHERE ic.is_failure) > 0
            ORDER BY failures DESC
            LIMIT 6
        """), {"start": start})
        top_failures = [
            {"intent": r.intent, "failures": r.failures, "total": r.total, "rate": float(r.rate or 0)}
            for r in rows.fetchall()
        ]

        # ── Recent feedback ───────────────────────────────────────────────────
        rows = await db.execute(text("""
            SELECT rating, sentiment, comment, created_at
            FROM user_feedback
            WHERE created_at >= :start
            ORDER BY created_at DESC
            LIMIT 10
        """), {"start": start})
        recent_feedback = [
            {
                "rating": r.rating,
                "sentiment": r.sentiment,
                "comment": r.comment,
                "created_at": r.created_at.isoformat(),
            }
            for r in rows.fetchall()
        ]

    return {
        "period_label": f"Last {days} days",
        "total_sessions": totals.total_sessions or 0,
        "total_messages": msg_row.total_messages or 0,
        "engagement_rate": float(eng_row.engagement_rate or 0),
        "failure_rate": float(fail_row.failure_rate or 0),
        "avg_rating": float(rating_row.avg_rating) if rating_row.avg_rating else None,
        "top_intents": top_intents,
        "engagement_trend": engagement_trend,
        "failure_trend": failure_trend,
        "volume_trend": volume_trend,
        "top_failures": top_failures,
        "recent_feedback": recent_feedback,
    }