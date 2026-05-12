"""
MCP Tool: query_conversation_metrics

Executes parameterised analytics queries against PostgreSQL.
The LLM calls this when it needs quantitative data:
  "engagement rate last week", "message volume by hour", etc.

All queries are parameterised — no string interpolation, no SQL injection.
"""

import time
from datetime import datetime, timedelta
from typing import Any

from sqlalchemy import text

from db.postgres import get_db
from core.cache import analysis_cache_key, cache_get, cache_set
from core.config import settings
from core.logging import get_logger
from schemas import (
    MetricsQueryInput,
    MetricsQueryOutput,
    MetricResult,
    MetricDataPoint,
    MetricType,
    AggregationPeriod,
    TrendDirection,
)

logger = get_logger(__name__)

# Mapping: aggregation period → PostgreSQL date_trunc value
_TRUNC = {
    AggregationPeriod.HOUR: "hour",
    AggregationPeriod.DAY: "day",
    AggregationPeriod.WEEK: "week",
    AggregationPeriod.MONTH: "month",
}


async def query_conversation_metrics(params: MetricsQueryInput) -> MetricsQueryOutput:
    """
    Execute analytics queries for the requested metrics.

    The LLM provides:
    - Which metrics to compute (engagement_rate, failure_rate, etc.)
    - The time window
    - Optional filters (intent, channel, etc.)
    - Whether to compare with the previous period

    Returns structured time-series data with trend indicators.
    """
    t0 = time.perf_counter()

    cache_key = analysis_cache_key(
        "metrics",
        f"{params.metrics}:{params.time_range}:{params.aggregation_period}"
    )
    cached = await cache_get(cache_key)
    if cached:
        logger.debug("metrics_cache_hit")
        return MetricsQueryOutput(**cached)

    results: list[MetricResult] = []

    # Compute each requested metric in parallel would be ideal,
    # but SQLAlchemy async sessions aren't shareable — run sequentially
    # (still fast as these are indexed queries)
    async with get_db() as db:
        for metric in params.metrics:
            result = await _compute_metric(db, metric, params)
            results.append(result)

    duration_ms = (time.perf_counter() - t0) * 1000

    output = MetricsQueryOutput(
        time_range=params.time_range,
        aggregation_period=params.aggregation_period,
        results=results,
        filters_applied=params.filters,
        query_duration_ms=round(duration_ms, 2),
    )

    await cache_set(
        cache_key,
        output.model_dump(mode="json"),
        settings.cache_ttl_analysis,
    )

    logger.info(
        "metrics_query_complete",
        metrics=[m.value for m in params.metrics],
        duration_ms=round(duration_ms, 1),
    )
    return output


# ── Individual metric computers ───────────────────────────────────────────────

async def _compute_metric(db, metric: MetricType, params: MetricsQueryInput) -> MetricResult:
    dispatch = {
        MetricType.ENGAGEMENT_RATE:      _engagement_rate,
        MetricType.FAILURE_RATE:         _failure_rate,
        MetricType.RESOLUTION_RATE:      _resolution_rate,
        MetricType.AVG_TURNS:            _avg_turns,
        MetricType.SESSION_DURATION:     _session_duration,
        MetricType.MESSAGE_VOLUME:       _message_volume,
        MetricType.UNIQUE_USERS:         _unique_users,
        MetricType.INTENT_DISTRIBUTION:  _intent_distribution,
    }
    fn = dispatch[metric]
    return await fn(db, params)


async def _message_volume(db, params: MetricsQueryInput) -> MetricResult:
    trunc = _TRUNC[params.aggregation_period]
    query = text("""
        SELECT
            DATE_TRUNC(:trunc, created_at) AS bucket,
            COUNT(*)                        AS value
        FROM chat_messages
        WHERE created_at BETWEEN :start AND :end
          AND role = 'user'
        GROUP BY bucket
        ORDER BY bucket
    """)
    rows = await db.execute(query, {
        "trunc": trunc,
        "start": params.time_range.start,
        "end": params.time_range.end,
    })
    time_series = [
        MetricDataPoint(timestamp=r.bucket, value=float(r.value))
        for r in rows.fetchall()
    ]

    current = sum(p.value for p in time_series)
    previous = None
    change_pct = None

    if params.compare_with_previous_period:
        prev_start, prev_end = _previous_period(params.time_range)
        prev_rows = await db.execute(query, {
            "trunc": trunc, "start": prev_start, "end": prev_end,
        })
        previous = float(sum(r.value for r in prev_rows.fetchall()))
        if previous > 0:
            change_pct = ((current - previous) / previous) * 100

    return MetricResult(
        metric=MetricType.MESSAGE_VOLUME,
        current_value=current,
        previous_value=previous,
        change_pct=change_pct,
        trend=_infer_trend(time_series),
        time_series=time_series,
        unit="messages",
    )


async def _engagement_rate(db, params: MetricsQueryInput) -> MetricResult:
    """
    Engagement = sessions with ≥2 user messages / total sessions.
    """
    trunc = _TRUNC[params.aggregation_period]
    query = text("""
        WITH session_stats AS (
            SELECT
                DATE_TRUNC(:trunc, cs.created_at) AS bucket,
                cs.session_id,
                COUNT(cm.id) FILTER (WHERE cm.role = 'user') AS user_turns
            FROM chat_sessions cs
            LEFT JOIN chat_messages cm ON cm.session_id = cs.session_id
            WHERE cs.created_at BETWEEN :start AND :end
            GROUP BY bucket, cs.session_id
        )
        SELECT
            bucket,
            ROUND(
                100.0 * COUNT(*) FILTER (WHERE user_turns >= 2) / NULLIF(COUNT(*), 0),
                2
            ) AS value
        FROM session_stats
        GROUP BY bucket
        ORDER BY bucket
    """)
    rows = await db.execute(query, {
        "trunc": trunc,
        "start": params.time_range.start,
        "end": params.time_range.end,
    })
    time_series = [
        MetricDataPoint(timestamp=r.bucket, value=float(r.value or 0))
        for r in rows.fetchall()
    ]
    current = (
        sum(p.value for p in time_series) / len(time_series)
        if time_series else 0.0
    )
    return MetricResult(
        metric=MetricType.ENGAGEMENT_RATE,
        current_value=round(current, 2),
        trend=_infer_trend(time_series),
        time_series=time_series,
        unit="%",
    )


async def _failure_rate(db, params: MetricsQueryInput) -> MetricResult:
    trunc = _TRUNC[params.aggregation_period]
    query = text("""
        SELECT
            DATE_TRUNC(:trunc, cm.created_at)  AS bucket,
            ROUND(
                100.0 * COUNT(*) FILTER (WHERE ic.is_failure = TRUE)
                       / NULLIF(COUNT(*), 0),
                2
            ) AS value
        FROM chat_messages cm
        LEFT JOIN intent_classifications ic ON ic.message_id = cm.id
        WHERE cm.created_at BETWEEN :start AND :end
          AND cm.role = 'user'
        GROUP BY bucket
        ORDER BY bucket
    """)
    rows = await db.execute(query, {
        "trunc": trunc,
        "start": params.time_range.start,
        "end": params.time_range.end,
    })
    time_series = [
        MetricDataPoint(timestamp=r.bucket, value=float(r.value or 0))
        for r in rows.fetchall()
    ]
    current = (
        sum(p.value for p in time_series) / len(time_series)
        if time_series else 0.0
    )
    return MetricResult(
        metric=MetricType.FAILURE_RATE,
        current_value=round(current, 2),
        trend=_infer_trend(time_series),
        time_series=time_series,
        unit="%",
    )


async def _resolution_rate(db, params: MetricsQueryInput) -> MetricResult:
    trunc = _TRUNC[params.aggregation_period]
    query = text("""
        SELECT
            DATE_TRUNC(:trunc, cs.created_at)  AS bucket,
            ROUND(
                100.0 * COUNT(*) FILTER (WHERE cs.resolved = TRUE)
                       / NULLIF(COUNT(*), 0),
                2
            ) AS value
        FROM chat_sessions cs
        WHERE cs.created_at BETWEEN :start AND :end
        GROUP BY bucket
        ORDER BY bucket
    """)
    rows = await db.execute(query, {
        "trunc": trunc,
        "start": params.time_range.start,
        "end": params.time_range.end,
    })
    time_series = [
        MetricDataPoint(timestamp=r.bucket, value=float(r.value or 0))
        for r in rows.fetchall()
    ]
    current = (
        sum(p.value for p in time_series) / len(time_series)
        if time_series else 0.0
    )
    return MetricResult(
        metric=MetricType.RESOLUTION_RATE,
        current_value=round(current, 2),
        trend=_infer_trend(time_series),
        time_series=time_series,
        unit="%",
    )


async def _avg_turns(db, params: MetricsQueryInput) -> MetricResult:
    trunc = _TRUNC[params.aggregation_period]
    query = text("""
        SELECT
            DATE_TRUNC(:trunc, cs.created_at) AS bucket,
            ROUND(AVG(msg_counts.cnt), 2)     AS value
        FROM chat_sessions cs
        JOIN (
            SELECT session_id, COUNT(*) AS cnt
            FROM chat_messages
            WHERE role = 'user'
            GROUP BY session_id
        ) msg_counts ON msg_counts.session_id = cs.session_id
        WHERE cs.created_at BETWEEN :start AND :end
        GROUP BY bucket
        ORDER BY bucket
    """)
    rows = await db.execute(query, {
        "trunc": trunc,
        "start": params.time_range.start,
        "end": params.time_range.end,
    })
    time_series = [
        MetricDataPoint(timestamp=r.bucket, value=float(r.value or 0))
        for r in rows.fetchall()
    ]
    current = (
        sum(p.value for p in time_series) / len(time_series)
        if time_series else 0.0
    )
    return MetricResult(
        metric=MetricType.AVG_TURNS,
        current_value=round(current, 2),
        trend=_infer_trend(time_series),
        time_series=time_series,
        unit="turns",
    )


async def _session_duration(db, params: MetricsQueryInput) -> MetricResult:
    trunc = _TRUNC[params.aggregation_period]
    query = text("""
        SELECT
            DATE_TRUNC(:trunc, cs.created_at)           AS bucket,
            ROUND(AVG(
                EXTRACT(EPOCH FROM (cs.ended_at - cs.created_at)) / 60
            ), 2)                                        AS value
        FROM chat_sessions cs
        WHERE cs.created_at BETWEEN :start AND :end
          AND cs.ended_at IS NOT NULL
        GROUP BY bucket
        ORDER BY bucket
    """)
    rows = await db.execute(query, {
        "trunc": trunc,
        "start": params.time_range.start,
        "end": params.time_range.end,
    })
    time_series = [
        MetricDataPoint(timestamp=r.bucket, value=float(r.value or 0))
        for r in rows.fetchall()
    ]
    current = (
        sum(p.value for p in time_series) / len(time_series)
        if time_series else 0.0
    )
    return MetricResult(
        metric=MetricType.SESSION_DURATION,
        current_value=round(current, 2),
        trend=_infer_trend(time_series),
        time_series=time_series,
        unit="minutes",
    )


async def _unique_users(db, params: MetricsQueryInput) -> MetricResult:
    trunc = _TRUNC[params.aggregation_period]
    query = text("""
        SELECT
            DATE_TRUNC(:trunc, created_at) AS bucket,
            COUNT(DISTINCT user_id)        AS value
        FROM chat_sessions
        WHERE created_at BETWEEN :start AND :end
        GROUP BY bucket
        ORDER BY bucket
    """)
    rows = await db.execute(query, {
        "trunc": trunc,
        "start": params.time_range.start,
        "end": params.time_range.end,
    })
    time_series = [
        MetricDataPoint(timestamp=r.bucket, value=float(r.value))
        for r in rows.fetchall()
    ]
    return MetricResult(
        metric=MetricType.UNIQUE_USERS,
        current_value=sum(p.value for p in time_series),
        trend=_infer_trend(time_series),
        time_series=time_series,
        unit="users",
    )


async def _intent_distribution(db, params: MetricsQueryInput) -> MetricResult:
    """Returns the most common intent as current_value; full dist in time_series."""
    query = text("""
        SELECT
            ic.intent              AS label,
            COUNT(*)               AS value
        FROM intent_classifications ic
        JOIN chat_messages cm ON cm.id = ic.message_id
        WHERE cm.created_at BETWEEN :start AND :end
        GROUP BY ic.intent
        ORDER BY value DESC
        LIMIT 20
    """)
    rows = await db.execute(query, {
        "start": params.time_range.start,
        "end": params.time_range.end,
    })
    from datetime import datetime as dt
    now = dt.utcnow()
    time_series = [
        MetricDataPoint(timestamp=now, value=float(r.value), label=r.label)
        for r in rows.fetchall()
    ]
    top_count = time_series[0].value if time_series else 0.0
    return MetricResult(
        metric=MetricType.INTENT_DISTRIBUTION,
        current_value=top_count,
        trend=TrendDirection.STABLE,
        time_series=time_series,
        unit="messages",
    )


# ── Helpers ───────────────────────────────────────────────────────────────────

def _infer_trend(series: list[MetricDataPoint]) -> TrendDirection:
    """
    Simple linear regression slope to determine trend direction.
    Requires at least 3 data points.
    """
    if len(series) < 3:
        return TrendDirection.STABLE

    values = [p.value for p in series]
    n = len(values)
    mean_v = sum(values) / n

    # Variance check — if very low, it's stable
    variance = sum((v - mean_v) ** 2 for v in values) / n
    if variance < 0.01 * mean_v:
        return TrendDirection.STABLE

    # Simple first/last comparison for direction
    first_half = sum(values[: n // 2]) / (n // 2)
    second_half = sum(values[n // 2 :]) / (n - n // 2)
    change = (second_half - first_half) / max(abs(first_half), 1e-10)

    if change > 0.05:
        return TrendDirection.INCREASING
    elif change < -0.05:
        return TrendDirection.DECREASING
    else:
        return TrendDirection.STABLE


def _previous_period(time_range) -> tuple[datetime, datetime]:
    """Return the equivalent previous time window for period-over-period comparison."""
    delta = time_range.end - time_range.start
    return time_range.start - delta, time_range.start