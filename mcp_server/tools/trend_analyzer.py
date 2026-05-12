"""
MCP Tools:
  - analyze_trends      : Linear regression + seasonality detection + optional forecasting
  - get_failure_intents : Top failing intents with rates, trends, and example messages
"""

import time
from datetime import datetime, timedelta

import numpy as np

from db.postgres import get_db
from core.logging import get_logger
from schemas import (
    TrendAnalyzerInput,
    TrendAnalyzerOutput,
    TrendResult,
    TrendSegment,
    ForecastPoint,
    TrendDirection,
    AggregationPeriod,
    FailureIntentsInput,
    FailureIntentsOutput,
    FailureIntent,
    MetricType,
)

logger = get_logger(__name__)


# Tool: analyze_trends
async def analyze_trends(params: TrendAnalyzerInput) -> TrendAnalyzerOutput:
    """
    Compute trend analysis for one or more metrics.

    Steps per metric:
    1. Fetch aggregated time series from PostgreSQL
    2. Fit a linear regression line
    3. Detect change-point segments
    4. Optionally forecast future periods
    5. Detect seasonal patterns (weekly / monthly)
    6. Generate a natural-language insight string

    Cross-metric: compute pairwise Pearson correlations.
    """
    t0 = time.perf_counter()
    results: list[TrendResult] = []
    series_map: dict[str, list[float]] = {}   # For correlation matrix

    for metric in params.metrics:
        raw_series = await _fetch_series(metric, params)
        if len(raw_series) < 3:
            continue

        timestamps = [r[0] for r in raw_series]
        values = np.array([r[1] for r in raw_series])
        series_map[metric.value] = values.tolist()

        direction, slope, r_sq = _linear_trend(values)
        segments = _detect_segments(timestamps, values)
        seasonal = _detect_seasonality(timestamps, values) if params.include_seasonality else None
        forecast = (
            _forecast(timestamps, values, params.forecast_periods, params.aggregation_period)
            if params.forecast_periods > 0 else []
        )
        insight = _generate_insight(metric, direction, slope, r_sq, seasonal, len(raw_series))

        results.append(TrendResult(
            metric=metric,
            overall_direction=direction,
            overall_slope=round(slope, 6),
            r_squared=round(r_sq, 4),
            segments=segments,
            seasonal_pattern=seasonal,
            forecast=forecast,
            insight=insight,
        ))

    # Pairwise correlations
    correlations: dict[str, float] = {}
    metric_names = list(series_map.keys())
    for i in range(len(metric_names)):
        for j in range(i + 1, len(metric_names)):
            a = np.array(series_map[metric_names[i]])
            b = np.array(series_map[metric_names[j]])
            min_len = min(len(a), len(b))
            if min_len > 2:
                corr = float(np.corrcoef(a[:min_len], b[:min_len])[0, 1])
                key = f"{metric_names[i]}↔{metric_names[j]}"
                correlations[key] = round(corr, 4)

    duration_ms = (time.perf_counter() - t0) * 1000
    logger.info(
        "trend_analysis_complete",
        metrics=[m.value for m in params.metrics],
        results=len(results),
        duration_ms=round(duration_ms, 1),
    )

    return TrendAnalyzerOutput(
        time_range=params.time_range,
        aggregation_period=params.aggregation_period,
        results=results,
        cross_metric_correlations=correlations,
        analysis_duration_ms=round(duration_ms, 2),
    )


# Tool: get_failure_intents
async def get_failure_intents(params: FailureIntentsInput) -> FailureIntentsOutput:
    """
    Return the top-N most frequently failing intents.

    Includes:
    - Failure count and rate per intent
    - Trend direction (is it getting worse?)
    - Sample failed messages for context
    """
    t0 = time.perf_counter()

    from sqlalchemy import text

    async with get_db() as db:
        rows = await db.execute(text("""
            SELECT
                ic.intent,
                COUNT(*) FILTER (WHERE ic.is_failure = TRUE)   AS failure_count,
                COUNT(*)                                        AS total_count,
                ROUND(
                    100.0 * COUNT(*) FILTER (WHERE ic.is_failure = TRUE)
                         / NULLIF(COUNT(*), 0),
                    2
                )                                              AS failure_rate,
                AVG(ic.confidence)                             AS avg_confidence
            FROM intent_classifications ic
            JOIN chat_messages cm ON cm.id = ic.message_id
            WHERE cm.created_at BETWEEN :start AND :end
              AND ic.intent IS NOT NULL
            GROUP BY ic.intent
            HAVING COUNT(*) FILTER (WHERE ic.is_failure = TRUE) >= :min_occ
            ORDER BY failure_count DESC
            LIMIT :top_n
        """), {
            "start": params.time_range.start,
            "end": params.time_range.end,
            "min_occ": params.min_occurrences,
            "top_n": params.top_n,
        })
        intent_rows = rows.fetchall()

        samples: dict[str, list[str]] = {}
        if params.include_examples and intent_rows:
            for row in intent_rows:
                sample_rows = await db.execute(text("""
                    SELECT cm.content
                    FROM chat_messages cm
                    JOIN intent_classifications ic ON ic.message_id = cm.id
                    WHERE ic.intent = :intent
                      AND ic.is_failure = TRUE
                      AND cm.created_at BETWEEN :start AND :end
                    ORDER BY RANDOM()
                    LIMIT 3
                """), {
                    "intent": row.intent,
                    "start": params.time_range.start,
                    "end": params.time_range.end,
                })
                samples[row.intent] = [r.content for r in sample_rows.fetchall()]

        # ── Trend per intent (compare first/second half of window) ────────────
        mid = params.time_range.start + (
            params.time_range.end - params.time_range.start
        ) / 2

        intent_trends: dict[str, TrendDirection] = {}
        for row in intent_rows:
            first_half = await db.execute(text("""
                SELECT COUNT(*) AS cnt
                FROM intent_classifications ic
                JOIN chat_messages cm ON cm.id = ic.message_id
                WHERE ic.intent = :intent AND ic.is_failure = TRUE
                  AND cm.created_at BETWEEN :start AND :mid
            """), {"intent": row.intent, "start": params.time_range.start, "mid": mid})

            second_half = await db.execute(text("""
                SELECT COUNT(*) AS cnt
                FROM intent_classifications ic
                JOIN chat_messages cm ON cm.id = ic.message_id
                WHERE ic.intent = :intent AND ic.is_failure = TRUE
                  AND cm.created_at BETWEEN :mid AND :end
            """), {"intent": row.intent, "mid": mid, "end": params.time_range.end})

            f = first_half.scalar() or 0
            s = second_half.scalar() or 0
            if s > f * 1.1:
                intent_trends[row.intent] = TrendDirection.INCREASING
            elif s < f * 0.9:
                intent_trends[row.intent] = TrendDirection.DECREASING
            else:
                intent_trends[row.intent] = TrendDirection.STABLE

        # ── Totals ────────────────────────────────────────────────────────────
        totals = await db.execute(text("""
            SELECT
                COUNT(*) FILTER (WHERE ic.is_failure = TRUE) AS total_failures,
                COUNT(*)                                      AS total_interactions
            FROM intent_classifications ic
            JOIN chat_messages cm ON cm.id = ic.message_id
            WHERE cm.created_at BETWEEN :start AND :end
        """), {"start": params.time_range.start, "end": params.time_range.end})
        total_row = totals.fetchone()
        total_failures = total_row.total_failures or 0
        total_interactions = total_row.total_interactions or 1

    intents = [
        FailureIntent(
            intent=row.intent,
            failure_count=row.failure_count,
            total_count=row.total_count,
            failure_rate=float(row.failure_rate or 0),
            avg_confidence=float(row.avg_confidence or 0),
            trend=intent_trends.get(row.intent, TrendDirection.STABLE),
            sample_messages=samples.get(row.intent, []),
        )
        for row in intent_rows
    ]

    duration_ms = (time.perf_counter() - t0) * 1000
    logger.info(
        "failure_intents_complete",
        intents_found=len(intents),
        total_failures=total_failures,
        duration_ms=round(duration_ms, 1),
    )

    return FailureIntentsOutput(
        time_range=params.time_range,
        intents=intents,
        total_failures=total_failures,
        total_interactions=total_interactions,
        overall_failure_rate=round(
            100.0 * total_failures / total_interactions, 2
        ),
    )


# Internal helpers
def _linear_trend(values: np.ndarray) -> tuple[TrendDirection, float, float]:
    """OLS linear regression. Returns (direction, slope, R²)."""
    n = len(values)
    x = np.arange(n)
    x_mean, y_mean = x.mean(), values.mean()
    ss_xy = ((x - x_mean) * (values - y_mean)).sum()
    ss_xx = ((x - x_mean) ** 2).sum()

    slope = ss_xy / ss_xx if ss_xx != 0 else 0.0

    y_pred = x * slope + (y_mean - slope * x_mean)
    ss_res = ((values - y_pred) ** 2).sum()
    ss_tot = ((values - y_mean) ** 2).sum()
    r_sq = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0.0

    rel_slope = slope / max(abs(y_mean), 1e-10)
    if rel_slope > 0.02:
        direction = TrendDirection.INCREASING
    elif rel_slope < -0.02:
        direction = TrendDirection.DECREASING
    else:
        direction = TrendDirection.STABLE

    return direction, float(slope), float(max(r_sq, 0.0))


def _detect_segments(
    timestamps: list[datetime], values: np.ndarray, min_segment: int = 5
) -> list[TrendSegment]:
    """
    Naive change-point detection using rolling window slope changes.
    For production, consider ruptures or statsmodels Pelt.
    """
    if len(values) < min_segment * 2:
        direction, slope, _ = _linear_trend(values)
        return [TrendSegment(
            start=timestamps[0],
            end=timestamps[-1],
            direction=direction,
            slope=round(slope, 6),
            confidence=0.8,
        )]

    segments: list[TrendSegment] = []
    window = max(min_segment, len(values) // 4)
    prev_direction = None
    seg_start = 0

    for i in range(window, len(values), window // 2):
        chunk = values[max(0, i - window): i]
        direction, slope, _ = _linear_trend(chunk)

        if direction != prev_direction and prev_direction is not None:
            segments.append(TrendSegment(
                start=timestamps[seg_start],
                end=timestamps[i - 1],
                direction=prev_direction,
                slope=round(slope, 6),
                confidence=0.7,
            ))
            seg_start = i

        prev_direction = direction

    # Final segment
    if prev_direction:
        direction, slope, _ = _linear_trend(values[seg_start:])
        segments.append(TrendSegment(
            start=timestamps[seg_start],
            end=timestamps[-1],
            direction=direction,
            slope=round(slope, 6),
            confidence=0.75,
        ))

    return segments or [TrendSegment(
        start=timestamps[0],
        end=timestamps[-1],
        direction=TrendDirection.STABLE,
        slope=0.0,
        confidence=0.5,
    )]


def _detect_seasonality(timestamps: list[datetime], values: np.ndarray) -> str | None:
    """
    Detect weekly seasonality by comparing day-of-week averages.
    Returns a description string or None if no pattern found.
    """
    if len(values) < 14:     # Need at least 2 weeks
        return None

    day_groups: dict[int, list[float]] = {i: [] for i in range(7)}
    for ts, v in zip(timestamps, values):
        day_groups[ts.weekday()].append(float(v))

    day_avgs = {d: np.mean(vs) for d, vs in day_groups.items() if vs}
    if not day_avgs:
        return None

    overall_mean = np.mean(values)
    if overall_mean == 0:
        return None

    peak_day = max(day_avgs, key=day_avgs.get)
    trough_day = min(day_avgs, key=day_avgs.get)

    peak_ratio = day_avgs[peak_day] / overall_mean
    trough_ratio = day_avgs[trough_day] / overall_mean

    days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]

    if peak_ratio > 1.2 or trough_ratio < 0.8:
        return (
            f"Weekly pattern detected: peak on {days[peak_day]} "
            f"({peak_ratio:.1f}x avg), trough on {days[trough_day]} "
            f"({trough_ratio:.1f}x avg)"
        )
    return None


def _forecast(
    timestamps: list[datetime],
    values: np.ndarray,
    n_periods: int,
    period: AggregationPeriod,
) -> list[ForecastPoint]:
    """
    Prophet-based forecast with uncertainty intervals.
    Falls back to linear extrapolation if Prophet is unavailable.
    """
    try:
        from prophet import Prophet
        import pandas as pd

        df = pd.DataFrame({
            "ds": [ts.replace(tzinfo=None) if ts.tzinfo else ts for ts in timestamps],
            "y": values.tolist(),
        })

        period_freqs = {
            AggregationPeriod.HOUR: "h",
            AggregationPeriod.DAY: "D",
            AggregationPeriod.WEEK: "W",
            AggregationPeriod.MONTH: "MS",
        }
        freq = period_freqs[period]

        m = Prophet(
            interval_width=0.80,
            daily_seasonality=(period == AggregationPeriod.HOUR),
            weekly_seasonality=(period in (AggregationPeriod.DAY, AggregationPeriod.HOUR)),
            yearly_seasonality=False,
        )
        # Suppress Prophet's verbose Stan output
        import logging
        logging.getLogger("prophet").setLevel(logging.WARNING)
        logging.getLogger("cmdstanpy").setLevel(logging.WARNING)

        m.fit(df)
        future = m.make_future_dataframe(periods=n_periods, freq=freq)
        forecast_df = m.predict(future).tail(n_periods)

        return [
            ForecastPoint(
                timestamp=row["ds"].to_pydatetime(),
                predicted_value=round(max(float(row["yhat"]), 0), 4),
                lower_bound=round(max(float(row["yhat_lower"]), 0), 4),
                upper_bound=round(float(row["yhat_upper"]), 4),
            )
            for _, row in forecast_df.iterrows()
        ]

    except Exception:
        # Linear extrapolation fallback
        n = len(values)
        x = np.arange(n)
        _, slope, _ = _linear_trend(values)
        intercept = values.mean() - slope * x.mean()
        std = float(np.std(values - (x * slope + intercept)))
        ci_multiplier = 1.282

        period_deltas = {
            AggregationPeriod.HOUR: timedelta(hours=1),
            AggregationPeriod.DAY: timedelta(days=1),
            AggregationPeriod.WEEK: timedelta(weeks=1),
            AggregationPeriod.MONTH: timedelta(days=30),
        }
        delta = period_deltas[period]
        last_ts = timestamps[-1]

        return [
            ForecastPoint(
                timestamp=last_ts + delta * i,
                predicted_value=round(max(float(slope * (n + i - 1) + intercept), 0), 4),
                lower_bound=round(max(float(slope * (n + i - 1) + intercept) - ci_multiplier * std * (1 + i * 0.05), 0), 4),
                upper_bound=round(float(slope * (n + i - 1) + intercept) + ci_multiplier * std * (1 + i * 0.05), 4),
            )
            for i in range(1, n_periods + 1)
        ]


def _generate_insight(
    metric: MetricType,
    direction: TrendDirection,
    slope: float,
    r_sq: float,
    seasonal: str | None,
    n_points: int,
) -> str:
    """Generate a concise natural-language insight for the trend."""
    direction_words = {
        TrendDirection.INCREASING: "upward trend",
        TrendDirection.DECREASING: "downward trend",
        TrendDirection.STABLE: "stable pattern",
        TrendDirection.VOLATILE: "volatile behaviour",
    }

    confidence_word = "strong" if r_sq > 0.7 else "moderate" if r_sq > 0.4 else "weak"
    parts = [
        f"{metric.value.replace('_', ' ').title()} shows a "
        f"{confidence_word} {direction_words[direction]} "
        f"(R²={r_sq:.2f}, slope={slope:.4f}) "
        f"across {n_points} data points."
    ]
    if seasonal:
        parts.append(seasonal)

    return " ".join(parts)


# Data fetcher
_METRIC_QUERIES = {
    MetricType.MESSAGE_VOLUME: """
        SELECT DATE_TRUNC(:trunc, created_at) AS ts, COUNT(*) AS val
        FROM chat_messages
        WHERE created_at BETWEEN :start AND :end AND role = 'user'
        GROUP BY ts ORDER BY ts
    """,
    MetricType.FAILURE_RATE: """
        SELECT DATE_TRUNC(:trunc, cm.created_at) AS ts,
               COALESCE(100.0 * COUNT(*) FILTER (WHERE ic.is_failure)
                       / NULLIF(COUNT(*), 0), 0) AS val
        FROM chat_messages cm
        LEFT JOIN intent_classifications ic ON ic.message_id = cm.id
        WHERE cm.created_at BETWEEN :start AND :end AND cm.role = 'user'
        GROUP BY ts ORDER BY ts
    """,
    MetricType.ENGAGEMENT_RATE: """
        WITH s AS (
            SELECT DATE_TRUNC(:trunc, cs.created_at) AS ts,
                   COUNT(cm.id) FILTER (WHERE cm.role = 'user') AS turns
            FROM chat_sessions cs
            LEFT JOIN chat_messages cm ON cm.session_id = cs.session_id
            WHERE cs.created_at BETWEEN :start AND :end
            GROUP BY ts, cs.session_id
        )
        SELECT ts, COALESCE(100.0 * COUNT(*) FILTER (WHERE turns >= 2)
                           / NULLIF(COUNT(*), 0), 0) AS val
        FROM s GROUP BY ts ORDER BY ts
    """,
    MetricType.UNIQUE_USERS: """
        SELECT DATE_TRUNC(:trunc, created_at) AS ts, COUNT(DISTINCT user_id) AS val
        FROM chat_sessions
        WHERE created_at BETWEEN :start AND :end
        GROUP BY ts ORDER BY ts
    """,
}

_TRUNC_MAP = {
    AggregationPeriod.HOUR: "hour",
    AggregationPeriod.DAY: "day",
    AggregationPeriod.WEEK: "week",
    AggregationPeriod.MONTH: "month",
}


async def _fetch_series(
    metric: MetricType, params: TrendAnalyzerInput
) -> list[tuple[datetime, float]]:
    from sqlalchemy import text

    sql = _METRIC_QUERIES.get(metric, _METRIC_QUERIES[MetricType.MESSAGE_VOLUME])
    trunc = _TRUNC_MAP[params.aggregation_period]

    async with get_db() as db:
        rows = await db.execute(
            text(sql),
            {
                "trunc": trunc,
                "start": params.time_range.start,
                "end": params.time_range.end,
            },
        )
        return [(row.ts, float(row.val)) for row in rows.fetchall()]