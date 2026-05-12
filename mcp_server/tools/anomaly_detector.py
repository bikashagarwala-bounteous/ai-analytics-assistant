"""
MCP Tool: detect_anomalies

Statistical anomaly detection on chatbot metrics.
Supports Z-score, IQR, and Isolation Forest methods.

The LLM decides:
  - Which metric to check
  - What sensitivity threshold to use
  - Which method fits the data characteristics

The tool executes deterministically and returns structured anomalies
with severity classifications and possible causes.
"""

import time
from datetime import datetime

import numpy as np

from db.postgres import get_db
from core.logging import get_logger
from schemas import (
    AnomalyDetectorInput,
    AnomalyDetectorOutput,
    Anomaly,
    MetricType,
    SeverityLevel,
    TrendDirection,
)

logger = get_logger(__name__)


async def detect_anomalies(params: AnomalyDetectorInput) -> AnomalyDetectorOutput:
    """
    Detect statistical anomalies in the specified metric.

    Returns anomalies with:
    - Observed vs expected values
    - Severity classification
    - Human-readable description
    - Possible causes (based on heuristics)
    """
    t0 = time.perf_counter()

    # ── Fetch time series data ────────────────────────────────────────────────
    time_series = await _fetch_metric_series(params)

    if len(time_series) < 5:
        logger.warning(
            "anomaly_detection_insufficient_data",
            points=len(time_series),
            metric=params.metric.value,
        )
        return AnomalyDetectorOutput(
            metric=params.metric,
            time_range=params.time_range,
            method=params.method,
            anomalies=[],
            total_data_points=len(time_series),
            anomaly_rate=0.0,
            baseline_mean=0.0,
            baseline_std=0.0,
            severity_summary={},
        )

    timestamps = [row[0] for row in time_series]
    values = np.array([float(row[1]) for row in time_series])

    # ── Detect anomalies ──────────────────────────────────────────────────────
    if params.method == "zscore":
        anomaly_flags, z_scores = _zscore_detection(values, params.sensitivity)
    elif params.method == "iqr":
        anomaly_flags, z_scores = _iqr_detection(values, params.sensitivity)
    else:  # isolation_forest
        anomaly_flags, z_scores = _isolation_forest_detection(values)

    baseline_mean = float(np.mean(values))
    baseline_std = float(np.std(values))

    # ── Build anomaly objects ─────────────────────────────────────────────────
    anomalies: list[Anomaly] = []
    for i, (ts, val, is_anomaly, z) in enumerate(
        zip(timestamps, values, anomaly_flags, z_scores)
    ):
        if not is_anomaly:
            continue

        deviation = float(val - baseline_mean)
        severity = _classify_severity(abs(z))
        description = _describe_anomaly(params.metric, val, baseline_mean, deviation)
        causes = _infer_causes(params.metric, val, baseline_mean, deviation)

        anomalies.append(Anomaly(
            timestamp=ts,
            metric=params.metric,
            observed_value=round(float(val), 4),
            expected_value=round(baseline_mean, 4),
            deviation=round(deviation, 4),
            z_score=round(float(z), 3),
            severity=severity,
            description=description,
            possible_causes=causes,
        ))

    severity_order = {
        SeverityLevel.CRITICAL: 0,
        SeverityLevel.HIGH: 1,
        SeverityLevel.MEDIUM: 2,
        SeverityLevel.LOW: 3,
    }
    anomalies.sort(key=lambda a: (severity_order[a.severity], a.timestamp))

    anomaly_rate = len(anomalies) / len(values) if values.size > 0 else 0.0
    severity_summary = {level.value: 0 for level in SeverityLevel}
    for a in anomalies:
        severity_summary[a.severity.value] += 1

    duration_ms = (time.perf_counter() - t0) * 1000

    logger.info(
        "anomaly_detection_complete",
        metric=params.metric.value,
        method=params.method,
        total_points=len(time_series),
        anomalies_found=len(anomalies),
        duration_ms=round(duration_ms, 1),
    )

    return AnomalyDetectorOutput(
        metric=params.metric,
        time_range=params.time_range,
        method=params.method,
        anomalies=anomalies,
        total_data_points=len(time_series),
        anomaly_rate=round(anomaly_rate, 4),
        baseline_mean=round(baseline_mean, 4),
        baseline_std=round(baseline_std, 4),
        severity_summary=severity_summary,
    )


# ── Detection algorithms ──────────────────────────────────────────────────────

def _zscore_detection(
    values: np.ndarray, threshold: float
) -> tuple[list[bool], list[float]]:
    """
    Z-score method: flag points where |z| > threshold.
    Uses median absolute deviation (MAD) for robustness against outliers.
    """
    median = np.median(values)
    mad = np.median(np.abs(values - median))
    if mad == 0:
        mad = np.std(values) or 1.0

    # Modified Z-score (robust to outliers)
    z_scores = 0.6745 * (values - median) / mad
    flags = [abs(z) > threshold for z in z_scores]
    return flags, z_scores.tolist()


def _iqr_detection(
    values: np.ndarray, sensitivity: float
) -> tuple[list[bool], list[float]]:
    """
    IQR method: flag points outside [Q1 - k*IQR, Q3 + k*IQR].
    k is derived from sensitivity (lower sensitivity → larger k → fewer flags).
    """
    q1, q3 = np.percentile(values, [25, 75])
    iqr = q3 - q1
    k = 3.0 / sensitivity     # sensitivity=2 → k=1.5 (standard), sensitivity=3 → k=1.0

    lower = q1 - k * iqr
    upper = q3 + k * iqr

    mean = float(np.mean(values))
    std = float(np.std(values)) or 1.0
    z_scores = ((values - mean) / std).tolist()
    flags = [(v < lower or v > upper) for v in values]
    return flags, z_scores


def _isolation_forest_detection(
    values: np.ndarray,
) -> tuple[list[bool], list[float]]:
    """
    Isolation Forest — better for non-Gaussian distributions.
    Falls back to Z-score if sklearn is unavailable.
    """
    try:
        from sklearn.ensemble import IsolationForest

        X = values.reshape(-1, 1)
        clf = IsolationForest(contamination=0.1, random_state=42)
        preds = clf.fit_predict(X)
        scores = clf.score_samples(X)

        flags = [p == -1 for p in preds]
        # Normalise scores to pseudo z-scores for severity classification
        z_scores = ((-scores - (-scores).mean()) / max((-scores).std(), 1e-8)).tolist()
        return flags, z_scores

    except ImportError:
        logger.warning("sklearn_not_available_falling_back_to_zscore")
        return _zscore_detection(values, threshold=2.0)


# ── Classification helpers ────────────────────────────────────────────────────

def _classify_severity(abs_z: float) -> SeverityLevel:
    if abs_z >= 4.0:
        return SeverityLevel.CRITICAL
    elif abs_z >= 3.0:
        return SeverityLevel.HIGH
    elif abs_z >= 2.0:
        return SeverityLevel.MEDIUM
    else:
        return SeverityLevel.LOW


def _describe_anomaly(
    metric: MetricType, value: float, mean: float, deviation: float
) -> str:
    direction = "spike" if deviation > 0 else "drop"
    pct = abs(deviation / mean * 100) if mean != 0 else 0

    metric_labels = {
        MetricType.ENGAGEMENT_RATE: "engagement rate",
        MetricType.FAILURE_RATE: "failure rate",
        MetricType.RESOLUTION_RATE: "resolution rate",
        MetricType.MESSAGE_VOLUME: "message volume",
        MetricType.AVG_TURNS: "average conversation turns",
        MetricType.SESSION_DURATION: "session duration",
        MetricType.UNIQUE_USERS: "unique user count",
        MetricType.INTENT_DISTRIBUTION: "intent distribution",
    }
    label = metric_labels.get(metric, metric.value)
    return (
        f"Unusual {direction} in {label}: observed {value:.2f} "
        f"vs baseline {mean:.2f} ({pct:.1f}% deviation)"
    )


def _infer_causes(
    metric: MetricType, value: float, mean: float, deviation: float
) -> list[str]:
    """
    Heuristic possible-cause suggestions based on metric type and direction.
    These are starting points for the analyst agent's investigation.
    """
    is_spike = deviation > 0
    causes: list[str] = []

    if metric == MetricType.FAILURE_RATE:
        if is_spike:
            causes = [
                "New intent categories not covered by training data",
                "Model update or regression introduced",
                "Unusual user query patterns (e.g., new use case)",
                "Data quality issue in intent classifier",
            ]
        else:
            causes = [
                "Recent model improvement or retraining",
                "New FAQ content added that resolved common failures",
            ]

    elif metric == MetricType.ENGAGEMENT_RATE:
        if not is_spike:
            causes = [
                "Increased bot-like or test traffic",
                "Users abandoning after first message (poor first-turn response)",
                "UI/UX change reducing conversation depth",
                "Onboarding flow change",
            ]
        else:
            causes = [
                "Viral content or campaign driving engagement",
                "New feature launch attracting curious users",
            ]

    elif metric == MetricType.MESSAGE_VOLUME:
        if is_spike:
            causes = [
                "Marketing campaign or product launch",
                "External event driving users to the chatbot",
                "Load test or automated traffic",
            ]
        else:
            causes = [
                "Outage or downtime period",
                "Seasonal dip",
                "Competitor product launch",
            ]

    elif metric == MetricType.SESSION_DURATION:
        if is_spike:
            causes = [
                "Users struggling to find answers (repeated attempts)",
                "Complex queries requiring multi-turn resolution",
                "Slow response latency degrading experience",
            ]
        else:
            causes = [
                "Improved first-turn resolution",
                "Users abandoning quickly",
            ]

    return causes


# ── Data fetcher ──────────────────────────────────────────────────────────────

async def _fetch_metric_series(
    params: AnomalyDetectorInput,
) -> list[tuple[datetime, float]]:
    """Fetch hourly time series for the requested metric from PostgreSQL."""

    # Metric-to-query mapping
    queries = {
        MetricType.MESSAGE_VOLUME: """
            SELECT DATE_TRUNC('hour', created_at) AS ts, COUNT(*) AS val
            FROM chat_messages
            WHERE created_at BETWEEN :start AND :end AND role = 'user'
            GROUP BY ts ORDER BY ts
        """,
        MetricType.FAILURE_RATE: """
            SELECT DATE_TRUNC('hour', cm.created_at) AS ts,
                   COALESCE(
                       100.0 * COUNT(*) FILTER (WHERE ic.is_failure)
                            / NULLIF(COUNT(*), 0), 0
                   ) AS val
            FROM chat_messages cm
            LEFT JOIN intent_classifications ic ON ic.message_id = cm.id
            WHERE cm.created_at BETWEEN :start AND :end AND cm.role = 'user'
            GROUP BY ts ORDER BY ts
        """,
        MetricType.ENGAGEMENT_RATE: """
            WITH s AS (
                SELECT DATE_TRUNC('hour', cs.created_at) AS ts,
                       COUNT(cm.id) FILTER (WHERE cm.role = 'user') AS turns
                FROM chat_sessions cs
                LEFT JOIN chat_messages cm ON cm.session_id = cs.session_id
                WHERE cs.created_at BETWEEN :start AND :end
                GROUP BY ts, cs.session_id
            )
            SELECT ts,
                   COALESCE(
                       100.0 * COUNT(*) FILTER (WHERE turns >= 2)
                            / NULLIF(COUNT(*), 0), 0
                   ) AS val
            FROM s GROUP BY ts ORDER BY ts
        """,
        MetricType.UNIQUE_USERS: """
            SELECT DATE_TRUNC('hour', created_at) AS ts,
                   COUNT(DISTINCT user_id) AS val
            FROM chat_sessions
            WHERE created_at BETWEEN :start AND :end
            GROUP BY ts ORDER BY ts
        """,
    }

    sql = queries.get(params.metric)
    if not sql:
        # Fallback: message volume as proxy
        sql = queries[MetricType.MESSAGE_VOLUME]

    from sqlalchemy import text
    async with get_db() as db:
        rows = await db.execute(
            text(sql),
            {"start": params.time_range.start, "end": params.time_range.end},
        )
        return [(row.ts, float(row.val)) for row in rows.fetchall()]