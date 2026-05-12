"""
MCP Tools:
  - record_feedback        : Store a user's rating/comment on a bot response
  - get_feedback_analytics : Retrieve aggregated feedback for the dashboard

The feedback data feeds directly into:
  1. The Streamlit analytics dashboard (charts, NPS, helpfulness rate)
  2. Prompt optimization (record_prompt_performance)
  3. LangSmith run scoring (score_run)
  4. Model fine-tuning signals (flagged in PostgreSQL for future use)

Dashboard sync: the backend polls `get_feedback_analytics` on a schedule
and updates the dashboard's Redis cache, keeping charts near-real-time.
"""

import time
import uuid
from datetime import datetime

from sqlalchemy import text

from db.postgres import get_db
from core.langsmith_tracer import score_run
from core.config import settings
from core.logging import get_logger
from schemas import (
    FeedbackRecordInput,
    FeedbackRecordOutput,
    FeedbackAnalyticsInput,
    FeedbackAnalyticsOutput,
    FeedbackDataPoint,
    FeedbackSentiment,
)

logger = get_logger(__name__)


async def record_feedback(params: FeedbackRecordInput) -> FeedbackRecordOutput:
    """
    Persist user feedback for a bot response.

    Side effects:
    - Stores in PostgreSQL `user_feedback` table
    - Updates LangSmith run score if run_id is available
    - Flags low-rated responses (≤ 2) for review
    """
    t0 = time.perf_counter()
    feedback_id = str(uuid.uuid4())
    now = datetime.utcnow()

    async with get_db() as db:
        await db.execute(text("""
            INSERT INTO user_feedback (
                feedback_id, session_id, message_id,
                rating, sentiment, comment,
                response_was_helpful, intent_was_correct, suggested_intent,
                created_at
            ) VALUES (
                :feedback_id, :session_id, :message_id,
                :rating, :sentiment, :comment,
                :helpful, :intent_correct, :suggested_intent,
                :now
            )
        """), {
            "feedback_id": feedback_id,
            "session_id": params.session_id,
            "message_id": params.message_id,
            "rating": params.rating,
            "sentiment": params.sentiment.value,
            "comment": params.comment,
            "helpful": params.response_was_helpful,
            "intent_correct": params.intent_was_correct,
            "suggested_intent": params.suggested_intent,
            "now": now,
        })

        # ── Flag low-rated responses for review ───────────────────────────────
        will_affect_training = False
        if params.rating <= 2 or params.sentiment == FeedbackSentiment.NEGATIVE:
            await db.execute(text("""
                UPDATE chat_messages
                SET flagged_for_review = TRUE
                WHERE id = :msg_id
            """), {"msg_id": params.message_id})
            will_affect_training = True
            logger.info(
                "message_flagged_for_review",
                message_id=params.message_id,
                rating=params.rating,
            )

        # ── Update intent correction if provided ──────────────────────────────
        if params.suggested_intent and not params.intent_was_correct:
            await db.execute(text("""
                UPDATE intent_classifications
                SET corrected_intent = :correct_intent, corrected_at = :now
                WHERE message_id = :msg_id
            """), {
                "correct_intent": params.suggested_intent,
                "msg_id": params.message_id,
                "now": now,
            })

    # ── LangSmith score (fire and forget) ────────────────────────────────────
    normalized_score = (params.rating - 1) / 4.0   # 1-5 → 0.0-1.0
    await score_run(
        run_id=params.message_id,
        score=normalized_score,
        comment=params.comment or "",
    )

    duration_ms = (time.perf_counter() - t0) * 1000
    logger.info(
        "feedback_recorded",
        feedback_id=feedback_id,
        rating=params.rating,
        sentiment=params.sentiment.value,
        will_affect_training=will_affect_training,
        duration_ms=round(duration_ms, 1),
    )

    return FeedbackRecordOutput(
        feedback_id=feedback_id,
        recorded_at=now,
        will_affect_training=will_affect_training,
    )


async def get_feedback_analytics(
    params: FeedbackAnalyticsInput,
) -> FeedbackAnalyticsOutput:
    """
    Return aggregated feedback metrics grouped by the requested dimension.
    Used by the analytics dashboard to render feedback charts.

    group_by options:
    - "day"       : Time series of daily feedback metrics
    - "intent"    : Feedback breakdown per intent category
    - "sentiment" : Distribution across positive/negative/neutral
    - "rating"    : Star rating distribution (1-5)
    """
    t0 = time.perf_counter()

    async with get_db() as db:
        # ── Group-by query dispatch ───────────────────────────────────────────
        if params.group_by == "day":
            data = await _feedback_by_day(db, params)
        elif params.group_by == "intent":
            data = await _feedback_by_intent(db, params)
        elif params.group_by == "sentiment":
            data = await _feedback_by_sentiment(db, params)
        else:  # rating
            data = await _feedback_by_rating(db, params)

        # ── Totals ────────────────────────────────────────────────────────────
        totals = await db.execute(text("""
            SELECT
                COUNT(*)        AS total,
                AVG(rating)     AS avg_rating
            FROM user_feedback
            WHERE created_at BETWEEN :start AND :end
        """), {"start": params.time_range.start, "end": params.time_range.end})
        total_row = totals.fetchone()

        total_count = int(total_row.total or 0)
        avg_rating = float(total_row.avg_rating or 0.0)

        # ── NPS calculation (requires enough data) ────────────────────────────
        nps_score = None
        if total_count >= 10:
            nps_row = await db.execute(text("""
                SELECT
                    COUNT(*) FILTER (WHERE rating >= 4)  AS promoters,
                    COUNT(*) FILTER (WHERE rating <= 2)  AS detractors,
                    COUNT(*)                             AS total
                FROM user_feedback
                WHERE created_at BETWEEN :start AND :end
            """), {"start": params.time_range.start, "end": params.time_range.end})
            nps_data = nps_row.fetchone()
            if nps_data and nps_data.total > 0:
                promoter_pct = nps_data.promoters / nps_data.total * 100
                detractor_pct = nps_data.detractors / nps_data.total * 100
                nps_score = round(promoter_pct - detractor_pct, 1)

    duration_ms = (time.perf_counter() - t0) * 1000
    logger.info(
        "feedback_analytics_complete",
        group_by=params.group_by,
        data_points=len(data),
        total_feedback=total_count,
        duration_ms=round(duration_ms, 1),
    )

    return FeedbackAnalyticsOutput(
        time_range=params.time_range,
        group_by=params.group_by,
        data=data,
        overall_avg_rating=round(avg_rating, 2),
        total_feedback_count=total_count,
        nps_score=nps_score,
    )


# ── Group-by query implementations ───────────────────────────────────────────

async def _feedback_by_day(db, params) -> list[FeedbackDataPoint]:
    rows = await db.execute(text("""
        SELECT
            DATE_TRUNC('day', created_at)::TEXT             AS label,
            COUNT(*)                                         AS count,
            AVG(rating)                                      AS avg_rating,
            100.0 * COUNT(*) FILTER (WHERE response_was_helpful)
                   / NULLIF(COUNT(*), 0)                    AS helpfulness_rate,
            100.0 * COUNT(*) FILTER (WHERE sentiment = 'negative')
                   / NULLIF(COUNT(*), 0)                    AS negative_rate
        FROM user_feedback
        WHERE created_at BETWEEN :start AND :end
        GROUP BY 1
        ORDER BY 1
    """), {"start": params.time_range.start, "end": params.time_range.end})
    return _rows_to_datapoints(rows.fetchall(), params.min_count)


async def _feedback_by_intent(db, params) -> list[FeedbackDataPoint]:
    rows = await db.execute(text("""
        SELECT
            ic.intent                                        AS label,
            COUNT(uf.feedback_id)                            AS count,
            AVG(uf.rating)                                   AS avg_rating,
            100.0 * COUNT(*) FILTER (WHERE uf.response_was_helpful)
                   / NULLIF(COUNT(uf.feedback_id), 0)       AS helpfulness_rate,
            100.0 * COUNT(*) FILTER (WHERE uf.sentiment = 'negative')
                   / NULLIF(COUNT(uf.feedback_id), 0)       AS negative_rate
        FROM user_feedback uf
        JOIN chat_messages cm ON cm.id = uf.message_id
        JOIN intent_classifications ic ON ic.message_id = cm.id
        WHERE uf.created_at BETWEEN :start AND :end
          AND ic.intent IS NOT NULL
        GROUP BY ic.intent
        HAVING COUNT(*) >= :min_count
        ORDER BY count DESC
        LIMIT 20
    """), {
        "start": params.time_range.start,
        "end": params.time_range.end,
        "min_count": params.min_count,
    })
    return _rows_to_datapoints(rows.fetchall(), params.min_count)


async def _feedback_by_sentiment(db, params) -> list[FeedbackDataPoint]:
    rows = await db.execute(text("""
        SELECT
            sentiment                                        AS label,
            COUNT(*)                                         AS count,
            AVG(rating)                                      AS avg_rating,
            100.0 * COUNT(*) FILTER (WHERE response_was_helpful)
                   / NULLIF(COUNT(*), 0)                    AS helpfulness_rate,
            100.0 * COUNT(*) FILTER (WHERE sentiment = 'negative')
                   / NULLIF(COUNT(*), 0)                    AS negative_rate
        FROM user_feedback
        WHERE created_at BETWEEN :start AND :end
        GROUP BY sentiment
        ORDER BY count DESC
    """), {"start": params.time_range.start, "end": params.time_range.end})
    return _rows_to_datapoints(rows.fetchall(), 0)


async def _feedback_by_rating(db, params) -> list[FeedbackDataPoint]:
    rows = await db.execute(text("""
        SELECT
            rating::TEXT                                     AS label,
            COUNT(*)                                         AS count,
            AVG(rating)                                      AS avg_rating,
            100.0 * COUNT(*) FILTER (WHERE response_was_helpful)
                   / NULLIF(COUNT(*), 0)                    AS helpfulness_rate,
            100.0 * COUNT(*) FILTER (WHERE sentiment = 'negative')
                   / NULLIF(COUNT(*), 0)                    AS negative_rate
        FROM user_feedback
        WHERE created_at BETWEEN :start AND :end
        GROUP BY rating
        ORDER BY rating
    """), {"start": params.time_range.start, "end": params.time_range.end})
    return _rows_to_datapoints(rows.fetchall(), 0)


def _rows_to_datapoints(rows, min_count: int) -> list[FeedbackDataPoint]:
    return [
        FeedbackDataPoint(
            label=str(row.label),
            count=int(row.count),
            avg_rating=round(float(row.avg_rating or 0), 2),
            helpfulness_rate=round(float(row.helpfulness_rate or 0), 2),
            negative_rate=round(float(row.negative_rate or 0), 2),
        )
        for row in rows
        if int(row.count) >= min_count
    ]