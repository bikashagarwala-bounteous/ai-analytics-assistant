"""
GET  /report          — list saved reports
GET  /report/{id}     — get a specific report
GET  /dashboard       — metrics snapshot for the Streamlit dashboard
POST /feedback        — submit user feedback on a message
POST /optimize        — run the prompt optimizer via MCP (bypasses LangGraph router)
"""

import uuid
from datetime import datetime, timezone

from fastapi import APIRouter, HTTPException, Query
from sqlalchemy import text

from db.connections import get_db
from services.dashboard_service import get_dashboard_metrics, invalidate_dashboard_cache
from services.mcp_client import call_tool
from schemas.api import (
    ReportListItem,
    ReportDetail,
    FeedbackRequest,
    FeedbackResponse,
    DashboardMetrics,
    OptimizeRequest,
)
from core.logging import get_logger

router = APIRouter(tags=["reports"])
logger = get_logger(__name__)



@router.get("/report", response_model=list[ReportListItem])
async def list_reports(
    limit: int = Query(default=20, le=100),
    offset: int = Query(default=0, ge=0),
):
    """List saved analysis reports, newest first."""
    async with get_db() as db:
        rows = await db.execute(text("""
            SELECT report_id, title, summary, created_at, run_id
            FROM reports
            ORDER BY created_at DESC
            LIMIT :limit OFFSET :offset
        """), {"limit": limit, "offset": offset})
        return [
            ReportListItem(
                report_id=r.report_id,
                title=r.title,
                summary=r.summary[:200],
                created_at=r.created_at,
                run_id=r.run_id,
            )
            for r in rows.fetchall()
        ]


@router.get("/report/{report_id}", response_model=ReportDetail)
async def get_report(report_id: str):
    """Get the full content of a saved report."""
    async with get_db() as db:
        row = await db.execute(text("""
            SELECT report_id, title, summary, content_json,
                   created_at, run_id, session_id
            FROM reports
            WHERE report_id = :id
        """), {"id": report_id})
        result = row.fetchone()

    if not result:
        raise HTTPException(status_code=404, detail="Report not found")

    import json
    content = result.content_json
    if isinstance(content, str):
        content = json.loads(content)

    return ReportDetail(
        report_id=result.report_id,
        title=result.title,
        summary=result.summary,
        content=content,
        created_at=result.created_at,
        run_id=result.run_id,
        session_id=result.session_id,
    )


@router.get("/dashboard", response_model=DashboardMetrics)
async def dashboard(
    days: int = Query(default=7, ge=1, le=90),
    force: bool = Query(default=False),
):
    """
    Return aggregated metrics for the Streamlit dashboard.
    Pass force=true to bypass the Redis cache (used by the Refresh button).
    """
    data = await get_dashboard_metrics(days, force=force)
    return DashboardMetrics(**data)


@router.post("/feedback", response_model=FeedbackResponse)
async def submit_feedback(request: FeedbackRequest):
    """
    Submit user feedback for a bot message.
    Routes through the MCP server's feedback tool so ratings flow
    into LangSmith and the analytics pipeline.
    """
    result = await call_tool("submit_feedback", {
        "session_id": request.session_id,
        "message_id": request.message_id,
        "rating": request.rating,
        "sentiment": request.sentiment,
        "response_was_helpful": request.response_was_helpful,
        "comment": request.comment,
        "intent_was_correct": request.intent_was_correct,
        "suggested_intent": request.suggested_intent,
    })

    await invalidate_dashboard_cache()

    return FeedbackResponse(
        feedback_id=result.get("feedback_id") or str(uuid.uuid4()),
        recorded_at=result.get("recorded_at") or datetime.now(timezone.utc),
    )


@router.post("/optimize")
async def optimize_prompt(request: OptimizeRequest):
    """
    Run the prompt optimizer directly via the MCP improve_prompt tool.
    Bypasses the LangGraph pipeline — no router classification needed.
    """
    result = await call_tool("improve_prompt", {
        "prompt_key": request.prompt_key,
        "current_template": request.current_template,
        "optimization_goal": request.optimization_goal,
        "n_variants": request.n_variants,
        "use_feedback_data": request.use_feedback_data,
    })
    return result