"""
POST /analyze — triggers the full analytics pipeline and returns
structured findings without the conversational wrapper.

Used by the Streamlit dashboard's "Run Analysis" button and by
automated scheduled analysis jobs.
"""

import time

from fastapi import APIRouter, HTTPException
from sqlalchemy import text

from agents.graph import run_analytics_pipeline
from services.session_service import get_or_create_session
from services.dashboard_service import invalidate_dashboard_cache
from db.connections import get_db
from schemas.api import AnalyzeRequest, AnalyzeResponse, ToolCallSummary
from core.logging import get_logger

router = APIRouter(prefix="/analyze", tags=["analyze"])
logger = get_logger(__name__)


@router.post("", response_model=AnalyzeResponse)
async def analyze(request: AnalyzeRequest):
    """
    Run the analytics pipeline for a query and return structured results.

    Unlike /chat, this returns the raw analyst findings alongside
    the summary — useful for the dashboard and for building reports.
    """
    t0 = time.perf_counter()
    session_id = await get_or_create_session(request.session_id)

    state = await run_analytics_pipeline(request.query, session_id)

    if state.status == "failed":
        raise HTTPException(
            status_code=500,
            detail=state.error or "Pipeline failed",
        )

    findings = state.analyst_output
    tool_summaries = [
        ToolCallSummary(
            tool_name=tc.tool_name,
            duration_ms=tc.duration_ms,
            success=tc.error is None,
            error=tc.error,
        )
        for tc in state.tool_calls
    ]

    await invalidate_dashboard_cache()

    return AnalyzeResponse(
        run_id=state.run_id or "",
        session_id=session_id,
        query=request.query,
        findings=findings.findings if findings else [],
        root_causes=findings.root_causes if findings else [],
        recommended_actions=findings.recommended_actions if findings else [],
        confidence=findings.confidence if findings else 0.0,
        summary=state.summary_output or "",
        tool_calls=tool_summaries,
        status=state.status,
        duration_ms=int((time.perf_counter() - t0) * 1000),
    )


@router.get("/runs")
async def list_runs(limit: int = 20, offset: int = 0):
    """List recent agent runs for the dashboard history panel."""
    async with get_db() as db:
        rows = await db.execute(text("""
            SELECT
                run_id, query, status,
                total_duration_ms, created_at, completed_at
            FROM agent_runs
            ORDER BY created_at DESC
            LIMIT :limit OFFSET :offset
        """), {"limit": limit, "offset": offset})
        runs = [
            {
                "run_id": r.run_id,
                "query": r.query[:100],
                "status": r.status,
                "duration_ms": r.total_duration_ms,
                "created_at": r.created_at.isoformat() if r.created_at else None,
            }
            for r in rows.fetchall()
        ]
    return {"runs": runs, "limit": limit, "offset": offset}


@router.get("/runs/{run_id}")
async def get_run(run_id: str):
    """Get full details for a specific agent run."""
    async with get_db() as db:
        row = await db.execute(text("""
            SELECT * FROM agent_runs WHERE run_id = :run_id
        """), {"run_id": run_id})
        result = row.fetchone()
        if not result:
            raise HTTPException(status_code=404, detail="Run not found")
        return dict(result._mapping)