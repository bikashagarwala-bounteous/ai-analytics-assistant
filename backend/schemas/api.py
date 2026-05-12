"""
Request and response models for all FastAPI routes.
Kept separate from the agent state models so the API contract
can evolve independently of internal agent structures.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field


# ── /chat ─────────────────────────────────────────────────────────────────────

class ChatRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=4000)
    session_id: str | None = None       # If None, a new session is created
    stream: bool = True                 # False returns full response at once


class ChatResponse(BaseModel):
    session_id: str
    run_id: str
    answer: str
    status: Literal["completed", "failed"]
    tool_calls_count: int
    duration_ms: int


# ── /analyze ──────────────────────────────────────────────────────────────────

class AnalyzeRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=4000)
    session_id: str | None = None
    time_range_start: datetime | None = None
    time_range_end: datetime | None = None


class ToolCallSummary(BaseModel):
    tool_name: str
    duration_ms: float
    success: bool
    error: str | None = None


class AnalyzeResponse(BaseModel):
    run_id: str
    session_id: str | None
    query: str
    findings: list[str]
    root_causes: list[str]
    recommended_actions: list[str]
    confidence: float
    summary: str
    tool_calls: list[ToolCallSummary]
    status: str
    duration_ms: int


# ── /report ───────────────────────────────────────────────────────────────────

class ReportListItem(BaseModel):
    report_id: str
    title: str
    summary: str
    created_at: datetime
    run_id: str | None


class ReportDetail(BaseModel):
    report_id: str
    title: str
    summary: str
    content: dict[str, Any]
    created_at: datetime
    run_id: str | None
    session_id: str | None


# ── /sessions ─────────────────────────────────────────────────────────────────

class SessionCreate(BaseModel):
    user_id: str
    channel: str = "web"
    metadata: dict[str, Any] = Field(default_factory=dict)


class SessionResponse(BaseModel):
    session_id: str
    user_id: str
    channel: str
    created_at: datetime


# ── /feedback ─────────────────────────────────────────────────────────────────

class FeedbackRequest(BaseModel):
    session_id: str
    message_id: str
    rating: int = Field(..., ge=1, le=5)
    sentiment: Literal["positive", "negative", "neutral"]
    response_was_helpful: bool
    comment: str | None = Field(default=None, max_length=1000)
    intent_was_correct: bool | None = None
    suggested_intent: str | None = None


class FeedbackResponse(BaseModel):
    feedback_id: str
    recorded_at: datetime


# ── /optimize ─────────────────────────────────────────────────────────────────

class OptimizeRequest(BaseModel):
    prompt_key: str = Field(..., min_length=1, max_length=100)
    current_template: str = Field(..., min_length=1, max_length=8000)
    optimization_goal: str = "user_satisfaction"
    n_variants: int = Field(default=3, ge=1, le=5)
    use_feedback_data: bool = True


# ── /dashboard ────────────────────────────────────────────────────────────────

class DashboardMetrics(BaseModel):
    """Aggregated metrics snapshot for the Streamlit dashboard."""
    period_label: str                       # "Last 7 days"
    total_sessions: int
    total_messages: int
    engagement_rate: float                  # %
    failure_rate: float                     # %
    avg_rating: float | None
    top_intents: list[dict[str, Any]]
    engagement_trend: list[dict[str, Any]]  # [{date, value}]
    failure_trend: list[dict[str, Any]]
    volume_trend: list[dict[str, Any]]
    top_failures: list[dict[str, Any]]
    recent_feedback: list[dict[str, Any]]