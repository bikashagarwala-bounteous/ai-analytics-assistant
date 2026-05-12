"""
shared state for the LangGraph pipeline.
Flow: query → AnalystNode (tool calls, findings) → SummaryNode (text output) → END
"""

from __future__ import annotations

from datetime import datetime
from typing import Annotated, Any
import operator

from pydantic import BaseModel, Field
from langgraph.graph import MessagesState


# ── Tool call record ──────────────────────────────────────────────────────────

class ToolCallRecord(BaseModel):
    """Records a single MCP tool invocation for logging and observability."""
    tool_name: str
    input: dict[str, Any]
    output: dict[str, Any] | None = None
    error: str | None = None
    duration_ms: float = 0.0
    called_at: datetime = Field(default_factory=datetime.utcnow)


# ── Analyst findings ──────────────────────────────────────────────────────────

class AnalystFindings(BaseModel):
    """
    Structured output from the Analyst Agent.
    The Summary Agent receives this and turns it into a human-readable response.
    """
    query: str
    findings: list[str] = Field(
        default_factory=list,
        description="List of factual findings drawn from tool results",
    )
    evidence: list[dict[str, Any]] = Field(
        default_factory=list,
        description="Raw data supporting each finding (metric values, examples, etc.)",
    )
    anomalies_found: list[dict[str, Any]] = Field(default_factory=list)
    trends_identified: list[dict[str, Any]] = Field(default_factory=list)
    root_causes: list[str] = Field(
        default_factory=list,
        description="Hypothesised root causes if anomalies or drops were detected",
    )
    confidence: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Overall confidence in the findings based on data quality",
    )
    recommended_actions: list[str] = Field(default_factory=list)
    data_coverage: str = ""     # e.g. "7 days of data, 1,240 conversations"
    insufficient_data: bool = False


# ── Main graph state ──────────────────────────────────────────────────────────

class AgentState(BaseModel):
    """
    The full state object for the analytics pipeline.

    LangGraph uses this as a TypedDict-like container.
    The `Annotated[list, operator.add]` on tool_calls means LangGraph
    appends to the list rather than replacing it when a node writes to it.
    """

    # Input — set at pipeline entry, never modified
    query: str
    session_id: str | None = None
    run_id: str | None = None           # AgentRun.run_id, set before graph starts

    # Analyst node outputs
    analyst_output: AnalystFindings | None = None
    tool_calls: Annotated[list[ToolCallRecord], operator.add] = Field(default_factory=list)

    # Summary node outputs
    summary_output: str = ""

    # Routing / control
    should_use_tot: bool = False        # Set by router if question is complex
    analyst_retries: int = 0
    error: str | None = None
    status: str = "pending"             # pending | running | completed | failed

    # Timing
    started_at: datetime = Field(default_factory=datetime.utcnow)
    analyst_completed_at: datetime | None = None
    summary_completed_at: datetime | None = None

    class Config:
        arbitrary_types_allowed = True