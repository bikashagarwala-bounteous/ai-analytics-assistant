"""
SQLAlchemy ORM models. Every table defined in init.sql has a matching class here.
Used by the backend for inserts and simple lookups.
Complex analytics queries use raw SQL in the MCP tools for performance.
"""

import uuid
from datetime import datetime

from sqlalchemy import (
    Boolean, CheckConstraint, Float, ForeignKey,
    Integer, SmallInteger, String, Text, Index, TIMESTAMP,
)
from sqlalchemy.dialects.postgresql import JSONB, UUID

TIMESTAMPTZ = TIMESTAMP(timezone=True)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from db.connections import Base


def _uuid() -> str:
    return str(uuid.uuid4())


class ChatSession(Base):
    __tablename__ = "chat_sessions"

    session_id: Mapped[str] = mapped_column(
        UUID(as_uuid=False), primary_key=True, default=_uuid
    )
    user_id: Mapped[str] = mapped_column(String, nullable=False)
    channel: Mapped[str] = mapped_column(String, nullable=False, default="web")
    started_at: Mapped[datetime] = mapped_column(TIMESTAMPTZ, nullable=False, default=datetime.utcnow)
    ended_at: Mapped[datetime | None] = mapped_column(TIMESTAMPTZ, nullable=True)
    resolved: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    metadata_: Mapped[dict] = mapped_column("metadata", JSONB, nullable=False, default=dict)
    created_at: Mapped[datetime] = mapped_column(TIMESTAMPTZ, nullable=False, default=datetime.utcnow)

    messages: Mapped[list["ChatMessage"]] = relationship(
        back_populates="session", cascade="all, delete-orphan"
    )
    feedback: Mapped[list["UserFeedback"]] = relationship(
        back_populates="session", cascade="all, delete-orphan"
    )
    agent_runs: Mapped[list["AgentRun"]] = relationship(back_populates="session")


class ChatMessage(Base):
    __tablename__ = "chat_messages"

    id: Mapped[str] = mapped_column(UUID(as_uuid=False), primary_key=True, default=_uuid)
    session_id: Mapped[str] = mapped_column(
        UUID(as_uuid=False), ForeignKey("chat_sessions.session_id", ondelete="CASCADE"),
        nullable=False,
    )
    role: Mapped[str] = mapped_column(String, nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    flagged_for_review: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    langsmith_run_id: Mapped[str | None] = mapped_column(String, nullable=True)
    latency_ms: Mapped[int | None] = mapped_column(Integer, nullable=True)
    token_count: Mapped[int | None] = mapped_column(Integer, nullable=True)
    created_at: Mapped[datetime] = mapped_column(TIMESTAMPTZ, nullable=False, default=datetime.utcnow)

    session: Mapped["ChatSession"] = relationship(back_populates="messages")
    intent: Mapped["IntentClassification | None"] = relationship(
        back_populates="message", cascade="all, delete-orphan", uselist=False
    )
    feedback: Mapped[list["UserFeedback"]] = relationship(
        back_populates="message", cascade="all, delete-orphan"
    )

    __table_args__ = (
        CheckConstraint("role IN ('user', 'assistant', 'system')", name="ck_message_role"),
    )


class IntentClassification(Base):
    __tablename__ = "intent_classifications"

    id: Mapped[str] = mapped_column(UUID(as_uuid=False), primary_key=True, default=_uuid)
    message_id: Mapped[str] = mapped_column(
        UUID(as_uuid=False), ForeignKey("chat_messages.id", ondelete="CASCADE"),
        nullable=False, unique=True,
    )
    intent: Mapped[str] = mapped_column(String, nullable=False)
    confidence: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    is_failure: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    corrected_intent: Mapped[str | None] = mapped_column(String, nullable=True)
    corrected_at: Mapped[datetime | None] = mapped_column(TIMESTAMPTZ, nullable=True)
    raw_classifier_output: Mapped[dict] = mapped_column(JSONB, nullable=False, default=dict)
    created_at: Mapped[datetime] = mapped_column(TIMESTAMPTZ, nullable=False, default=datetime.utcnow)

    message: Mapped["ChatMessage"] = relationship(back_populates="intent")


class UserFeedback(Base):
    __tablename__ = "user_feedback"

    feedback_id: Mapped[str] = mapped_column(UUID(as_uuid=False), primary_key=True, default=_uuid)
    session_id: Mapped[str] = mapped_column(
        UUID(as_uuid=False), ForeignKey("chat_sessions.session_id", ondelete="CASCADE"),
        nullable=False,
    )
    message_id: Mapped[str] = mapped_column(
        UUID(as_uuid=False), ForeignKey("chat_messages.id", ondelete="CASCADE"),
        nullable=False,
    )
    rating: Mapped[int] = mapped_column(SmallInteger, nullable=False)
    sentiment: Mapped[str] = mapped_column(String, nullable=False)
    comment: Mapped[str | None] = mapped_column(Text, nullable=True)
    response_was_helpful: Mapped[bool] = mapped_column(Boolean, nullable=False)
    intent_was_correct: Mapped[bool | None] = mapped_column(Boolean, nullable=True)
    suggested_intent: Mapped[str | None] = mapped_column(String, nullable=True)
    created_at: Mapped[datetime] = mapped_column(TIMESTAMPTZ, nullable=False, default=datetime.utcnow)

    session: Mapped["ChatSession"] = relationship(back_populates="feedback")
    message: Mapped["ChatMessage"] = relationship(back_populates="feedback")

    __table_args__ = (
        CheckConstraint("rating BETWEEN 1 AND 5", name="ck_feedback_rating"),
        CheckConstraint("sentiment IN ('positive', 'negative', 'neutral')", name="ck_feedback_sentiment"),
    )


class AgentRun(Base):
    __tablename__ = "agent_runs"

    run_id: Mapped[str] = mapped_column(UUID(as_uuid=False), primary_key=True, default=_uuid)
    session_id: Mapped[str | None] = mapped_column(
        UUID(as_uuid=False), ForeignKey("chat_sessions.session_id", ondelete="SET NULL"),
        nullable=True,
    )
    query: Mapped[str] = mapped_column(Text, nullable=False)
    analyst_output: Mapped[dict | None] = mapped_column(JSONB, nullable=True)
    summary_output: Mapped[str | None] = mapped_column(Text, nullable=True)
    tool_calls: Mapped[list] = mapped_column(JSONB, nullable=False, default=list)
    status: Mapped[str] = mapped_column(String, nullable=False, default="pending")
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)
    total_duration_ms: Mapped[int | None] = mapped_column(Integer, nullable=True)
    analyst_duration_ms: Mapped[int | None] = mapped_column(Integer, nullable=True)
    summary_duration_ms: Mapped[int | None] = mapped_column(Integer, nullable=True)
    langsmith_run_id: Mapped[str | None] = mapped_column(String, nullable=True)
    created_at: Mapped[datetime] = mapped_column(TIMESTAMPTZ, nullable=False, default=datetime.utcnow)
    completed_at: Mapped[datetime | None] = mapped_column(TIMESTAMPTZ, nullable=True)

    session: Mapped["ChatSession | None"] = relationship(back_populates="agent_runs")
    report: Mapped["Report | None"] = relationship(back_populates="agent_run", uselist=False)

    __table_args__ = (
        CheckConstraint(
            "status IN ('pending', 'running', 'completed', 'failed')",
            name="ck_agent_run_status",
        ),
    )


class Report(Base):
    __tablename__ = "reports"

    report_id: Mapped[str] = mapped_column(UUID(as_uuid=False), primary_key=True, default=_uuid)
    session_id: Mapped[str | None] = mapped_column(
        UUID(as_uuid=False), ForeignKey("chat_sessions.session_id", ondelete="SET NULL"),
        nullable=True,
    )
    run_id: Mapped[str | None] = mapped_column(
        UUID(as_uuid=False), ForeignKey("agent_runs.run_id", ondelete="SET NULL"),
        nullable=True,
    )
    title: Mapped[str] = mapped_column(String, nullable=False)
    summary: Mapped[str] = mapped_column(Text, nullable=False)
    content_json: Mapped[dict] = mapped_column(JSONB, nullable=False, default=dict)
    time_range_start: Mapped[datetime | None] = mapped_column(TIMESTAMPTZ, nullable=True)
    time_range_end: Mapped[datetime | None] = mapped_column(TIMESTAMPTZ, nullable=True)
    created_at: Mapped[datetime] = mapped_column(TIMESTAMPTZ, nullable=False, default=datetime.utcnow)

    agent_run: Mapped["AgentRun | None"] = relationship(back_populates="report")


class PromptVariant(Base):
    __tablename__ = "prompt_variants"

    variant_id: Mapped[str] = mapped_column(String, primary_key=True)
    prompt_key: Mapped[str] = mapped_column(String, nullable=False)
    template: Mapped[str] = mapped_column(Text, nullable=False)
    description: Mapped[str] = mapped_column(String, nullable=False, default="")
    is_active: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    created_at: Mapped[datetime] = mapped_column(TIMESTAMPTZ, nullable=False, default=datetime.utcnow)

    performances: Mapped[list["PromptPerformance"]] = relationship(back_populates="variant")


class PromptPerformance(Base):
    __tablename__ = "prompt_performances"

    id: Mapped[str] = mapped_column(UUID(as_uuid=False), primary_key=True, default=_uuid)
    prompt_key: Mapped[str] = mapped_column(String, nullable=False)
    variant_id: Mapped[str | None] = mapped_column(
        String, ForeignKey("prompt_variants.variant_id", ondelete="SET NULL"), nullable=True
    )
    score: Mapped[float] = mapped_column(Float, nullable=False)
    feedback_text: Mapped[str | None] = mapped_column(Text, nullable=True)
    latency_ms: Mapped[float | None] = mapped_column(Float, nullable=True)
    tokens_used: Mapped[int | None] = mapped_column(Integer, nullable=True)
    recorded_at: Mapped[datetime] = mapped_column(TIMESTAMPTZ, nullable=False, default=datetime.utcnow)

    variant: Mapped["PromptVariant | None"] = relationship(back_populates="performances")