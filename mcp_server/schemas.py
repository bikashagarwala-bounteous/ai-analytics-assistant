"""
Pydantic v2 models for every MCP tool — inputs, outputs, and shared types.
These are the contracts between the LLM (tool caller) and the MCP tools.
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator, model_validator


# ── Shared primitives ─────────────────────────────────────────────────────────

class TimeRange(BaseModel):
    """Inclusive time window for filtering queries."""
    start: datetime
    end: datetime

    @model_validator(mode="after")
    def check_order(self) -> "TimeRange":
        if self.start >= self.end:
            raise ValueError("start must be before end")
        return self


class SeverityLevel(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class TrendDirection(str, Enum):
    INCREASING = "increasing"
    DECREASING = "decreasing"
    STABLE = "stable"
    VOLATILE = "volatile"


# ── Tool: vector_search ───────────────────────────────────────────────────────

class VectorSearchInput(BaseModel):
    """
    LLM-facing input for semantic conversation search.
    The LLM formulates the query based on user intent.
    """
    query: str = Field(
        ...,
        description="Natural language query to search against conversation history",
        min_length=3,
        max_length=1000,
    )
    limit: int = Field(
        default=10,
        ge=1,
        le=50,
        description="Maximum number of similar conversations to return",
    )
    time_range: TimeRange | None = Field(
        default=None,
        description="Optional time window to restrict results",
    )
    intent_filter: str | None = Field(
        default=None,
        description="Filter results to a specific intent category",
    )
    min_similarity: float = Field(
        default=0.6,
        ge=0.0,
        le=1.0,
        description="Minimum cosine similarity threshold (0=any, 1=exact)",
    )


class ConversationMatch(BaseModel):
    """A single retrieved conversation with its similarity score."""
    message_id: str
    session_id: str
    content: str
    intent: str | None
    similarity: float
    timestamp: datetime
    metadata: dict[str, Any] = Field(default_factory=dict)


class VectorSearchOutput(BaseModel):
    """Results from a semantic vector search."""
    query: str
    matches: list[ConversationMatch]
    total_found: int
    search_duration_ms: float


# ── Tool: metrics_query ───────────────────────────────────────────────────────

class MetricType(str, Enum):
    ENGAGEMENT_RATE = "engagement_rate"
    FAILURE_RATE = "failure_rate"
    RESOLUTION_RATE = "resolution_rate"
    AVG_TURNS = "avg_turns"
    SESSION_DURATION = "session_duration"
    MESSAGE_VOLUME = "message_volume"
    UNIQUE_USERS = "unique_users"
    INTENT_DISTRIBUTION = "intent_distribution"


class AggregationPeriod(str, Enum):
    HOUR = "hour"
    DAY = "day"
    WEEK = "week"
    MONTH = "month"


class MetricsQueryInput(BaseModel):
    """
    LLM-facing input for structured metrics queries.
    The LLM translates user questions into specific metric requests.
    """
    metrics: list[MetricType] = Field(
        ...,
        description="Which metrics to compute",
        min_length=1,
    )
    time_range: TimeRange = Field(
        ...,
        description="Time window for the metric computation",
    )
    aggregation_period: AggregationPeriod = Field(
        default=AggregationPeriod.DAY,
        description="Bucket size for time-series aggregation",
    )
    compare_with_previous_period: bool = Field(
        default=False,
        description="Also compute the same metrics for the preceding equivalent period",
    )
    filters: dict[str, Any] = Field(
        default_factory=dict,
        description="Key-value filters (e.g. intent='order_tracking', channel='web')",
    )


class MetricDataPoint(BaseModel):
    """Single time-bucketed metric value."""
    timestamp: datetime
    value: float
    label: str | None = None


class MetricResult(BaseModel):
    """Computed result for one metric type."""
    metric: MetricType
    current_value: float
    previous_value: float | None = None
    change_pct: float | None = None
    trend: TrendDirection
    time_series: list[MetricDataPoint]
    unit: str = ""


class MetricsQueryOutput(BaseModel):
    """Aggregated results for all requested metrics."""
    time_range: TimeRange
    aggregation_period: AggregationPeriod
    results: list[MetricResult]
    filters_applied: dict[str, Any]
    query_duration_ms: float


# ── Tool: anomaly_detector ────────────────────────────────────────────────────

class AnomalyDetectorInput(BaseModel):
    """
    LLM-facing input for statistical anomaly detection.
    The LLM specifies which metric and what sensitivity to use.
    """
    metric: MetricType = Field(
        ...,
        description="The metric to scan for anomalies",
    )
    time_range: TimeRange = Field(
        ...,
        description="Time window to scan",
    )
    sensitivity: float = Field(
        default=2.0,
        ge=1.0,
        le=5.0,
        description=(
            "Z-score threshold: 2.0=moderate (default), 3.0=strict, 1.5=sensitive. "
            "Lower = more anomalies detected"
        ),
    )
    method: Literal["zscore", "iqr", "isolation_forest"] = Field(
        default="zscore",
        description="Statistical method for anomaly detection",
    )


class Anomaly(BaseModel):
    """A detected anomaly with context."""
    timestamp: datetime
    metric: MetricType
    observed_value: float
    expected_value: float
    deviation: float                    # Absolute deviation from expected
    z_score: float | None = None
    severity: SeverityLevel
    description: str                    # Human-readable description of the anomaly
    possible_causes: list[str] = Field(default_factory=list)


class AnomalyDetectorOutput(BaseModel):
    """Anomaly detection results."""
    metric: MetricType
    time_range: TimeRange
    method: str
    anomalies: list[Anomaly]
    total_data_points: int
    anomaly_rate: float                 # % of data points that are anomalous
    baseline_mean: float
    baseline_std: float
    severity_summary: dict[str, int]    # {"high": 2, "medium": 5, ...}


# ── Tool: trend_analyzer ──────────────────────────────────────────────────────

class TrendAnalyzerInput(BaseModel):
    """
    LLM-facing input for trend analysis and forecasting.
    """
    metrics: list[MetricType] = Field(
        ...,
        description="Metrics to analyse for trends",
        min_length=1,
    )
    time_range: TimeRange = Field(
        ...,
        description="Historical window to analyse",
    )
    forecast_periods: int = Field(
        default=0,
        ge=0,
        le=30,
        description="Number of future periods to forecast (0 = no forecast)",
    )
    aggregation_period: AggregationPeriod = Field(
        default=AggregationPeriod.DAY,
    )
    include_seasonality: bool = Field(
        default=True,
        description="Detect and account for weekly/monthly seasonality patterns",
    )


class TrendSegment(BaseModel):
    """A continuous period with a consistent trend direction."""
    start: datetime
    end: datetime
    direction: TrendDirection
    slope: float                        # Rate of change per period
    confidence: float                   # 0-1 confidence in this trend


class ForecastPoint(BaseModel):
    timestamp: datetime
    predicted_value: float
    lower_bound: float                  # 80% confidence interval
    upper_bound: float


class TrendResult(BaseModel):
    metric: MetricType
    overall_direction: TrendDirection
    overall_slope: float
    r_squared: float                    # Goodness of fit for the trend line
    segments: list[TrendSegment]
    seasonal_pattern: str | None        # e.g. "weekly peak on Mondays"
    forecast: list[ForecastPoint]
    insight: str                        # LLM-generated natural language insight


class TrendAnalyzerOutput(BaseModel):
    """Multi-metric trend analysis results."""
    time_range: TimeRange
    aggregation_period: AggregationPeriod
    results: list[TrendResult]
    cross_metric_correlations: dict[str, float] = Field(
        default_factory=dict,
        description="Pairwise Pearson correlations between metrics",
    )
    analysis_duration_ms: float


# ── Tool: get_failure_intents ─────────────────────────────────────────────────

class FailureIntentsInput(BaseModel):
    """Retrieve the most frequently failing intents in a time window."""
    time_range: TimeRange
    top_n: int = Field(default=10, ge=1, le=50)
    min_occurrences: int = Field(
        default=1,
        description="Minimum failure count to include an intent. Default 1 for small datasets.",
    )
    include_examples: bool = Field(
        default=True,
        description="Include sample failed messages per intent",
    )


class FailureIntent(BaseModel):
    intent: str
    failure_count: int
    total_count: int
    failure_rate: float
    avg_confidence: float
    trend: TrendDirection
    sample_messages: list[str] = Field(default_factory=list)


class FailureIntentsOutput(BaseModel):
    time_range: TimeRange
    intents: list[FailureIntent]        # Sorted by failure_count desc
    total_failures: int
    total_interactions: int
    overall_failure_rate: float


# ── Tool: rag_query ───────────────────────────────────────────────────────────

class RAGQueryInput(BaseModel):
    """
    Full RAG pipeline input: retrieve relevant context then generate a grounded answer.
    The LLM calls this when it needs a factual answer backed by real conversation data.
    """
    query: str = Field(
        ...,
        description="The question to answer using retrieved conversation context",
        min_length=5,
        max_length=2000,
    )
    collection: Literal["conversations", "intents"] = Field(
        default="conversations",
        description="Which ChromaDB collection to retrieve from",
    )
    top_k: int = Field(
        default=8,
        ge=1,
        le=20,
        description="Number of context chunks to retrieve",
    )
    time_range: TimeRange | None = Field(
        default=None,
        description="Optional time filter for retrieved chunks",
    )
    temperature: float = Field(
        default=0.2,
        ge=0.0,
        le=1.0,
        description="Generation temperature — keep low for factual answers",
    )
    include_sources: bool = Field(
        default=True,
        description="Include source message IDs and similarity scores in output",
    )


class RAGSource(BaseModel):
    message_id: str
    content: str
    similarity: float
    timestamp: datetime
    intent: str | None = None


class RAGQueryOutput(BaseModel):
    query: str
    answer: str                         # LLM-generated grounded answer
    sources: list[RAGSource]            # Retrieved chunks used as context
    confidence: float                   # Avg similarity of top sources
    retrieval_ms: float
    generation_ms: float


# ── Tool: prompt_optimizer ────────────────────────────────────────────────────

class PromptVariant(BaseModel):
    """A single prompt variant with its performance record."""
    variant_id: str
    template: str
    description: str = ""
    avg_score: float = 0.0
    usage_count: int = 0
    created_at: datetime | None = None


class PromptOptimizeInput(BaseModel):
    """
    Optimize a prompt template based on historical performance data.
    The LLM calls this to suggest improvements to underperforming prompts.
    """
    prompt_key: str = Field(
        ...,
        description="Identifier for the prompt (e.g. 'analyst_system', 'summary_template')",
    )
    current_template: str = Field(
        ...,
        description="Current prompt text to improve",
    )
    optimization_goal: Literal[
        "accuracy", "conciseness", "user_satisfaction", "failure_reduction"
    ] = Field(
        default="user_satisfaction",
        description="What aspect to optimize for",
    )
    n_variants: int = Field(
        default=3,
        ge=1,
        le=5,
        description="Number of optimized variants to generate",
    )
    use_feedback_data: bool = Field(
        default=True,
        description="Use stored user feedback to guide optimization",
    )


class PromptOptimizeOutput(BaseModel):
    prompt_key: str
    original_template: str
    variants: list[PromptVariant]
    optimization_rationale: str         # LLM explanation of the improvements
    recommended_variant_id: str
    expected_improvement: str


class PromptPerformanceRecord(BaseModel):
    """Record a single prompt execution's performance for future optimization."""
    prompt_key: str
    variant_id: str
    score: float = Field(ge=0.0, le=1.0)
    feedback_text: str | None = None
    latency_ms: float | None = None
    tokens_used: int | None = None


# ── Tool: feedback ────────────────────────────────────────────────────────────

class FeedbackSentiment(str, Enum):
    POSITIVE = "positive"
    NEGATIVE = "negative"
    NEUTRAL = "neutral"


class FeedbackRecordInput(BaseModel):
    """Record a user's feedback on a bot response."""
    session_id: str
    message_id: str
    rating: int = Field(ge=1, le=5, description="1-5 star rating")
    sentiment: FeedbackSentiment
    comment: str | None = Field(default=None, max_length=1000)
    response_was_helpful: bool
    intent_was_correct: bool | None = None
    suggested_intent: str | None = None


class FeedbackRecordOutput(BaseModel):
    feedback_id: str
    recorded_at: datetime
    will_affect_training: bool          # True if score is extreme enough to flag


class FeedbackAnalyticsInput(BaseModel):
    """Retrieve aggregated feedback metrics for the analytics dashboard."""
    time_range: TimeRange
    group_by: Literal["day", "intent", "sentiment", "rating"] = "day"
    min_count: int = Field(default=1, ge=1)


class FeedbackDataPoint(BaseModel):
    label: str                          # Date, intent name, sentiment value, etc.
    count: int
    avg_rating: float
    helpfulness_rate: float             # % marked as helpful
    negative_rate: float


class FeedbackAnalyticsOutput(BaseModel):
    time_range: TimeRange
    group_by: str
    data: list[FeedbackDataPoint]
    overall_avg_rating: float
    total_feedback_count: int
    nps_score: float | None = None      # Net Promoter Score if enough data


# ── Guardrails ────────────────────────────────────────────────────────────────

class ThreatType(str, Enum):
    PROMPT_INJECTION = "prompt_injection"
    PII_DETECTED = "pii_detected"
    JAILBREAK_ATTEMPT = "jailbreak_attempt"
    POLICY_VIOLATION = "policy_violation"
    TOXIC_CONTENT = "toxic_content"
    SQL_INJECTION = "sql_injection"


class GuardrailViolation(BaseModel):
    threat_type: ThreatType
    severity: SeverityLevel
    description: str
    detected_pattern: str | None = None
    pii_entities: list[str] = Field(default_factory=list)


class GuardrailCheckResult(BaseModel):
    """Result of running input through all guardrail checks."""
    is_safe: bool
    violations: list[GuardrailViolation]
    sanitized_text: str                 # Text with PII redacted / injections neutralised
    original_text: str
    check_duration_ms: float


# ── Tree-of-Thought Reasoning ─────────────────────────────────────────────────

class ToTThought(BaseModel):
    """A single thought branch in the Tree-of-Thought reasoning."""
    thought_id: str
    depth: int
    content: str                        # The reasoning step content
    score: float = Field(ge=0.0, le=1.0, description="Evaluator score for this thought")
    is_terminal: bool = False
    children: list[str] = Field(default_factory=list, description="Child thought IDs")
    parent_id: str | None = None


class ToTReasoningInput(BaseModel):
    """
    Trigger Tree-of-Thought reasoning for complex analytical problems.
    Use when the user's question requires multi-step hypothesis generation and evaluation.
    """
    problem: str = Field(
        ...,
        description="The complex analytical problem to reason about",
        min_length=10,
    )
    context: str = Field(
        default="",
        description="Background data or metrics context to reason over",
    )
    max_depth: int = Field(default=3, ge=1, le=5)
    branching_factor: int = Field(
        default=3,
        ge=2,
        le=5,
        description="Number of thought branches to explore at each step",
    )
    strategy: Literal["bfs", "dfs", "best_first"] = Field(
        default="best_first",
        description="Tree traversal strategy",
    )


class ToTReasoningOutput(BaseModel):
    problem: str
    best_answer: str                    # Final answer from highest-scoring path
    reasoning_path: list[ToTThought]   # The winning chain of thoughts
    all_thoughts: list[ToTThought]     # Full tree for debugging/LangSmith
    confidence: float
    total_thoughts_explored: int
    reasoning_duration_ms: float


# ── LangSmith trace context ───────────────────────────────────────────────────

class TraceMetadata(BaseModel):
    """Metadata attached to every LangSmith trace."""
    tool_name: str
    session_id: str | None = None
    user_query: str | None = None
    run_id: str | None = None
    tags: list[str] = Field(default_factory=list)
    extra: dict[str, Any] = Field(default_factory=dict)


# ── MCP Server health ─────────────────────────────────────────────────────────

class HealthStatus(BaseModel):
    status: Literal["healthy", "degraded", "unhealthy"]
    version: str
    components: dict[str, bool]         # {postgres: True, redis: True, chromadb: True}
    rate_limit_status: dict[str, Any]
    uptime_seconds: float