"""
AI Analytics MCP Server — Streamable HTTP transport.
Startup: logging → LangSmith → Redis → PostgreSQL → ChromaDB.
FastMCP must own the lifecycle — re-wrapping in Starlette breaks the session_manager task group.
"""

import asyncio
import time
from typing import Any

from mcp.server.fastmcp import FastMCP

from core.config import settings
from core.logging import setup_logging, get_logger
from core.rate_limiter import get_status as get_rate_limit_status
from core.langsmith_tracer import init_langsmith, trace_tool
from core.guardrails import check_input, GuardrailError
from core.tot_reasoner import reason as tot_reason
from db.redis_client import init_redis_pool, close_redis_pool
from db.postgres import init_postgres, close_postgres
from db.chromadb_client import init_chromadb, close_chromadb
from schemas import (
    HealthStatus,
    TimeRange,
    MetricType,
    AggregationPeriod,
    RAGQueryInput,
    PromptOptimizeInput,
    PromptPerformanceRecord,
    FeedbackRecordInput,
    FeedbackSentiment,
    FeedbackAnalyticsInput,
    ToTReasoningInput,
)
from tools import (
    search_similar_conversations,
    query_conversation_metrics,
    detect_anomalies,
    analyze_trends,
    get_failure_intents,
    rag_query,
    optimize_prompt,
    record_prompt_performance,
    record_feedback,
    get_feedback_analytics,
)

# ── Logging (before anything else) ────────────────────────────────────────────
setup_logging()
logger = get_logger(__name__)

_start_time = time.time()

# ── Metric name normalization ──────────────────────────────────────────────────

_METRIC_SYNONYMS: dict[str, str] = {
    # engagement
    "engagement":               "engagement_rate",
    "engagement rate":          "engagement_rate",
    # failure / errors
    "failure":                  "failure_rate",
    "failures":                 "failure_rate",
    "error_rate":               "failure_rate",
    "errors":                   "failure_rate",
    "bot_failure_rate":         "failure_rate",
    # resolution / satisfaction
    "resolution":               "resolution_rate",
    "resolved":                 "resolution_rate",
    "user_satisfaction":        "resolution_rate",
    "satisfaction":             "resolution_rate",
    "csat":                     "resolution_rate",
    "customer_satisfaction":    "resolution_rate",
    # message volume
    "volume":                   "message_volume",
    "message_count":            "message_volume",
    "messages":                 "message_volume",
    "chat_volume":              "message_volume",
    "total_messages":           "message_volume",
    # session / user count
    "session_count":            "unique_users",
    "sessions":                 "unique_users",
    "session_volume":           "unique_users",
    "num_sessions":             "unique_users",
    "user_count":               "unique_users",
    "users":                    "unique_users",
    "total_users":              "unique_users",
    # session duration / length
    "avg_session_length":       "session_duration",
    "session_length":           "session_duration",
    "chat_duration":            "session_duration",
    "duration":                 "session_duration",
    "average_session_duration": "session_duration",
    # turns
    "turns":                    "avg_turns",
    "messages_per_session":     "avg_turns",
    "avg_messages":             "avg_turns",
    "average_turns":            "avg_turns",
    # intent distribution
    "intent":                   "intent_distribution",
    "intents":                  "intent_distribution",
    "intent_breakdown":         "intent_distribution",
}

_PERIOD_SYNONYMS: dict[str, str] = {
    "hourly": "hour",
    "daily":  "day",
    "weekly": "week",
    "monthly": "month",
}


def _normalize_metric(name: str) -> str:
    """Map any synonym to its canonical MetricType value."""
    key = name.lower().strip().replace(" ", "_").replace("-", "_")
    return _METRIC_SYNONYMS.get(key, key)


def _normalize_period(period: str) -> str:
    """Map 'daily' → 'day' etc."""
    return _PERIOD_SYNONYMS.get(period.lower().strip(), period)


# ── Startup / shutdown ─────────────────────────────────────────────────────────

async def _startup() -> None:
    logger.info("mcp_server_starting", version=settings.app_version)
    init_langsmith()

    try:
        await init_redis_pool()
        logger.info("redis_connected")
    except Exception as e:
        logger.error("redis_init_failed", error=str(e))
        raise

    try:
        await init_postgres()
        logger.info("postgres_connected")
    except Exception as e:
        logger.error("postgres_init_failed", error=str(e))
        raise

    try:
        await init_chromadb()
        logger.info("chromadb_connected")
    except Exception as e:
        logger.error("chromadb_init_failed", error=str(e))
        raise

    logger.info(
        "mcp_server_ready",
        host=settings.mcp_host,
        port=settings.mcp_port,
    )


async def _shutdown() -> None:
    logger.info("mcp_server_shutting_down")
    await close_redis_pool()
    await close_postgres()
    await close_chromadb()
    logger.info("mcp_server_stopped")

# ── FastMCP instance ───────────────────────────────────────────────────────────

mcp = FastMCP(
    name=settings.app_name,
    instructions="""
    You are connected to the AI Analytics MCP server.

    Available tools:

    1. search_conversations  — semantic search over conversation history
    2. get_metrics           — quantitative metrics (engagement, failure rate, volume)
    3. find_anomalies        — statistical anomaly detection on a metric
    4. get_trends            — trend direction, seasonality, forecasting
    5. top_failure_intents   — which intents fail most often
    6. ask_with_rag          — RAG: retrieve + generate grounded answer with citations
    7. improve_prompt        — analyse and optimise a prompt template
    8. submit_feedback       — record a user feedback rating
    9. feedback_analytics    — aggregated feedback metrics for the dashboard
    10. deep_analysis        — Tree-of-Thought reasoning for complex root cause questions

    Tool selection principle:
    - Start broad: use get_metrics to get an overview.
    - Drill down: use find_anomalies or get_trends on specific metrics.
    - Use search_conversations or ask_with_rag to find supporting evidence.
    - Use deep_analysis only when a single-step answer is insufficient.
    """,
)


# ── HTTP /health route ─────────────────────────────────────────────────────────
# Added directly to FastMCP so it runs inside the properly initialised server.
# Docker health check hits this endpoint.

@mcp.custom_route("/health", methods=["GET"])
async def http_health(request) -> Any:
    from starlette.responses import JSONResponse

    components = {"postgres": False, "redis": False, "chromadb": False}

    try:
        from db.postgres import get_engine
        from sqlalchemy import text as sa_text
        async with get_engine().connect() as conn:
            await conn.execute(sa_text("SELECT 1"))
        components["postgres"] = True
    except Exception:
        pass

    try:
        from db.redis_client import get_redis
        await get_redis().ping()
        components["redis"] = True
    except Exception:
        pass

    try:
        from db.chromadb_client import get_chromadb
        await get_chromadb().heartbeat()
        components["chromadb"] = True
    except Exception:
        pass

    all_ok = all(components.values())
    return JSONResponse(
        {
            "status": "healthy" if all_ok else "degraded",
            "version": settings.app_version,
            "components": components,
            "uptime_seconds": round(time.time() - _start_time, 1),
        },
        status_code=200 if all_ok else 503,
    )


# ── MCP resource: health status ────────────────────────────────────────────────

@mcp.resource("health://status")
async def health_check() -> str:
    import json
    components = {"postgres": False, "redis": False, "chromadb": False}
    try:
        from db.postgres import get_engine
        from sqlalchemy import text
        async with get_engine().connect() as conn:
            await conn.execute(text("SELECT 1"))
        components["postgres"] = True
    except Exception:
        pass
    try:
        from db.redis_client import get_redis
        await get_redis().ping()
        components["redis"] = True
    except Exception:
        pass
    try:
        from db.chromadb_client import get_chromadb
        await get_chromadb().heartbeat()
        components["chromadb"] = True
    except Exception:
        pass
    rl_status = await get_rate_limit_status()
    status = HealthStatus(
        status=(
            "healthy" if all(components.values())
            else "degraded" if any(components.values())
            else "unhealthy"
        ),
        version=settings.app_version,
        components=components,
        rate_limit_status=rl_status.model_dump(),
        uptime_seconds=round(time.time() - _start_time, 1),
    )
    return json.dumps(status.model_dump(mode="json"), indent=2)


# ═══════════════════════════════════════════════════════════════════════════════
# Tool registrations
# ═══════════════════════════════════════════════════════════════════════════════

@mcp.tool(
    description=(
        "Semantically search conversation history for messages similar to the query. "
        "Returns ranked matches with similarity scores, timestamps, and intent labels. "
        "Use this to find examples of a specific issue or pattern in real conversations."
    )
)
async def search_conversations(
    query: str,
    limit: int = 10,
    min_similarity: float = 0.6,
    intent_filter: str | None = None,
    time_range_start: str | None = None,
    time_range_end: str | None = None,
) -> dict[str, Any]:
    from datetime import datetime
    from schemas import VectorSearchInput, TimeRange

    time_range = None
    if time_range_start and time_range_end:
        time_range = TimeRange(
            start=datetime.fromisoformat(time_range_start),
            end=datetime.fromisoformat(time_range_end),
        )
    params = VectorSearchInput(
        query=query,
        limit=limit,
        min_similarity=min_similarity,
        intent_filter=intent_filter,
        time_range=time_range,
    )
    result = await search_similar_conversations(params)
    return result.model_dump(mode="json")


@mcp.tool(
    description=(
        "Query structured analytics metrics (engagement rate, failure rate, message volume, etc.) "
        "for a specified time window. Supports period-over-period comparison. "
        "Use this as the first tool when the user asks a quantitative analytics question."
    )
)
async def get_metrics(
    metrics: list[str],
    time_range_start: str,
    time_range_end: str,
    aggregation_period: str = "day",
    compare_with_previous: bool = False,
) -> dict[str, Any]:
    from datetime import datetime
    from schemas import MetricsQueryInput, TimeRange, MetricType, AggregationPeriod

    normalized = [_normalize_metric(m) for m in metrics]
    period = _normalize_period(aggregation_period)
    params = MetricsQueryInput(
        metrics=[MetricType(m) for m in normalized],
        time_range=TimeRange(
            start=datetime.fromisoformat(time_range_start),
            end=datetime.fromisoformat(time_range_end),
        ),
        aggregation_period=AggregationPeriod(period),
        compare_with_previous_period=compare_with_previous,
    )
    result = await query_conversation_metrics(params)
    return result.model_dump(mode="json")


@mcp.tool(
    description=(
        "Detect statistical anomalies in a specific metric using Z-score, IQR, "
        "or Isolation Forest. Returns anomalies with severity levels and possible causes. "
        "Use after get_metrics when you spot something unusual, or when the user asks "
        "'why did X drop/spike'."
    )
)
async def find_anomalies(
    metric: str,
    time_range_start: str,
    time_range_end: str,
    sensitivity: float = 2.0,
    method: str = "zscore",
) -> dict[str, Any]:
    from datetime import datetime
    from schemas import AnomalyDetectorInput, TimeRange, MetricType

    params = AnomalyDetectorInput(
        metric=MetricType(_normalize_metric(metric)),
        time_range=TimeRange(
            start=datetime.fromisoformat(time_range_start),
            end=datetime.fromisoformat(time_range_end),
        ),
        sensitivity=sensitivity,
        method=method,
    )
    result = await detect_anomalies(params)
    return result.model_dump(mode="json")


@mcp.tool(
    description=(
        "Analyse trends, change points, seasonality, and optionally forecast future values "
        "for one or more metrics. Also computes cross-metric correlations. "
        "Use when the user asks about patterns over time or future projections."
    )
)
async def get_trends(
    metrics: list[str],
    time_range_start: str,
    time_range_end: str,
    aggregation_period: str = "day",
    forecast_periods: int = 0,
    include_seasonality: bool = True,
) -> dict[str, Any]:
    from datetime import datetime
    from schemas import TrendAnalyzerInput, TimeRange, MetricType, AggregationPeriod

    normalized = [_normalize_metric(m) for m in metrics]
    period = _normalize_period(aggregation_period)
    params = TrendAnalyzerInput(
        metrics=[MetricType(m) for m in normalized],
        time_range=TimeRange(
            start=datetime.fromisoformat(time_range_start),
            end=datetime.fromisoformat(time_range_end),
        ),
        aggregation_period=AggregationPeriod(period),
        forecast_periods=forecast_periods,
        include_seasonality=include_seasonality,
    )
    result = await analyze_trends(params)
    return result.model_dump(mode="json")


@mcp.tool(
    description=(
        "Return the top-N most frequently failing intents with failure rates, "
        "trend directions, and example failed messages. "
        "Use directly when the user asks 'What are the top failure intents?' or "
        "'What is the bot struggling with?'."
    )
)
async def top_failure_intents(
    time_range_start: str,
    time_range_end: str,
    top_n: int = 10,
    min_occurrences: int = 1,
    include_examples: bool = True,
) -> dict[str, Any]:
    from datetime import datetime
    from schemas import FailureIntentsInput, TimeRange

    params = FailureIntentsInput(
        time_range=TimeRange(
            start=datetime.fromisoformat(time_range_start),
            end=datetime.fromisoformat(time_range_end),
        ),
        top_n=top_n,
        min_occurrences=min_occurrences,
        include_examples=include_examples,
    )
    result = await get_failure_intents(params)
    return result.model_dump(mode="json")


@mcp.tool(
    description=(
        "Full RAG pipeline: retrieve semantically relevant conversation chunks "
        "then generate a grounded, citation-backed answer. "
        "Use when you need a factual answer with evidence from real conversation data."
    )
)
async def ask_with_rag(
    query: str,
    collection: str = "conversations",
    top_k: int = 8,
    temperature: float = 0.2,
    time_range_start: str | None = None,
    time_range_end: str | None = None,
    include_sources: bool = True,
) -> dict[str, Any]:
    from datetime import datetime

    guard = await check_input(query, context="rag_tool")
    if not guard.is_safe:
        return {"error": "Input blocked by guardrails", "violations": [v.threat_type for v in guard.violations]}

    time_range = None
    if time_range_start and time_range_end:
        time_range = TimeRange(
            start=datetime.fromisoformat(time_range_start),
            end=datetime.fromisoformat(time_range_end),
        )
    params = RAGQueryInput(
        query=guard.sanitized_text,
        collection=collection,
        top_k=top_k,
        temperature=temperature,
        time_range=time_range,
        include_sources=include_sources,
    )
    async with trace_tool("ask_with_rag", inputs={"query": query[:200]}) as run:
        result = await rag_query(params)
        run.add_output(result)
    return result.model_dump(mode="json")


@mcp.tool(description="Analyse a prompt template's historical performance and generate improved variants.")
async def improve_prompt(
    prompt_key: str,
    current_template: str,
    optimization_goal: str = "user_satisfaction",
    n_variants: int = 3,
    use_feedback_data: bool = True,
) -> dict[str, Any]:
    guard = await check_input(current_template, context="prompt_template")
    if not guard.is_safe:
        return {"error": "Prompt template blocked by guardrails"}
    params = PromptOptimizeInput(
        prompt_key=prompt_key,
        current_template=guard.sanitized_text,
        optimization_goal=optimization_goal,
        n_variants=n_variants,
        use_feedback_data=use_feedback_data,
    )
    async with trace_tool("improve_prompt", inputs={"prompt_key": prompt_key}) as run:
        result = await optimize_prompt(params)
        run.add_output(result)
    return result.model_dump(mode="json")


@mcp.tool(description="Store a user feedback rating on a specific bot response.")
async def submit_feedback(
    session_id: str,
    message_id: str,
    rating: int,
    sentiment: str,
    response_was_helpful: bool,
    comment: str | None = None,
    intent_was_correct: bool | None = None,
    suggested_intent: str | None = None,
) -> dict[str, Any]:
    params = FeedbackRecordInput(
        session_id=session_id,
        message_id=message_id,
        rating=rating,
        sentiment=FeedbackSentiment(sentiment),
        response_was_helpful=response_was_helpful,
        comment=comment,
        intent_was_correct=intent_was_correct,
        suggested_intent=suggested_intent,
    )
    async with trace_tool("submit_feedback", inputs={"session_id": session_id}) as run:
        result = await record_feedback(params)
        run.add_output(result)
    return result.model_dump(mode="json")


@mcp.tool(description="Retrieve aggregated user feedback analytics grouped by day, intent, sentiment, or rating.")
async def feedback_analytics(
    time_range_start: str,
    time_range_end: str,
    group_by: str = "day",
    min_count: int = 1,
) -> dict[str, Any]:
    from datetime import datetime

    params = FeedbackAnalyticsInput(
        time_range=TimeRange(
            start=datetime.fromisoformat(time_range_start),
            end=datetime.fromisoformat(time_range_end),
        ),
        group_by=group_by,
        min_count=min_count,
    )
    async with trace_tool("feedback_analytics") as run:
        result = await get_feedback_analytics(params)
        run.add_output(result)
    return result.model_dump(mode="json")


@mcp.tool(
    description=(
        "Apply Tree-of-Thought multi-step reasoning to complex analytical problems. "
        "Use for root cause analysis or any question where a single-shot answer is insufficient."
    )
)
async def deep_analysis(
    problem: str,
    context: str = "",
    max_depth: int = 3,
    branching_factor: int = 3,
    strategy: str = "best_first",
) -> dict[str, Any]:
    guard = await check_input(problem, context="tot_problem")
    if not guard.is_safe:
        return {"error": "Problem statement blocked by guardrails"}
    params = ToTReasoningInput(
        problem=guard.sanitized_text,
        context=context,
        max_depth=max_depth,
        branching_factor=branching_factor,
        strategy=strategy,
    )
    async with trace_tool("deep_analysis", inputs={"problem": problem[:200], "strategy": strategy}) as run:
        result = await tot_reason(params)
        run.add_output(result)
    return result.model_dump(mode="json")


# ── Entry point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn

    _mcp_app = mcp.streamable_http_app()
    _expected_host = f"localhost:{settings.mcp_port}".encode()

    async def _app_with_host_rewrite(scope, receive, send) -> None:
        if scope["type"] == "http":
            scope["headers"] = [
                (b"host", _expected_host) if name == b"host" else (name, value)
                for name, value in scope.get("headers", [])
            ]
        await _mcp_app(scope, receive, send)

    async def _run() -> None:
        await _startup()
        try:
            config = uvicorn.Config(
                app=_app_with_host_rewrite,
                host=settings.mcp_host,
                port=settings.mcp_port,
                log_level="debug" if settings.debug else "info",
            )
            server = uvicorn.Server(config)
            await server.serve()
        finally:
            await _shutdown()

    asyncio.run(_run())