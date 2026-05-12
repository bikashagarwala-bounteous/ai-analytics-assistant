"""
LangGraph pipeline: START → router → analyst → summary → END.
Router short-circuits non-analytics queries with canned responses.
"""

import asyncio
import hashlib
import json
import time
import uuid
from datetime import datetime
from typing import Literal

from google import genai
from google.genai import types
from langgraph.graph import StateGraph, START, END

from agents.state import AgentState
from agents.analyst_agent import analyst_node
from agents.summary_agent import summary_node, summary_node_stream
from db.connections import get_db, get_redis
from core.config import settings
from core.logging import get_logger
from sqlalchemy import text

logger = get_logger(__name__)



_GREETING_RESPONSE = (
    "Hello! I'm an analytics assistant for your chatbot. "
    "I can help you analyse performance metrics, trends, failure patterns, "
    "and conversation quality.\n\n"
    "Try asking something like:\n"
    "- *Why did engagement drop last week?*\n"
    "- *What are the top failure intents this month?*\n"
    "- *Show me message volume trends for the past 30 days*"
)

_IDENTITY_RESPONSE = (
    "Yes — I'm an analytics assistant built to help you understand your chatbot's "
    "performance. I can answer questions about engagement, failure rates, user "
    "trends, and conversation quality using live data from your chatbot.\n\n"
    "What would you like to analyse?"
)

_CAPABILITY_RESPONSE = (
    "I can help you with:\n\n"
    "- **Metrics** — engagement rate, failure rate, resolution rate, message volume\n"
    "- **Trends** — how metrics are changing over time\n"
    "- **Anomalies** — unusual spikes or drops in any metric\n"
    "- **Failure analysis** — which intents fail most and why\n"
    "- **Conversation search** — find real examples from your conversation history\n\n"
    "Just ask your question in plain English."
)

_OFF_TOPIC_RESPONSE = (
    "I'm a chatbot analytics assistant — I can only answer questions about your "
    "chatbot's performance data (metrics, trends, failures, conversations, etc.).\n\n"
    "That question is outside my scope. Try asking something like:\n"
    "- *What are the top failure intents this week?*\n"
    "- *Has engagement improved over the last 30 days?*\n"
    "- *Show me message volume trends*"
)

_ROUTER_SYSTEM_PROMPT = """You are a query classifier for an analytics assistant that analyses chatbot performance data.

Classify the user's message into exactly one of these categories:
- "greeting": casual greetings, thanks, acknowledgements, one-word replies, etc
- "identity": questions about what the assistant is or who built it
- "capability": questions about what the assistant can do or help with
- "analytics": questions about chatbot performance, metrics, engagement, failure rates, intents, sessions, users, trends, or conversation data
- "off_topic": anything unrelated to chatbot analytics — general knowledge, personal advice, tech support, coding help, weather, news, etc.

Examples:
"Hi" → greeting
"Hello there" → greeting
"Thanks" → greeting
"ok got it" → greeting
"Are you a bot?" → identity
"What are you?" → identity
"You're an AI right?" → identity
"What can you do?" → capability
"How can you help me?" → capability
"What features do you have?" → capability
"Why did engagement drop last week?" → analytics
"What are the top failure intents?" → analytics
"Show me session trends for 30 days" → analytics
"My laptop is running slow" → off_topic
"What is the weather today?" → off_topic
"Write me a Python script" → off_topic
"How do I fix my wifi?" → off_topic

Respond with ONLY the category word. No explanation, no punctuation, no extra text."""


async def router_node(state: AgentState) -> dict:
    """Classify query; short-circuit non-analytics with canned responses. Retries on transient errors."""
    _VALID = {"greeting", "identity", "capability", "analytics", "off_topic"}
    _RETRYABLE = ("500", "503", "internal", "unavailable", "overloaded")
    client = genai.Client(api_key=settings.gemini_api_key)
    category = "analytics"

    for attempt in range(3):
        try:
            response = await asyncio.to_thread(
                client.models.generate_content,
                model=settings.gemini_model,
                contents=[state.query],
                config=types.GenerateContentConfig(
                    system_instruction=_ROUTER_SYSTEM_PROMPT,
                    temperature=0.0,
                    max_output_tokens=10,
                ),
            )
            raw = (response.text or "").strip().lower()
            category = raw.split()[0] if raw.split() else "analytics"
            if category not in _VALID:
                category = "analytics"
            break
        except Exception as exc:
            msg = str(exc).lower()
            if any(k in msg for k in _RETRYABLE) and attempt < 2:
                wait = 2 ** attempt
                logger.warning(
                    "router_llm_retry",
                    attempt=attempt,
                    retry_in=wait,
                    error=str(exc)[:120],
                )
                await asyncio.sleep(wait)
            else:
                logger.warning("router_llm_failed_defaulting_to_analytics", error=str(exc)[:120])
                break

    logger.debug("router_classified", query=state.query[:60], category=category)

    if category == "greeting":
        return {"summary_output": _GREETING_RESPONSE, "status": "completed"}
    if category == "identity":
        return {"summary_output": _IDENTITY_RESPONSE, "status": "completed"}
    if category == "capability":
        return {"summary_output": _CAPABILITY_RESPONSE, "status": "completed"}
    if category == "off_topic":
        return {"summary_output": _OFF_TOPIC_RESPONSE, "status": "completed"}
    return {"status": "running"}


def _route_after_router(state: AgentState) -> Literal["analyst", "__end__"]:
    if state.status == "completed":
        return END
    return "analyst"


def _route_after_analyst(state: AgentState) -> Literal["summary", "__end__"]:
    if state.status == "failed":
        return END
    return "summary"


def _user_friendly_error(raw: str) -> str:
    low = raw.lower()
    if "500" in low or "internal error" in low or "internal" in low:
        return "The AI service encountered a temporary error. Please try again in a moment."
    if "timeout" in low or "timed out" in low:
        return "The analysis took too long to complete. Try a simpler or narrower question."
    if "planning failed" in low:
        return "Could not generate an analysis plan. Please rephrase your question."
    if "rate limit" in low or "429" in low or "quota" in low:
        return "The AI service is rate-limited. Please wait a few seconds and try again."
    return raw[:120] if len(raw) > 120 else raw


def build_graph() -> StateGraph:
    graph = StateGraph(AgentState)

    graph.add_node("router", router_node)
    graph.add_node("analyst", analyst_node)
    graph.add_node("summary", summary_node)

    graph.add_edge(START, "router")
    graph.add_conditional_edges("router", _route_after_router, {"analyst": "analyst", END: END})
    graph.add_conditional_edges("analyst", _route_after_analyst, {"summary": "summary", END: END})
    graph.add_edge("summary", END)

    return graph


_compiled_graph = None


def get_compiled_graph():
    global _compiled_graph
    if _compiled_graph is None:
        _compiled_graph = build_graph().compile()
        logger.info("langgraph_compiled")
    return _compiled_graph


def _query_cache_key(query: str) -> str:
    digest = hashlib.sha256(query.strip().lower().encode()).hexdigest()[:20]
    return f"cache:pipeline:{digest}"


async def run_analytics_pipeline(
    query: str,
    session_id: str | None = None,
) -> AgentState:
    """Run the pipeline non-streaming. Redis-caches results by query hash."""
    cache_key = _query_cache_key(query)
    try:
        redis = get_redis()
        cached_raw = await redis.get(cache_key)
        if cached_raw:
            logger.info("pipeline_cache_hit", query=query[:60])
            cached_data = json.loads(cached_raw)
            return AgentState(**cached_data)
    except Exception as exc:
        logger.warning("pipeline_cache_read_failed", error=str(exc))

    run_id = str(uuid.uuid4())
    initial_state = AgentState(
        query=query,
        session_id=session_id,
        run_id=run_id,
    )

    await _create_run_record(run_id, session_id, query)

    t0 = time.perf_counter()
    final_state = initial_state

    try:
        graph = get_compiled_graph()
        result = await asyncio.wait_for(
            graph.ainvoke(initial_state.model_dump()),
            timeout=settings.agent_timeout_seconds,
        )
        final_state = AgentState(**result)
        duration_ms = int((time.perf_counter() - t0) * 1000)
        await _update_run_record(run_id, final_state, duration_ms, "completed")

        try:
            redis = get_redis()
            await redis.setex(
                cache_key,
                settings.cache_ttl_agent_run,
                json.dumps(final_state.model_dump(), default=str),
            )
            logger.debug("pipeline_cache_set", query=query[:60], ttl=settings.cache_ttl_agent_run)
        except Exception as exc:
            logger.warning("pipeline_cache_write_failed", error=str(exc))

    except asyncio.TimeoutError:
        duration_ms = int((time.perf_counter() - t0) * 1000)
        final_state.status = "failed"
        final_state.error = f"Pipeline timed out after {settings.agent_timeout_seconds}s"
        await _update_run_record(run_id, final_state, duration_ms, "failed")
        logger.error("pipeline_timeout", run_id=run_id, timeout=settings.agent_timeout_seconds)

    except Exception as exc:
        duration_ms = int((time.perf_counter() - t0) * 1000)
        final_state.status = "failed"
        final_state.error = str(exc)
        await _update_run_record(run_id, final_state, duration_ms, "failed")
        logger.error("pipeline_error", run_id=run_id, error=str(exc))
        raise

    return final_state


async def run_analytics_pipeline_stream(
    query: str,
    session_id: str | None = None,
):
    """Streaming pipeline. Yields text chunks and event: status strings over SSE."""
    run_id = str(uuid.uuid4())
    initial_state = AgentState(
        query=query,
        session_id=session_id,
        run_id=run_id,
    )

    await _create_run_record(run_id, session_id, query)
    t0 = time.perf_counter()

    try:
        router_updates = await router_node(initial_state)
        for key, val in router_updates.items():
            setattr(initial_state, key, val)

        if initial_state.status == "completed":
            yield initial_state.summary_output
            return

        yield "event:analyst_start"

        analyst_task = asyncio.create_task(analyst_node(initial_state))
        deadline = time.perf_counter() + (settings.agent_timeout_seconds - 10)

        while not analyst_task.done():
            remaining = deadline - time.perf_counter()
            if remaining <= 0:
                analyst_task.cancel()
                yield "event:error:Analyst timed out"
                return
            try:
                await asyncio.wait_for(asyncio.shield(analyst_task), timeout=min(20.0, remaining))
            except asyncio.TimeoutError:
                yield "event:heartbeat"

        if analyst_task.cancelled() or analyst_task.exception():
            exc = analyst_task.exception() if not analyst_task.cancelled() else None
            yield f"event:error:{_user_friendly_error(str(exc or 'Analyst cancelled'))}"
            return

        analyst_updates = analyst_task.result()
        for key, val in analyst_updates.items():
            if key == "tool_calls":
                initial_state.tool_calls.extend(val)
            else:
                setattr(initial_state, key, val)

        if initial_state.status == "failed":
            yield f"event:error:{_user_friendly_error(initial_state.error or 'Analysis failed')}"
            return

        yield "event:analyst_complete"
        yield "event:summary_start"

        async for chunk in summary_node_stream(initial_state):
            yield chunk

        duration_ms = int((time.perf_counter() - t0) * 1000)
        initial_state.status = "completed"
        await _update_run_record(run_id, initial_state, duration_ms, "completed")

        yield f"event:complete:{len(initial_state.tool_calls)}"

    except asyncio.TimeoutError:
        duration_ms = int((time.perf_counter() - t0) * 1000)
        await _update_run_record(run_id, initial_state, duration_ms, "failed")
        yield "event:error:Pipeline timed out"

    except Exception as exc:
        duration_ms = int((time.perf_counter() - t0) * 1000)
        await _update_run_record(run_id, initial_state, duration_ms, "failed")
        logger.error("stream_pipeline_error", error=str(exc))
        # not re-raised: caller also catches, which would emit a duplicate error SSE
        yield f"event:error:{_user_friendly_error(str(exc))}"

async def _create_run_record(run_id: str, session_id: str | None, query: str) -> None:
    try:
        async with get_db() as db:
            await db.execute(
                text("""
                    INSERT INTO agent_runs (run_id, session_id, query, status, created_at)
                    VALUES (:run_id, :session_id, :query, 'pending', NOW())
                """),
                {"run_id": run_id, "session_id": session_id, "query": query},
            )
    except Exception as exc:
        logger.warning("run_record_create_failed", error=str(exc))


async def _update_run_record(
    run_id: str, state: AgentState, duration_ms: int, status: str
) -> None:
    try:
        analyst_ms = None
        summary_ms = None
        if state.analyst_completed_at:
            analyst_ms = int(
                (state.analyst_completed_at - state.started_at).total_seconds() * 1000
            )
        if state.summary_completed_at and state.analyst_completed_at:
            summary_ms = int(
                (state.summary_completed_at - state.analyst_completed_at).total_seconds() * 1000
            )

        tool_calls_json = json.dumps(
            [tc.model_dump() for tc in state.tool_calls], default=str
        )
        analyst_output_json = (
            json.dumps(state.analyst_output.model_dump(), default=str)
            if state.analyst_output else None
        )

        async with get_db() as db:
            await db.execute(
                text("""
                    UPDATE agent_runs SET
                        status              = :status,
                        analyst_output      = :analyst_output,
                        summary_output      = :summary_output,
                        tool_calls          = :tool_calls,
                        error_message       = :error,
                        total_duration_ms   = :total_ms,
                        analyst_duration_ms = :analyst_ms,
                        summary_duration_ms = :summary_ms,
                        completed_at        = NOW()
                    WHERE run_id = :run_id
                """),
                {
                    "run_id": run_id,
                    "status": status,
                    "analyst_output": analyst_output_json,
                    "summary_output": state.summary_output or None,
                    "tool_calls": tool_calls_json,
                    "error": state.error,
                    "total_ms": duration_ms,
                    "analyst_ms": analyst_ms,
                    "summary_ms": summary_ms,
                },
            )
    except Exception as exc:
        logger.warning("run_record_update_failed", error=str(exc))