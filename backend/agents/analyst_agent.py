"""
Analyst Agent LangGraph node.
Gemini produces a JSON tool plan; Python executes MCP tools (parallel where possible)
and synthesizes results into AnalystFindings.
"""

import asyncio
import json
import time
from datetime import datetime, timedelta

from google import genai
from google.genai import types

from agents.state import AgentState, AnalystFindings, ToolCallRecord
from services.mcp_client import call_tool, MCPError
from core.config import settings
from core.logging import get_logger

logger = get_logger(__name__)


_PLANNING_SYSTEM_PROMPT = """You are the planning component of an analytics agent.
Your job is to decide which tools to call to answer the user's question about chatbot performance.

Available tools with EXACT argument names (use these names exactly):

get_metrics(metrics: list[str], time_range_start: str, time_range_end: str,
            aggregation_period: str = "day", compare_with_previous: bool = False)
  - metrics values and what they measure:
      "engagement_rate"     — % of sessions with >=2 user messages
      "failure_rate"        — % of messages classified as failures
      "resolution_rate"     — % of sessions marked resolved
      "avg_turns"           — average number of user messages PER SESSION (turns/messages per session)
      "session_duration"    — average session length in MINUTES (time, not message count)
      "message_volume"      — total count of user messages
      "unique_users"        — distinct user/session count
      "intent_distribution" — intent category breakdown
  - "avg_turns" is the correct metric for: "average turns", "turns per session",
    "messages per session", "how many messages per conversation"
  - "session_duration" is for: "how long sessions last", "session time in minutes"

find_anomalies(metric: str, time_range_start: str, time_range_end: str,
               sensitivity: float = 2.0, method: str = "zscore")

get_trends(metrics: list[str], time_range_start: str, time_range_end: str,
           aggregation_period: str = "day", forecast_periods: int = 0,
           include_seasonality: bool = True)

top_failure_intents(time_range_start: str, time_range_end: str,
                    top_n: int = 10, min_occurrences: int = 1, include_examples: bool = True)

search_conversations(query: str, limit: int = 10, min_similarity: float = 0.6,
                     intent_filter: str | None = None,
                     time_range_start: str | None = None, time_range_end: str | None = None)

ask_with_rag(query: str, collection: str = "conversations", top_k: int = 8,
             temperature: float = 0.2,
             time_range_start: str | None = None, time_range_end: str | None = None,
             include_sources: bool = True)

CRITICAL: Always use "time_range_start" and "time_range_end" (not start_time/end_time or
any other names). Values must be ISO 8601 strings, e.g. "2026-04-30T00:00:00".

Rules:
- Always start with get_metrics for any quantitative question.
- Use find_anomalies or get_trends when the user asks about changes over time.
- Use top_failure_intents for any question about bot failures or weaknesses.
- Use search_conversations or ask_with_rag to find supporting examples.
- Maximum 6 tool calls per plan.
- Default time range: last 7 days. Adjust based on the question.

Return a JSON object with this exact structure:
{
  "reasoning": "brief explanation of the plan",
  "tool_calls": [
    {
      "tool": "tool_name",
      "arguments": { ... },
      "depends_on": [],
      "parallel_group": 0
    }
  ],
  "use_tot_for_synthesis": false
}

parallel_group: tools with the same group number run concurrently.
depends_on: list of indices of tool calls that must complete first (0-indexed).
"""


_SYNTHESIS_SYSTEM_PROMPT = """You are an analytics data interpreter.
You receive raw tool results and extract structured findings.

Return a JSON object matching this exact structure:
{
  "findings": ["finding 1", "finding 2"],
  "evidence": [{"metric": "...", "value": ..., "period": "..."}],
  "anomalies_found": [{"metric": "...", "description": "...", "severity": "..."}],
  "trends_identified": [{"metric": "...", "direction": "...", "description": "..."}],
  "root_causes": [],
  "confidence": 0.8,
  "recommended_actions": ["action 1", "action 2"],
  "data_coverage": "X days of data, Y conversations",
  "insufficient_data": false
}

IMPORTANT: anomalies_found and trends_identified must be lists of objects (dicts), NOT lists of strings.

Be specific. Include actual numbers from the tool results.
- confidence: set 0.7-1.0 if any tool returned real data (even partial). Set 0.4-0.6
  if results were sparse. Set 0.1-0.3 only if every single tool returned completely
  empty results with zero rows. Never set 0.0 unless all tools failed with errors.
- RAG results (ask_with_rag) with sources count > 0 ARE valid data — treat them as
  evidence of what users discussed. Extract findings from the answer text.
- If top_failure_intents returned an empty list, note that no failures exceeded the
  threshold but do not treat this as a system failure — it may mean low failure rate.
- insufficient_data: only set true if every tool returned empty AND no errors explain why.
  If RAG returned sources, insufficient_data must be false.
- Do not invent missing time ranges or suggest the system needs configuration.
  Work with whatever data the tools returned.
"""


async def analyst_node(state: AgentState) -> dict:
    """Plan tool calls via Gemini, execute them (parallel where possible), synthesize findings."""
    t0 = time.perf_counter()
    logger.info("analyst_node_start", query=state.query[:80], run_id=state.run_id)

    tool_calls: list[ToolCallRecord] = []

    try:
        plan = await _generate_plan(state.query)
        logger.info(
            "analyst_plan_ready",
            tool_count=len(plan.get("tool_calls", [])),
            use_tot=plan.get("use_tot_for_synthesis", False),
        )
    except Exception as exc:
        logger.error("analyst_planning_failed", error=str(exc))
        return {
            "status": "failed",
            "error": f"Planning failed: {exc}",
        }

    tool_results: dict[int, dict] = {}
    planned_calls = plan.get("tool_calls", [])[:settings.agent_max_tool_calls]
    groups: dict[int, list[tuple[int, dict]]] = {}
    for idx, tc in enumerate(planned_calls):
        group = tc.get("parallel_group", idx)
        groups.setdefault(group, []).append((idx, tc))

    for group_id in sorted(groups.keys()):
        group_calls = groups[group_id]

        ready = []
        for idx, tc in group_calls:
            deps = tc.get("depends_on", [])
            if all(d in tool_results for d in deps):
                ready.append((idx, tc))
            else:
                logger.warning("tool_dependency_not_met", tool=tc["tool"], deps=deps)

        if not ready:
            continue

        tasks = [
            _execute_tool(idx, tc["tool"], tc.get("arguments", {}))
            for idx, tc in ready
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        for (idx, tc), result in zip(ready, results):
            if isinstance(result, Exception):
                record = ToolCallRecord(
                    tool_name=tc["tool"],
                    input=tc.get("arguments", {}),
                    error=str(result),
                )
                tool_calls.append(record)
                tool_results[idx] = {"error": str(result)}
            else:
                record, output = result
                tool_calls.append(record)
                tool_results[idx] = output

    use_tot = plan.get("use_tot_for_synthesis", False)

    if use_tot and any(not isinstance(r, dict) or "error" not in r for r in tool_results.values()):
        # Use Tree-of-Thought for complex synthesis
        findings = await _synthesize_with_tot(state.query, tool_results)
    else:
        findings = await _synthesize_findings(state.query, tool_results)

    duration_ms = int((time.perf_counter() - t0) * 1000)
    logger.info(
        "analyst_node_complete",
        findings=len(findings.findings),
        confidence=findings.confidence,
        tool_calls=len(tool_calls),
        duration_ms=duration_ms,
    )

    return {
        "analyst_output": findings,
        "tool_calls": tool_calls,
        "should_use_tot": use_tot,
        "analyst_completed_at": datetime.utcnow(),
        "status": "running",
    }


async def _generate_plan(query: str) -> dict:
    client = genai.Client(api_key=settings.gemini_api_key)

    now = datetime.utcnow()
    week_ago = now - timedelta(days=7)
    month_ago = now - timedelta(days=30)

    user_message = (
        f"Current UTC time: {now.isoformat()}\n"
        f"Last 7 days: {week_ago.isoformat()} to {now.isoformat()}\n"
        f"Last 30 days: {month_ago.isoformat()} to {now.isoformat()}\n\n"
        f"User question: {query}\n\n"
        "Return the JSON tool plan. No other text."
    )

    _RETRYABLE = ("500", "503", "internal", "unavailable", "overloaded")
    raw = ""
    for attempt in range(4):
        try:
            response = await asyncio.to_thread(
                client.models.generate_content,
                model=settings.gemini_model,
                contents=[user_message],
                config=types.GenerateContentConfig(
                    system_instruction=_PLANNING_SYSTEM_PROMPT,
                    temperature=0.1,
                    max_output_tokens=2048,
                ),
            )
            raw = (response.text or "").strip()
            if not raw:
                if attempt < 3:
                    logger.warning("plan_empty_response_retrying", attempt=attempt)
                    await asyncio.sleep(2 ** attempt)
                    continue
                raise RuntimeError("Gemini returned empty plan response after retries")
            break
        except Exception as exc:
            if any(k in str(exc).lower() for k in _RETRYABLE) and attempt < 3:
                wait = 2 ** attempt
                logger.warning(
                    "analyst_plan_retry",
                    attempt=attempt,
                    retry_in=wait,
                    error=str(exc)[:120],
                )
                await asyncio.sleep(wait)
            else:
                raise

    if raw.startswith("```"):
        raw = raw.split("\n", 1)[1].rsplit("```", 1)[0]

    return json.loads(raw)


async def _execute_tool(
    idx: int,
    tool_name: str,
    arguments: dict,
) -> tuple[ToolCallRecord, dict]:
    t0 = time.perf_counter()
    try:
        output = await call_tool(tool_name, arguments)
        duration_ms = (time.perf_counter() - t0) * 1000
        record = ToolCallRecord(
            tool_name=tool_name,
            input=arguments,
            output=output,
            duration_ms=round(duration_ms, 2),
        )
        logger.debug("tool_executed", tool=tool_name, duration_ms=round(duration_ms, 1))
        return record, output

    except MCPError as exc:
        duration_ms = (time.perf_counter() - t0) * 1000
        logger.warning("tool_execution_failed", tool=tool_name, error=str(exc))
        record = ToolCallRecord(
            tool_name=tool_name,
            input=arguments,
            error=str(exc),
            duration_ms=round(duration_ms, 2),
        )
        return record, {"error": str(exc)}


async def _synthesize_findings(query: str, tool_results: dict[int, dict]) -> AnalystFindings:
    client = genai.Client(api_key=settings.gemini_api_key)
    results_text = json.dumps(
        {str(k): _truncate_result(v) for k, v in tool_results.items()},
        indent=2,
        default=str,
    )

    user_message = (
        f"User question: {query}\n\n"
        f"Tool results:\n{results_text}\n\n"
        "Synthesize these results into the JSON findings object. No other text."
    )

    _RETRYABLE = ("500", "503", "internal", "unavailable", "overloaded")
    raw = ""
    for attempt in range(4):
        try:
            response = await asyncio.to_thread(
                client.models.generate_content,
                model=settings.gemini_model,
                contents=[user_message],
                config=types.GenerateContentConfig(
                    system_instruction=_SYNTHESIS_SYSTEM_PROMPT,
                    temperature=0.1,
                    max_output_tokens=2048,
                ),
            )
            raw = (response.text or "").strip()
            if not raw:
                if attempt < 3:
                    logger.warning("synthesis_empty_response_retrying", attempt=attempt)
                    await asyncio.sleep(2 ** attempt)
                    continue
                raise RuntimeError("Gemini returned empty synthesis response after retries")
            break
        except Exception as exc:
            if any(k in str(exc).lower() for k in _RETRYABLE) and attempt < 3:
                wait = 2 ** attempt
                logger.warning(
                    "analyst_synthesis_retry",
                    attempt=attempt,
                    retry_in=wait,
                    error=str(exc)[:120],
                )
                await asyncio.sleep(wait)
            else:
                raise

    if raw.startswith("```"):
        raw = raw.split("\n", 1)[1].rsplit("```", 1)[0]

    data = json.loads(raw)

    # Gemini occasionally returns lists of strings rather than dicts for these fields
    for list_field in ("anomalies_found", "trends_identified", "evidence"):
        if list_field in data and isinstance(data[list_field], list):
            data[list_field] = [
                {"description": item} if isinstance(item, str) else item
                for item in data[list_field]
            ]

    return AnalystFindings(query=query, **data)


async def _synthesize_with_tot(query: str, tool_results: dict[int, dict]) -> AnalystFindings:
    """Use the MCP Tree-of-Thought tool for synthesis; falls back to standard synthesis."""
    try:
        context = json.dumps(
            {str(k): _truncate_result(v) for k, v in tool_results.items()},
            default=str,
        )
        tot_result = await call_tool("deep_analysis", {
            "problem": query,
            "context": context[:3000],
            "max_depth": 3,
            "branching_factor": 3,
            "strategy": "best_first",
        })

        best_answer = tot_result.get("best_answer", "")
        confidence = tot_result.get("confidence", 0.7)

        findings = await _synthesize_findings(query, tool_results)
        if best_answer:
            findings.findings.insert(0, f"Deep analysis: {best_answer}")
        findings.confidence = max(findings.confidence, confidence)
        return findings

    except Exception as exc:
        logger.warning("tot_synthesis_failed_falling_back", error=str(exc))
        return await _synthesize_findings(query, tool_results)


def _truncate_result(result: dict, max_items: int = 20) -> dict:
    """Cap list lengths in tool results to stay within synthesis prompt token limits."""
    if not isinstance(result, dict):
        return result

    truncated = {}
    for key, val in result.items():
        if isinstance(val, list) and len(val) > max_items:
            truncated[key] = val[:max_items]
            truncated[f"{key}_truncated"] = f"(showing {max_items} of {len(val)})"
        elif isinstance(val, dict):
            truncated[key] = _truncate_result(val, max_items)
        else:
            truncated[key] = val
    return truncated