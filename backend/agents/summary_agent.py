"""
Summary Agent LangGraph node.
Converts AnalystFindings into a streamed business-readable response.
Loads the active prompt variant from PostgreSQL and persists the completed report.
"""

import asyncio
import json
import time
import uuid
from collections.abc import AsyncGenerator
from datetime import datetime

from google import genai
from google.genai import types

from agents.state import AgentState, AnalystFindings
from db.connections import get_db
from db.models import Report, PromptPerformance
from core.config import settings
from core.logging import get_logger
from sqlalchemy import text

logger = get_logger(__name__)

_DEFAULT_SUMMARY_PROMPT = """You are a business intelligence writer. You receive structured
analysis data and write clear, concise summaries for business stakeholders.

Guidelines:
- Lead with the most important finding.
- Use plain language. Avoid statistical jargon unless necessary.
- Back every claim with a specific number or example.
- End with 2-3 concrete recommended actions.
- Keep the total response under 400 words unless the question is complex."""


async def summary_node(state: AgentState) -> dict:
    """Generate a human-readable summary from analyst findings and store the report."""
    t0 = time.perf_counter()
    logger.info("summary_node_start", run_id=state.run_id)

    if state.analyst_output is None:
        return {
            "summary_output": "No analysis data was produced. The analyst agent may have encountered an error.",
            "status": "completed",
            "summary_completed_at": datetime.utcnow(),
        }

    system_prompt = await _load_active_prompt("summary_system")
    summary_text, token_count = await _generate_summary(
        state.analyst_output, system_prompt
    )

    duration_ms = int((time.perf_counter() - t0) * 1000)

    report_id = await _store_report(state, summary_text)

    # Record prompt performance (score derived from analyst confidence)
    await _record_performance(
        prompt_key="summary_system",
        score=state.analyst_output.confidence,
        latency_ms=float(duration_ms),
        tokens_used=token_count,
    )

    logger.info(
        "summary_node_complete",
        report_id=report_id,
        duration_ms=duration_ms,
        tokens=token_count,
    )

    return {
        "summary_output": summary_text,
        "status": "completed",
        "summary_completed_at": datetime.utcnow(),
    }


async def summary_node_stream(
    state: AgentState,
) -> AsyncGenerator[str, None]:
    """Stream summary chunks via Gemini async API; stores report on completion."""
    t0 = time.perf_counter()
    logger.info("summary_node_stream_start", run_id=state.run_id)

    if state.analyst_output is None:
        yield "No analysis data was produced."
        return

    system_prompt = await _load_active_prompt("summary_system")
    user_message = _build_user_message(state.analyst_output)

    client = genai.Client(api_key=settings.gemini_api_key)
    full_response = []
    token_count = 0

    queue: asyncio.Queue[str | None] = asyncio.Queue()

    async def _stream_worker() -> None:
        _RETRYABLE = ("500", "503", "internal", "unavailable", "overloaded")
        try:
            for attempt in range(4):
                try:
                    stream = await client.aio.models.generate_content_stream(
                        model=settings.gemini_model,
                        contents=[user_message],
                        config=types.GenerateContentConfig(
                            system_instruction=system_prompt,
                            temperature=settings.summary_temperature,
                            max_output_tokens=2048,
                        ),
                    )
                    async for chunk in stream:
                        piece = chunk.text or ""
                        if piece:
                            full_response.append(piece)
                            await queue.put(piece)
                    break  # success
                except Exception as exc:
                    msg = str(exc).lower()
                    if any(k in msg for k in _RETRYABLE) and attempt < 3:
                        wait = 2 ** attempt
                        logger.warning(
                            "summary_stream_retry",
                            attempt=attempt,
                            retry_in=wait,
                            error=str(exc)[:80],
                        )
                        await asyncio.sleep(wait)
                    else:
                        logger.error("summary_stream_failed", error=str(exc)[:80])
                        raise
        finally:
            await queue.put(None)

    task = asyncio.create_task(_stream_worker())

    while True:
        chunk = await queue.get()
        if chunk is None:
            break
        yield chunk

    try:
        await task
    except Exception as exc:
        if full_response:
            raise
        logger.warning("summary_stream_all_retries_failed_falling_back", error=str(exc)[:120])
        complete_text, token_count = await _generate_summary(state.analyst_output, system_prompt)
        yield complete_text
        full_response.append(complete_text)

    complete_text = "".join(full_response)
    duration_ms = int((time.perf_counter() - t0) * 1000)

    await _store_report(state, complete_text)
    await _record_performance(
        prompt_key="summary_system",
        score=state.analyst_output.confidence,
        latency_ms=float(duration_ms),
        tokens_used=token_count,
    )

    logger.info("summary_node_stream_complete", duration_ms=duration_ms)


async def _generate_summary(
    findings: AnalystFindings, system_prompt: str
) -> tuple[str, int]:
    client = genai.Client(api_key=settings.gemini_api_key)
    user_message = _build_user_message(findings)

    _RETRYABLE = ("500", "503", "internal", "unavailable", "overloaded")

    for attempt in range(5):
        try:
            response = await asyncio.to_thread(
                client.models.generate_content,
                model=settings.gemini_model,
                contents=[user_message],
                config=types.GenerateContentConfig(
                    system_instruction=system_prompt,
                    temperature=settings.summary_temperature,
                    max_output_tokens=2048,
                ),
            )
            text = response.text or ""
            token_count = len(text.split()) * 4 // 3
            return text, token_count
        except Exception as exc:
            msg = str(exc).lower()
            if any(k in msg for k in _RETRYABLE) and attempt < 4:
                wait = 2 ** attempt
                logger.warning(
                    "summary_gemini_retry",
                    attempt=attempt,
                    retry_in=wait,
                    error=str(exc)[:80],
                )
                await asyncio.sleep(wait)
            else:
                raise
    raise RuntimeError("Summary generation failed after retries")


def _build_user_message(findings: AnalystFindings) -> str:
    """Serialize findings into a prompt-ready string; strips time-series and caps list lengths."""
    d = findings.model_dump()
    evidence = d.get("evidence", [])
    d["evidence"] = evidence[:5]
    if len(evidence) > 5:
        d["evidence_truncated"] = f"({len(evidence) - 5} more items not shown)"

    def _strip_time_series(obj):
        if isinstance(obj, dict):
            return {k: _strip_time_series(v) for k, v in obj.items() if k != "time_series"}
        if isinstance(obj, list):
            return [_strip_time_series(i) for i in obj]
        return obj

    d["anomalies_found"] = _strip_time_series(d.get("anomalies_found", []))[:10]
    d["trends_identified"] = _strip_time_series(d.get("trends_identified", []))[:10]

    payload = json.dumps(d, indent=2, default=str)

    return (
        f"Question: {findings.query}\n\n"
        f"Analysis findings:\n{payload}\n\n"
        "Write the business summary now."
    )


async def _load_active_prompt(prompt_key: str) -> str:
    try:
        async with get_db() as db:
            row = await db.execute(
                text("""
                    SELECT template FROM prompt_variants
                    WHERE prompt_key = :key AND is_active = TRUE
                    ORDER BY created_at DESC
                    LIMIT 1
                """),
                {"key": prompt_key},
            )
            result = row.fetchone()
            if result:
                return result.template
    except Exception as exc:
        logger.warning("prompt_load_failed_using_default", error=str(exc))
    return _DEFAULT_SUMMARY_PROMPT


async def _store_report(state: AgentState, summary_text: str) -> str:
    report_id = str(uuid.uuid4())
    try:
        async with get_db() as db:
            findings = state.analyst_output
            content = {
                "findings": findings.findings if findings else [],
                "evidence": findings.evidence if findings else [],
                "anomalies": findings.anomalies_found if findings else [],
                "trends": findings.trends_identified if findings else [],
                "root_causes": findings.root_causes if findings else [],
                "recommended_actions": findings.recommended_actions if findings else [],
                "tool_calls": [tc.model_dump() for tc in state.tool_calls],
            }
            await db.execute(
                text("""
                    INSERT INTO reports
                        (report_id, session_id, run_id, title, summary, content_json, created_at)
                    VALUES
                        (:report_id, :session_id, :run_id, :title, :summary, :content, NOW())
                """),
                {
                    "report_id": report_id,
                    "session_id": state.session_id,
                    "run_id": state.run_id,
                    "title": state.query[:120],
                    "summary": summary_text,
                    "content": json.dumps(content, default=str),
                },
            )
    except Exception as exc:
        logger.error("report_store_failed", error=str(exc))
    return report_id


async def _record_performance(
    prompt_key: str,
    score: float,
    latency_ms: float,
    tokens_used: int,
) -> None:
    try:
        async with get_db() as db:
            await db.execute(
                text("""
                    INSERT INTO prompt_performances
                        (id, prompt_key, score, latency_ms, tokens_used, recorded_at)
                    VALUES
                        (:id, :key, :score, :latency_ms, :tokens, NOW())
                """),
                {
                    "id": str(uuid.uuid4()),
                    "key": prompt_key,
                    "score": score,
                    "latency_ms": latency_ms,
                    "tokens": tokens_used,
                },
            )
    except Exception as exc:
        logger.warning("performance_record_failed", error=str(exc))