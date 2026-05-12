"""
POST /chat endpoint.
stream=true: SSE (real-time tokens). stream=false: blocking JSON response.
"""

import json
import time
import uuid
from collections.abc import AsyncGenerator

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse

from agents.graph import run_analytics_pipeline, run_analytics_pipeline_stream
from services.session_service import (
    get_or_create_session,
    store_message,
    close_session,
)
from services.dashboard_service import invalidate_dashboard_cache
from schemas.api import ChatRequest, ChatResponse
from core.logging import get_logger

router = APIRouter(prefix="/chat", tags=["chat"])
logger = get_logger(__name__)


@router.post("")
async def chat(request: ChatRequest):
    """
    Send a message and receive an analytics response.

    With stream=true, returns an SSE stream where each data line is either:
      - A text chunk:           data: {"type": "chunk", "text": "..."}
      - A status event:         data: {"type": "event", "name": "analyst_start"}
      - The final summary:      data: {"type": "done", "run_id": "...", "session_id": "..."}
      - An error:               data: {"type": "error", "message": "..."}

    With stream=false, returns a standard JSON ChatResponse.
    """
    if request.stream:
        return StreamingResponse(
            _stream_response(request),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no",
            },
        )
    return await _blocking_response(request)


async def _blocking_response(request: ChatRequest) -> ChatResponse:
    t0 = time.perf_counter()
    session_id = await get_or_create_session(request.session_id)

    await store_message(session_id, "user", request.query)

    state = await run_analytics_pipeline(request.query, session_id)

    await store_message(
        session_id,
        "assistant",
        state.summary_output or "",
        latency_ms=int((time.perf_counter() - t0) * 1000),
    )

    await close_session(session_id, resolved=(state.status == "completed"))
    await invalidate_dashboard_cache()

    return ChatResponse(
        session_id=session_id,
        run_id=state.run_id or "",
        answer=state.summary_output or "",
        status=state.status,
        tool_calls_count=len(state.tool_calls),
        duration_ms=int((time.perf_counter() - t0) * 1000),
    )


async def _stream_response(request: ChatRequest) -> AsyncGenerator[bytes, None]:
    session_id = await get_or_create_session(request.session_id)
    await store_message(session_id, "user", request.query)

    full_response_parts: list[str] = []
    run_id = str(uuid.uuid4())
    t0 = time.perf_counter()

    try:
        async for chunk in run_analytics_pipeline_stream(request.query, session_id):
            if chunk.startswith("event:complete"):
                parts = chunk.split(":")
                tool_count = int(parts[2]) if len(parts) > 2 else 0
                full_text = "".join(full_response_parts)
                message_id = await store_message(
                    session_id,
                    "assistant",
                    full_text,
                    latency_ms=int((time.perf_counter() - t0) * 1000),
                )
                await close_session(session_id, resolved=True)
                await invalidate_dashboard_cache()
                yield _sse({
                    "type": "done",
                    "run_id": run_id,
                    "message_id": message_id,
                    "session_id": session_id,
                    "duration_ms": int((time.perf_counter() - t0) * 1000),
                    "tool_calls_count": tool_count,
                })

            elif chunk.startswith("event:"):
                event_name = chunk[6:]

                if event_name.startswith("error:"):
                    error_msg = event_name[6:]
                    await close_session(session_id, resolved=False)
                    yield _sse({"type": "error", "message": error_msg})
                    return

                yield _sse({"type": "event", "name": event_name})

            else:
                full_response_parts.append(chunk)
                yield _sse({"type": "chunk", "text": chunk})

    except Exception as exc:
        logger.error("stream_error", error=str(exc))
        await close_session(session_id, resolved=False)
        yield _sse({"type": "error", "message": "An internal error occurred"})


def _sse(data: dict) -> bytes:
    return f"data: {json.dumps(data)}\n\n".encode()