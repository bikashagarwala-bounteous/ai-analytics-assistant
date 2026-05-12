"""
LangSmith integration for MCP tool observability.

What gets traced:
  - Every tool invocation (input, output, latency, tokens)
  - Guardrail check results
  - Gemini calls (via the gemini_client wrapper)
  - Errors and retries
  - RAG retrieval quality (similarity scores)
  - ToT reasoning paths

Usage:
    async with trace_tool("search_conversations", session_id="abc") as run:
        result = await search_similar_conversations(params)
        run.add_output(result)

All traces are viewable in the LangSmith UI at https://smith.langchain.com
"""

import asyncio
import time
import uuid
from contextlib import asynccontextmanager
from typing import Any, AsyncGenerator

from core.config import settings
from core.logging import get_logger

logger = get_logger(__name__)

# ── LangSmith client (lazy init) ──────────────────────────────────────────────

_ls_client = None
_ls_enabled = False


def _init_langsmith() -> bool:
    """
    Initialise LangSmith client.
    Returns True if LangSmith is available and configured, False otherwise.
    """
    global _ls_client, _ls_enabled

    if not settings.langsmith_api_key:
        logger.info("langsmith_disabled_no_api_key")
        return False

    try:
        from langsmith import Client
        import os

        os.environ["LANGCHAIN_API_KEY"] = settings.langsmith_api_key
        os.environ["LANGCHAIN_TRACING_V2"] = "true"
        os.environ["LANGCHAIN_PROJECT"] = settings.langsmith_project

        _ls_client = Client(
            api_key=settings.langsmith_api_key,
            api_url=settings.langsmith_endpoint,
        )
        _ls_enabled = True
        logger.info(
            "langsmith_initialized",
            project=settings.langsmith_project,
            endpoint=settings.langsmith_endpoint,
        )
        return True
    except ImportError:
        logger.warning("langsmith_not_installed_tracing_disabled")
        return False
    except Exception as exc:
        logger.warning("langsmith_init_failed", error=str(exc))
        return False


# ── Run context ───────────────────────────────────────────────────────────────

class ToolRun:
    """
    Represents a single traced tool execution.
    Collects inputs, outputs, metadata, and errors before flushing to LangSmith.
    """

    def __init__(
        self,
        tool_name: str,
        session_id: str | None = None,
        parent_run_id: str | None = None,
        tags: list[str] | None = None,
    ) -> None:
        self.run_id = str(uuid.uuid4())
        self.tool_name = tool_name
        self.session_id = session_id
        self.parent_run_id = parent_run_id
        self.tags = tags or []
        self.start_time = time.perf_counter()
        self.wall_start = time.time()

        self._inputs: dict[str, Any] = {}
        self._outputs: dict[str, Any] = {}
        self._metadata: dict[str, Any] = {}
        self._error: str | None = None
        self._child_runs: list[str] = []

    def add_input(self, **kwargs: Any) -> None:
        self._inputs.update(kwargs)

    def add_output(self, output: Any) -> None:
        if hasattr(output, "model_dump"):
            self._outputs["result"] = output.model_dump(mode="json")
        elif isinstance(output, dict):
            self._outputs.update(output)
        else:
            self._outputs["result"] = str(output)

    def add_metadata(self, **kwargs: Any) -> None:
        self._metadata.update(kwargs)

    def set_error(self, error: Exception) -> None:
        self._error = f"{type(error).__name__}: {str(error)}"

    @property
    def latency_ms(self) -> float:
        return (time.perf_counter() - self.start_time) * 1000

    def _flush(self) -> None:
        """Send trace to LangSmith."""
        if not _ls_enabled or _ls_client is None:
            return

        try:
            from langsmith.schemas import RunTypeEnum

            _ls_client.create_run(
                id=self.run_id,
                name=self.tool_name,
                run_type=RunTypeEnum.tool,
                inputs=self._inputs,
                outputs=self._outputs if not self._error else None,
                error=self._error,
                start_time=self.wall_start,
                end_time=self.wall_start + self.latency_ms / 1000,
                extra={
                    "metadata": {
                        "session_id": self.session_id,
                        "latency_ms": round(self.latency_ms, 2),
                        **self._metadata,
                    },
                    "tags": self.tags,
                },
                parent_run_id=self.parent_run_id,
            )
            logger.debug(
                "langsmith_trace_flushed",
                tool=self.tool_name,
                run_id=self.run_id[:8],
                latency_ms=round(self.latency_ms, 1),
            )
        except Exception as exc:
            logger.warning("langsmith_flush_failed", error=str(exc)[:100])


# ── Context manager ───────────────────────────────────────────────────────────

@asynccontextmanager
async def trace_tool(
    tool_name: str,
    session_id: str | None = None,
    parent_run_id: str | None = None,
    tags: list[str] | None = None,
    inputs: dict[str, Any] | None = None,
) -> AsyncGenerator[ToolRun, None]:
    """
    Async context manager that traces a tool call from entry to exit.
    """
    run = ToolRun(
        tool_name=tool_name,
        session_id=session_id,
        parent_run_id=parent_run_id,
        tags=[settings.environment, *( tags or [])],
    )

    if inputs:
        run.add_input(**inputs)

    try:
        yield run
    except Exception as exc:
        run.set_error(exc)
        raise
    finally:
        asyncio.create_task(_flush_async(run))


async def _flush_async(run: ToolRun) -> None:
    """Non-blocking trace flush."""
    try:
        await asyncio.to_thread(run._flush)
    except Exception as exc:
        logger.debug("langsmith_async_flush_error", error=str(exc))


# ── Gemini call tracer ────────────────────────────────────────────────────────

@asynccontextmanager
async def trace_llm_call(
    model: str,
    prompt: str,
    session_id: str | None = None,
    parent_run_id: str | None = None,
) -> AsyncGenerator[ToolRun, None]:
    """Trace a single Gemini API call."""
    run = ToolRun(
        tool_name=f"gemini:{model}",
        session_id=session_id,
        parent_run_id=parent_run_id,
        tags=["llm", model, settings.environment],
    )
    run.add_input(prompt=prompt[:500], model=model)

    try:
        yield run
    except Exception as exc:
        run.set_error(exc)
        raise
    finally:
        asyncio.create_task(_flush_async(run))


# ── Feedback scorer ───────────────────────────────────────────────────────────

async def score_run(run_id: str, score: float, comment: str = "") -> None:
    """
    Attach a user feedback score to a LangSmith run.
    This feeds back into the prompt optimization loop.
    """
    if not _ls_enabled or _ls_client is None:
        return

    try:
        await asyncio.to_thread(
            _ls_client.create_feedback,
            run_id=run_id,
            key="user_rating",
            score=score,
            comment=comment,
        )
        logger.debug("langsmith_score_added", run_id=run_id[:8], score=score)
    except Exception as exc:
        logger.warning("langsmith_score_failed", error=str(exc))


# ── Startup initialisation ────────────────────────────────────────────────────

def init_langsmith() -> None:
    """Call once at startup to configure LangSmith."""
    _init_langsmith()