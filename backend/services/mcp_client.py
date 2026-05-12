"""
Uses the MCP Python SDK client instead of raw httpx.
The SDK handles the initialize/initialized handshake automatically,
which is required before any tool calls can be made.

Each call_tool() opens a fresh session. This is slightly slower than
a persistent connection but is reliable and stateless — no session
state to manage across requests.
"""

import asyncio
from typing import Any

from mcp.client.streamable_http import streamable_http_client
from mcp import ClientSession

from core.config import settings
from core.logging import get_logger

logger = get_logger(__name__)


class MCPError(Exception):
    """Raised when an MCP tool call fails."""
    def __init__(self, tool: str, message: str) -> None:
        self.tool = tool
        super().__init__(f"MCP tool '{tool}' failed: {message}")


async def call_tool(
    tool_name: str,
    arguments: dict[str, Any],
    retries: int = 3,
) -> dict[str, Any]:
    """
    Call an MCP tool via the SDK client.

    Opens a streamable HTTP connection, performs the MCP initialization
    handshake, calls the tool, and closes the connection.

    Retries on connection errors with exponential backoff.
    """
    last_error: Exception | None = None

    for attempt in range(retries):
        try:
            async with streamable_http_client(settings.mcp_server_url) as (
                read_stream,
                write_stream,
                _,
            ):
                async with ClientSession(read_stream, write_stream) as session:
                    await session.initialize()

                    result = await session.call_tool(tool_name, arguments=arguments)

                    # The SDK returns a CallToolResult object.
                    # Extract the text content and parse as dict.
                    if result.content:
                        import json
                        for block in result.content:
                            if hasattr(block, "text"):
                                try:
                                    return json.loads(block.text)
                                except (json.JSONDecodeError, TypeError):
                                    return {"text": block.text}

                    logger.warning("mcp_tool_empty_result", tool=tool_name)
                    return {}

        except MCPError:
            raise
        except Exception as exc:
            last_error = exc
            wait = min(2 ** attempt, 10)
            if attempt < retries - 1:
                logger.warning(
                    "mcp_call_retry",
                    tool=tool_name,
                    attempt=attempt,
                    retry_in=wait,
                    error=str(exc)[:120],
                )
                await asyncio.sleep(wait)
            else:
                logger.error(
                    "mcp_call_failed",
                    tool=tool_name,
                    attempts=retries,
                    error=str(exc)[:200],
                )

    raise MCPError(
        tool_name,
        f"Failed after {retries} attempts. Last: {last_error}",
    ) from last_error


# ── Startup / shutdown stubs ──────────────────────────────────────────────────
# The SDK client manages its own connections per call — no persistent pool needed.

async def init_mcp_client() -> None:
    """Verify the MCP server is reachable at startup."""
    try:
        async with streamable_http_client(settings.mcp_server_url) as (r, w, _):
            async with ClientSession(r, w) as session:
                await session.initialize()
        logger.info("mcp_client_initialized", url=settings.mcp_server_url)
    except Exception as exc:
        # Log but don't crash — MCP server might still be starting up
        logger.warning("mcp_client_init_warning", url=settings.mcp_server_url, error=str(exc))


async def close_mcp_client() -> None:
    logger.info("mcp_client_closed")