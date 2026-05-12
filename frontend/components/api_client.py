"""
HTTP client the Streamlit pages use to talk to the FastAPI backend.
Handles streaming SSE responses and regular JSON calls.
"""

import json
import os
import time
from typing import Any, Generator

import httpx
import streamlit as st

BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8080")
TIMEOUT = httpx.Timeout(connect=5.0, read=300.0, write=10.0, pool=5.0)


# ── Regular JSON calls ────────────────────────────────────────────────────────

def get(path: str, params: dict | None = None) -> dict | list | None:
    try:
        r = httpx.get(f"{BACKEND_URL}{path}", params=params, timeout=TIMEOUT)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        st.error(f"Request failed: {e}")
        return None


def post(path: str, body: dict) -> dict | None:
    try:
        r = httpx.post(f"{BACKEND_URL}{path}", json=body, timeout=TIMEOUT)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        st.error(f"Request failed: {e}")
        return None


# ── Streaming SSE ─────────────────────────────────────────────────────────────

def stream_chat(query: str, session_id: str | None = None) -> Generator[dict, None, None]:
    """
    Stream an SSE chat response from the backend.
    Yields parsed event dicts:
      {"type": "chunk",  "text": "..."}
      {"type": "event",  "name": "analyst_start" | "analyst_complete" | ...}
      {"type": "done",   "run_id": "...", "session_id": "...", "duration_ms": ...}
      {"type": "error",  "message": "..."}
    """
    body = {"query": query, "session_id": session_id, "stream": True}

    with httpx.stream(
        "POST",
        f"{BACKEND_URL}/chat",
        json=body,
        timeout=TIMEOUT,
    ) as response:
        for line in response.iter_lines():
            if not line.startswith("data: "):
                continue
            try:
                yield json.loads(line[6:])
            except json.JSONDecodeError:
                continue