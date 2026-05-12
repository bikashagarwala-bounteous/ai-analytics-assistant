"""
Chat interface with the analytics agent.
Streams the response token by token and shows tool execution status.
"""

import time
import uuid

import streamlit as st

from components.api_client import stream_chat, post


# ── Suggested queries shown on first load ─────────────────────────────────────
_SUGGESTIONS = [
    "Why did engagement drop last week?",
    "What are the top failure intents this month?",
    "Show me message volume trends for the past 30 days",
    "Which intents have improved the most recently?",
    "Are there any anomalies in the failure rate?",
]


def render():
    col_title, col_new = st.columns([5, 1])
    col_title.title("Analytics Chat")
    col_title.caption("Ask questions about your chatbot's performance in plain English.")
    if col_new.button("New Chat", use_container_width=True):
        st.session_state.messages = []
        st.session_state.session_id = None
        st.session_state.last_message_id = None
        st.rerun()

    # ── Session state ─────────────────────────────────────────────────────────
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "session_id" not in st.session_state:
        st.session_state.session_id = None
    if "last_message_id" not in st.session_state:
        st.session_state.last_message_id = None

    # ── Suggestion chips (only when conversation is empty) ────────────────────
    if not st.session_state.messages:
        st.markdown("**Try asking:**")
        cols = st.columns(len(_SUGGESTIONS))
        for col, suggestion in zip(cols, _SUGGESTIONS):
            if col.button(suggestion, use_container_width=True, key=f"sug_{suggestion[:20]}"):
                st.session_state.pending_query = suggestion
                st.rerun()

    # ── Conversation history ──────────────────────────────────────────────────
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("meta"):
                meta = msg["meta"]
                st.caption(
                    f"Tools used: {meta.get('tool_calls_count', 0)} · "
                    f"Duration: {meta.get('duration_ms', 0)}ms · "
                    f"Run ID: `{meta.get('run_id', '')[:8]}`"
                )
            if (
                msg["role"] == "assistant"
                and msg.get("meta")
                and st.session_state.get("session_id")
            ):
                _render_feedback_widget(
                    session_id=st.session_state.session_id,
                    message_id=msg["meta"].get("message_id") or msg["meta"].get("run_id", ""),
                )

    # ── Handle pending query from suggestion chips ────────────────────────────
    query = st.session_state.pop("pending_query", None)

    # ── Chat input ────────────────────────────────────────────────────────────
    if typed := st.chat_input("Ask about your chatbot metrics..."):
        query = typed

    if not query:
        return

    # ── Display user message ──────────────────────────────────────────────────
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    # ── Stream assistant response ─────────────────────────────────────────────
    with st.chat_message("assistant"):
        status_placeholder = st.empty()
        text_placeholder = st.empty()
        response_text = ""
        meta = {}

        _STATUS_LABELS = {
            "analyst_start":    "Analyst is planning tool calls...",
            "analyst_complete": "Analysis complete. Writing summary...",
            "summary_start":    "Generating response...",
        }

        t0 = time.perf_counter()
        try:
            for event in stream_chat(query, st.session_state.session_id):
                etype = event.get("type")

                if etype == "event":
                    name = event.get("name", "")
                    if name == "heartbeat":
                        continue
                    if label := _STATUS_LABELS.get(name):
                        status_placeholder.info(label)

                elif etype == "chunk":
                    response_text += event.get("text", "")
                    text_placeholder.markdown(response_text + "▌")

                elif etype == "done":
                    status_placeholder.empty()
                    text_placeholder.markdown(response_text)
                    st.session_state.session_id = event.get("session_id")
                    meta = {
                        "run_id": event.get("run_id", ""),
                        "message_id": event.get("message_id", ""),
                        "duration_ms": event.get("duration_ms", 0),
                        "tool_calls_count": event.get("tool_calls_count", 0),
                    }
                    st.session_state.last_message_id = event.get("message_id")

                elif etype == "error":
                    status_placeholder.empty()
                    st.error(f"Error: {event.get('message', 'Unknown error')}")
                    return

        except Exception as e:
            st.error(f"Connection error: {e}")
            return

        # ── Show metadata ─────────────────────────────────────────────────────
        duration = int((time.perf_counter() - t0) * 1000)
        st.caption(f"Duration: {duration}ms · Run: `{meta.get('run_id','')[:8]}`")

        # ── Feedback widget ───────────────────────────────────────────────────
        if st.session_state.session_id and response_text:
            _render_feedback_widget(
                session_id=st.session_state.session_id,
                message_id=meta.get("message_id") or meta.get("run_id", str(uuid.uuid4())),
            )

    st.session_state.messages.append({
        "role": "assistant",
        "content": response_text,
        "meta": meta,
    })


def _render_feedback_widget(session_id: str, message_id: str):
    """Inline thumbs feedback rendered after each assistant response."""
    key_prefix = f"fb_{message_id[:8]}"

    if st.session_state.get(f"{key_prefix}_submitted"):
        st.caption("Feedback recorded.")
        return

    # Dislike was clicked — show comment form before submitting
    if st.session_state.get(f"{key_prefix}_awaiting_comment"):
        comment = st.text_input(
            "What went wrong? (optional)",
            key=f"{key_prefix}_comment_input",
            placeholder="e.g. Wrong metric, outdated data, unclear response…",
        )
        col_send, col_skip, _ = st.columns([1, 1, 5])
        if col_send.button("Send", key=f"{key_prefix}_send"):
            _submit_feedback(
                session_id, message_id,
                rating=2, sentiment="negative", helpful=False,
                comment=comment or None,
            )
            st.session_state[f"{key_prefix}_submitted"] = True
            st.session_state.pop(f"{key_prefix}_awaiting_comment", None)
            st.rerun()
        if col_skip.button("Skip", key=f"{key_prefix}_skip"):
            _submit_feedback(
                session_id, message_id,
                rating=2, sentiment="negative", helpful=False,
            )
            st.session_state[f"{key_prefix}_submitted"] = True
            st.session_state.pop(f"{key_prefix}_awaiting_comment", None)
            st.rerun()
        return

    c1, c2, _ = st.columns([1, 1, 6])
    if c1.button("👍", key=f"{key_prefix}_up"):
        _submit_feedback(session_id, message_id, rating=5, sentiment="positive", helpful=True)
        st.session_state[f"{key_prefix}_submitted"] = True
        st.rerun()
    if c2.button("👎", key=f"{key_prefix}_down"):
        st.session_state[f"{key_prefix}_awaiting_comment"] = True
        st.rerun()


def _submit_feedback(
    session_id: str,
    message_id: str,
    rating: int,
    sentiment: str,
    helpful: bool,
    comment: str | None = None,
) -> None:
    post("/feedback", {
        "session_id": session_id,
        "message_id": message_id,
        "rating": rating,
        "sentiment": sentiment,
        "response_was_helpful": helpful,
        "comment": comment,
    })

render()