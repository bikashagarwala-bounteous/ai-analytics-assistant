"""
chat session lifecycle and message persistence.
Each user message is stored in PostgreSQL, intent-classified via Gemini, and queued for ChromaDB indexing.
"""

import asyncio
import json
import uuid
from datetime import datetime

from sqlalchemy import text

from db.connections import get_db, get_redis
from db.models import ChatSession, ChatMessage
from core.config import settings
from core.logging import get_logger

logger = get_logger(__name__)

# Redis key for the indexing queue — background task drains this
_INDEX_QUEUE_KEY = "queue:index_messages"


async def get_or_create_session(
    session_id: str | None,
    user_id: str = "anonymous",
    channel: str = "web",
) -> str:
    """
    Return an existing session_id or create a new session row.
    Used at the start of every /chat request.
    """
    if session_id:
        # Verify it exists
        async with get_db() as db:
            row = await db.execute(
                text("SELECT session_id FROM chat_sessions WHERE session_id = :sid"),
                {"sid": session_id},
            )
            if row.fetchone():
                return session_id
        logger.warning("session_not_found_creating_new", provided_id=session_id)

    new_id = str(uuid.uuid4())
    async with get_db() as db:
        await db.execute(
            text("""
                INSERT INTO chat_sessions (session_id, user_id, channel, started_at, created_at)
                VALUES (:sid, :uid, :channel, NOW(), NOW())
            """),
            {"sid": new_id, "uid": user_id, "channel": channel},
        )
    logger.debug("session_created", session_id=new_id)
    return new_id


async def store_message(
    session_id: str,
    role: str,
    content: str,
    latency_ms: int | None = None,
    token_count: int | None = None,
    langsmith_run_id: str | None = None,
) -> str:
    """
    Persist a chat message and return its UUID.
    Queues the message for background ChromaDB indexing.
    """
    message_id = str(uuid.uuid4())
    async with get_db() as db:
        await db.execute(
            text("""
                INSERT INTO chat_messages
                    (id, session_id, role, content, latency_ms, token_count,
                     langsmith_run_id, created_at)
                VALUES
                    (:id, :sid, :role, :content, :latency_ms, :token_count,
                     :ls_run_id, NOW())
            """),
            {
                "id": message_id,
                "sid": session_id,
                "role": role,
                "content": content,
                "latency_ms": latency_ms,
                "token_count": token_count,
                "ls_run_id": langsmith_run_id,
            },
        )

    # Classify intent and index — both non-blocking
    if role == "user":
        await _enqueue_for_indexing(message_id, session_id, content)
        asyncio.create_task(_classify_intent(message_id, content))

    return message_id


async def classify_and_store_intent(
    message_id: str,
    intent: str,
    confidence: float,
    is_failure: bool,
    raw_output: dict | None = None,
) -> None:
    """Store the intent classification for a user message."""
    async with get_db() as db:
        await db.execute(
            text("""
                INSERT INTO intent_classifications
                    (id, message_id, intent, confidence, is_failure,
                     raw_classifier_output, created_at)
                VALUES
                    (:id, :msg_id, :intent, :confidence, :is_failure, :raw, NOW())
                ON CONFLICT DO NOTHING
            """),
            {
                "id": str(uuid.uuid4()),
                "msg_id": message_id,
                "intent": intent,
                "confidence": confidence,
                "is_failure": is_failure,
                "raw": json.dumps(raw_output or {}),
            },
        )


async def close_session(session_id: str, resolved: bool = False) -> None:
    """Mark a session as ended."""
    async with get_db() as db:
        await db.execute(
            text("""
                UPDATE chat_sessions
                SET ended_at = NOW(), resolved = :resolved
                WHERE session_id = :sid
            """),
            {"sid": session_id, "resolved": resolved},
        )


async def _enqueue_for_indexing(
    message_id: str, session_id: str, content: str
) -> None:
    """Push message metadata to Redis list for the background indexer."""
    try:
        import json
        redis = get_redis()
        payload = json.dumps({
            "message_id": message_id,
            "session_id": session_id,
            "content": content,
            "queued_at": datetime.utcnow().isoformat(),
        })
        await redis.rpush(_INDEX_QUEUE_KEY, payload)
    except Exception as exc:
        # Never let indexing queue failures break the chat flow
        logger.warning("index_enqueue_failed", message_id=message_id, error=str(exc))


async def _classify_intent(message_id: str, content: str) -> None:
    """
    Classify the intent of a user message using Gemini.

    The LLM derives the intent name from the message itself — no fixed list.
    It also determines whether the message represents a failure case
    (unclear request, out of scope, or something a bot would typically fail on).

    Runs as a background task and never blocks the chat response.
    """
    import asyncio
    import os
    from google import genai
    from google.genai import types

    _PROMPT = """\
Analyse this customer support message and return a JSON object with exactly these fields:

{{
  "intent": "<snake_case intent name>",
  "confidence": <float 0.0-1.0>,
  "is_failure": <true|false>,
  "failure_reason": "<brief reason if is_failure is true, else null>"
}}

Rules:
- intent: a short snake_case label that describes what the user wants.
  Derive it from the message — do not use a fixed list.
  Examples: password_reset, track_order, cancel_subscription, billing_question,
  report_bug, request_feature, general_complaint, unclear_request.
  Use your judgement for anything else.
- confidence: how confident you are in this classification (1.0 = certain).
- is_failure: true if the message is likely to be hard for a chatbot to resolve.
  Mark true when: the request is ambiguous, emotionally charged, requires human
  judgement, is outside typical bot scope, or the message is too short/vague.
- failure_reason: one sentence explaining why it is a failure, or null.

Message: {message}

Reply with the JSON object only. No other text."""

    try:
        client = genai.Client(api_key=os.getenv("GEMINI_API_KEY", ""))

        _RETRYABLE = ("500", "503", "internal", "unavailable", "overloaded")
        response = None
        for attempt in range(3):
            try:
                response = await asyncio.to_thread(
                    client.models.generate_content,
                    model=settings.gemini_model,
                    contents=[_PROMPT.format(message=content)],
                    config=types.GenerateContentConfig(
                        temperature=0.0,
                        max_output_tokens=100,
                    ),
                )
                break
            except Exception as exc:
                if any(k in str(exc).lower() for k in _RETRYABLE) and attempt < 2:
                    await asyncio.sleep(2 ** attempt)
                else:
                    raise
        if response is None:
            return

        import json
        raw = (response.text or "").strip()
        if not raw:
            logger.warning("intent_classification_empty_response", message_id=message_id)
            return

        if raw.startswith("```"):
            raw = raw.split("\n", 1)[1].rsplit("```", 1)[0].strip()

        data = json.loads(raw)

        intent = str(data.get("intent", "unknown")).lower().replace(" ", "_")[:64]
        confidence = float(data.get("confidence", 0.5))
        is_failure = bool(data.get("is_failure", False))

        await classify_and_store_intent(
            message_id=message_id,
            intent=intent,
            confidence=confidence,
            is_failure=is_failure,
            raw_output=data,
        )

        logger.debug(
            "intent_classified",
            message_id=message_id,
            intent=intent,
            confidence=confidence,
            is_failure=is_failure,
        )

    except Exception as exc:
        logger.warning(
            "intent_classification_failed",
            message_id=message_id,
            error=str(exc)[:120],
        )