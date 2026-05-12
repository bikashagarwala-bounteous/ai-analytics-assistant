"""
multi-layer input safety for MCP tool inputs/outputs.
Checks: prompt injection, jailbreak, SQL injection, PII redaction (Presidio), toxic content.
Call check_input(); blocking violations raise GuardrailError. Use result.sanitized_text.
"""

import re
import time
import hashlib
from typing import NamedTuple

from core.config import settings
from core.logging import get_logger
from schemas import (
    GuardrailCheckResult,
    GuardrailViolation,
    ThreatType,
    SeverityLevel,
)

logger = get_logger(__name__)


class GuardrailError(Exception):
    """Raised when a guardrail check fails and the request should be blocked."""

    def __init__(self, result: GuardrailCheckResult) -> None:
        self.result = result
        types = [v.threat_type.value for v in result.violations]
        super().__init__(f"Guardrail violation(s): {', '.join(types)}")


# ── Prompt Injection Patterns ─────────────────────────────────────────────────
# Collected from known red-teaming research and OWASP LLM Top 10

_INJECTION_PATTERNS: list[re.Pattern] = [
    # Direct instruction override
    re.compile(r"ignore\s+(all\s+)?(previous|prior|above)\s+instructions?", re.I),
    re.compile(r"disregard\s+(all\s+)?(previous|prior)\s+(instructions?|prompts?)", re.I),
    re.compile(r"forget\s+(everything|all)\s+(you|i)\s+(know|told|said)", re.I),

    # Role hijacking
    re.compile(r"you\s+are\s+now\s+(an?\s+)?(evil|unrestricted|jailbroken|DAN)", re.I),
    re.compile(r"act\s+as\s+(if\s+you\s+(are|were)\s+)?(not|no longer)\s+an?\s+AI", re.I),
    re.compile(r"pretend\s+(you\s+have\s+no\s+(restrictions?|rules?|guidelines?))", re.I),

    # System prompt extraction
    re.compile(r"(print|show|reveal|output|repeat|tell me)\s+(your\s+)?(system\s+prompt|instructions?|context)", re.I),
    re.compile(r"what\s+(are|were)\s+your\s+(original\s+)?(instructions?|system\s+prompts?)", re.I),

    # Indirect / encoded injection
    re.compile(r"base64[_\s]decode", re.I),
    re.compile(r"\\u00[0-9a-f]{2}", re.I),   # Unicode escape sequences

    # Tool / function override
    re.compile(r"<\s*tool[_\s]*call\s*>", re.I),
    re.compile(r"<\s*function[_\s]*call\s*>", re.I),
    re.compile(r"\[INST\]|\[/INST\]|<\|im_start\|>|<\|im_end\|>"),  # Model-specific tokens

    # Data exfiltration via prompt
    re.compile(r"send\s+(all|this|the)\s+(data|context|results?)\s+to\s+http", re.I),
]

# ── Jailbreak Patterns ────────────────────────────────────────────────────────

_JAILBREAK_PATTERNS: list[re.Pattern] = [
    re.compile(r"\bDAN\b"),                     # "Do Anything Now"
    re.compile(r"jailbreak", re.I),
    re.compile(r"developer\s+mode", re.I),
    re.compile(r"god\s+mode", re.I),
    re.compile(r"bypass\s+(all\s+)?(safety|filter|restriction|guardrail)", re.I),
    re.compile(r"no\s+(ethical|moral|safety)\s+(restrictions?|constraints?)", re.I),
]

# ── SQL Injection Patterns ────────────────────────────────────────────────────

_SQL_PATTERNS: list[re.Pattern] = [
    re.compile(r"(;\s*DROP\s+TABLE|;\s*DELETE\s+FROM|;\s*TRUNCATE)", re.I),
    re.compile(r"UNION\s+(ALL\s+)?SELECT", re.I),
    re.compile(r"OR\s+['\"]?\s*1\s*=\s*1", re.I),
    re.compile(r"--\s*$", re.M),               # SQL comment at end of line
    re.compile(r"/\*.*?\*/", re.DOTALL),       # Block comment
    re.compile(r"(xp_cmdshell|exec\s*\()", re.I),
]

# ── Toxic Content ─────────────────────────────────────────────────────────────
# Minimal list — expand or replace with a proper classifier in production

_TOXIC_KEYWORDS: frozenset[str] = frozenset([
    "kill yourself", "kys", "go die",
])


# ── PII Detector (Presidio) ───────────────────────────────────────────────────

_presidio_analyzer = None
_presidio_anonymizer = None


def _get_presidio():
    """Lazy-load Presidio to avoid slowing startup."""
    import logging
    logging.getLogger("presidio-analyzer").setLevel(logging.ERROR)
    logging.getLogger("presidio_analyzer").setLevel(logging.ERROR)
    logging.getLogger("presidio-anonymizer").setLevel(logging.ERROR)
    logging.getLogger("presidio_anonymizer").setLevel(logging.ERROR)
    global _presidio_analyzer, _presidio_anonymizer
    if _presidio_analyzer is None:
        try:
            from presidio_analyzer import AnalyzerEngine
            from presidio_anonymizer import AnonymizerEngine
            from presidio_analyzer import RecognizerRegistry
            from presidio_analyzer.nlp_engine import SpacyNlpEngine, NerModelConfiguration

            ner_config = NerModelConfiguration(
                labels_to_ignore=[
                    "CARDINAL", "ORDINAL", "QUANTITY", "PERCENT",
                    "DATE", "TIME", "MONEY", "WORK_OF_ART",
                    "EVENT", "FAC", "LAW", "LANGUAGE", "NORP",
                ]
            )
            nlp_engine = SpacyNlpEngine(
                models=[{"lang_code": "en", "model_name": "en_core_web_sm"}],
                ner_model_configuration=ner_config,
            )
            registry = RecognizerRegistry(supported_languages=["en"])
            registry.load_predefined_recognizers(languages=["en"])
            _presidio_analyzer = AnalyzerEngine(
                nlp_engine=nlp_engine,
                registry=registry,
                default_score_threshold=0.75,
            )
            _presidio_anonymizer = AnonymizerEngine()
            logger.debug("presidio_loaded")
        except ImportError:
            logger.warning("presidio_not_installed_pii_detection_disabled")
    return _presidio_analyzer, _presidio_anonymizer


def _detect_and_redact_pii(text: str) -> tuple[list[str], str]:
    analyzer, anonymizer = _get_presidio()
    if analyzer is None:
        return [], text

    results = analyzer.analyze(text=text, language="en")
    results = [r for r in results if r.score >= 0.75]

    entity_types = list({r.entity_type for r in results})
    if not results:
        return [], text

    from presidio_anonymizer.entities import OperatorConfig
    anonymized = anonymizer.anonymize(
        text=text,
        analyzer_results=results,
        operators={"DEFAULT": OperatorConfig("replace", {"new_value": "<REDACTED>"})},
    )
    return entity_types, anonymized.text


# ── Main check function ───────────────────────────────────────────────────────

async def check_input(text: str, context: str = "tool_input") -> GuardrailCheckResult:
    """
    Run all guardrail checks on an input string.

    Args:
        text:     The text to check (user query, tool argument, etc.)
        context:  Human-readable label for logging (e.g. "chat_message", "query_param")

    Returns:
        GuardrailCheckResult with is_safe, violations, and sanitized_text.
    """
    t0 = time.perf_counter()
    violations: list[GuardrailViolation] = []
    sanitized = text

    # ── 1. Prompt injection ───────────────────────────────────────────────────
    for pattern in _INJECTION_PATTERNS:
        match = pattern.search(text)
        if match:
            violations.append(GuardrailViolation(
                threat_type=ThreatType.PROMPT_INJECTION,
                severity=SeverityLevel.CRITICAL,
                description=f"Prompt injection pattern detected: '{match.group()[:60]}'",
                detected_pattern=pattern.pattern[:80],
            ))
            break   # One injection violation is enough to block

    # ── 2. Jailbreak ─────────────────────────────────────────────────────────
    for pattern in _JAILBREAK_PATTERNS:
        match = pattern.search(text)
        if match:
            violations.append(GuardrailViolation(
                threat_type=ThreatType.JAILBREAK_ATTEMPT,
                severity=SeverityLevel.HIGH,
                description=f"Jailbreak attempt: '{match.group()[:60]}'",
                detected_pattern=pattern.pattern[:80],
            ))
            break

    # ── 3. SQL injection ──────────────────────────────────────────────────────
    for pattern in _SQL_PATTERNS:
        match = pattern.search(text)
        if match:
            violations.append(GuardrailViolation(
                threat_type=ThreatType.SQL_INJECTION,
                severity=SeverityLevel.HIGH,
                description=f"SQL injection pattern: '{match.group()[:60]}'",
                detected_pattern=pattern.pattern[:80],
            ))
            # Sanitize: strip the dangerous fragment
            sanitized = pattern.sub("", sanitized)
            break

    # ── 4. PII detection ──────────────────────────────────────────────────────
    pii_entities, sanitized = _detect_and_redact_pii(sanitized)
    if pii_entities:
        violations.append(GuardrailViolation(
            threat_type=ThreatType.PII_DETECTED,
            severity=SeverityLevel.MEDIUM,
            description=f"PII detected and redacted: {', '.join(pii_entities[:5])}",
            pii_entities=pii_entities[:10],
        ))

    # ── 5. Toxic content ──────────────────────────────────────────────────────
    text_lower = text.lower()
    for keyword in _TOXIC_KEYWORDS:
        if keyword in text_lower:
            violations.append(GuardrailViolation(
                threat_type=ThreatType.TOXIC_CONTENT,
                severity=SeverityLevel.HIGH,
                description="Toxic content detected",
            ))
            break

    duration_ms = (time.perf_counter() - t0) * 1000

    # Only prompt injection, jailbreak, and SQL injection block the request.
    # PII is redacted but allowed through. Toxic content blocks.
    blocking_types = {
        ThreatType.PROMPT_INJECTION,
        ThreatType.JAILBREAK_ATTEMPT,
        ThreatType.SQL_INJECTION,
        ThreatType.TOXIC_CONTENT,
    }
    is_safe = not any(v.threat_type in blocking_types for v in violations)

    if violations:
        logger.warning(
            "guardrail_violation",
            context=context,
            is_safe=is_safe,
            violations=[v.threat_type.value for v in violations],
            duration_ms=round(duration_ms, 2),
        )
    else:
        logger.debug("guardrail_pass", context=context, duration_ms=round(duration_ms, 2))

    return GuardrailCheckResult(
        is_safe=is_safe,
        violations=violations,
        sanitized_text=sanitized,
        original_text=text,
        check_duration_ms=round(duration_ms, 2),
    )


async def check_output(text: str) -> str:
    """
    Light check on LLM output — redact any PII that leaked through.
    Returns the sanitized output.
    """
    _, sanitized = _detect_and_redact_pii(text)
    return sanitized


def validate_tool_args(**kwargs) -> dict:
    """
    Run guardrails on all string arguments to a tool.
    Returns sanitized kwargs. Raises GuardrailError if any arg is unsafe.
    """
    import asyncio

    async def _check_all():
        results = {}
        for key, val in kwargs.items():
            if isinstance(val, str) and val.strip():
                result = await check_input(val, context=f"tool_arg:{key}")
                if not result.is_safe:
                    raise GuardrailError(result)
                results[key] = result.sanitized_text
            else:
                results[key] = val
        return results

    # Run synchronously when called from a sync context
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            # We're inside an async context — use create_task instead
            return kwargs   # Caller should use async validate_tool_args_async
        return loop.run_until_complete(_check_all())
    except RuntimeError:
        return kwargs


async def validate_tool_args_async(**kwargs) -> dict:
    """Async version for use inside async tool functions."""
    results = {}
    for key, val in kwargs.items():
        if isinstance(val, str) and val.strip():
            result = await check_input(val, context=f"tool_arg:{key}")
            if not result.is_safe:
                raise GuardrailError(result)
            results[key] = result.sanitized_text
        else:
            results[key] = val
    return results