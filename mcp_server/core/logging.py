"""
Structured JSON logging via structlog.
Every module gets a logger with bound context (service name, environment).
"""

import logging
import sys
import structlog
from core.config import settings


def setup_logging() -> None:
    """Configure structlog + standard library logging. Call once at startup."""

    log_level = logging.DEBUG if settings.debug else logging.INFO

    # ── stdlib logging base ──────────────────────────────────────────────────
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=log_level,
    )

    # Silence noisy third-party loggers
    for noisy in ("httpx", "httpcore", "asyncio", "chromadb", "presidio_analyzer", "presidio_anonymizer"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    # ── structlog processors ─────────────────────────────────────────────────
    shared_processors: list = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
    ]

    if settings.environment == "production":
        # JSON output for log aggregators (Datadog, Loki, CloudWatch, etc.)
        renderer = structlog.processors.JSONRenderer()
    else:
        # Pretty coloured output for local dev
        renderer = structlog.dev.ConsoleRenderer(colors=True)

    structlog.configure(
        processors=shared_processors + [renderer],
        wrapper_class=structlog.make_filtering_bound_logger(log_level),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=True,
    )


def get_logger(name: str) -> structlog.BoundLogger:
    """Return a logger pre-bound with service metadata."""
    return structlog.get_logger(name).bind(
        service=settings.app_name,
        version=settings.app_version,
        env=settings.environment,
    )