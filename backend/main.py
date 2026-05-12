"""
Backend FastAPI application. Startup: logging → postgres → redis → MCP client → LangGraph.
Routes: /chat, /analyze, /report.
"""

from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from core.config import settings
from core.logging import setup_logging, get_logger
from db.connections import init_postgres, close_postgres, init_redis, close_redis
from services.mcp_client import init_mcp_client, close_mcp_client
from agents.graph import get_compiled_graph
from api.routes.chat import router as chat_router
from api.routes.analyze import router as analyze_router
from api.routes.report import router as report_router
from api.routes.techniques import router as techniques_router

setup_logging()
logger = get_logger(__name__)

# ── LangSmith — enable LangGraph auto-tracing if API key is present ───────────
if settings.langsmith_api_key:
    import os
    os.environ.setdefault("LANGCHAIN_API_KEY", settings.langsmith_api_key)
    os.environ.setdefault("LANGCHAIN_TRACING_V2", "true")
    os.environ.setdefault("LANGCHAIN_PROJECT", settings.langsmith_project)
    os.environ.setdefault("LANGCHAIN_ENDPOINT", settings.langsmith_endpoint)
    logger.info("langsmith_tracing_enabled", project=settings.langsmith_project)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown sequence."""
    logger.info("backend_starting", version=settings.app_version)

    await init_postgres()
    logger.info("postgres_ready")

    await init_redis()
    logger.info("redis_ready")

    await init_mcp_client()
    logger.info("mcp_client_ready")

    # Pre-compile the graph so the first request isn't slow
    get_compiled_graph()
    logger.info("langgraph_ready")

    logger.info("backend_ready", port=settings.backend_port)
    yield

    logger.info("backend_shutting_down")
    await close_mcp_client()
    await close_redis()
    await close_postgres()
    logger.info("backend_stopped")


app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    lifespan=lifespan,
    docs_url="/docs" if settings.debug else None,
    redoc_url=None,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"] if settings.debug else ["http://localhost:8501"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


app.include_router(chat_router)
app.include_router(analyze_router)
app.include_router(report_router)
app.include_router(techniques_router)


@app.get("/health")
async def health():
    return {"status": "healthy", "version": settings.app_version}


if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host=settings.backend_host,
        port=settings.backend_port,
        reload=settings.debug,
        log_level="debug" if settings.debug else "info",
    )