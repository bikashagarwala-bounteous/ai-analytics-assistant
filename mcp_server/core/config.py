"""
Centralised settings — loaded once from environment / .env file.
Every other module imports `settings` from here.
"""

from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ── App ──────────────────────────────────────────────────────────────────
    app_name: str = "AI Analytics MCP Server"
    app_version: str = "1.0.0"
    debug: bool = False
    environment: str = "production"

    # ── MCP Server ───────────────────────────────────────────────────────────
    mcp_host: str = "0.0.0.0"
    mcp_port: int = 8001
    mcp_path: str = "/mcp"                  # Streamable HTTP mount path

    # ── Gemini ───────────────────────────────────────────────────────────────
    gemini_api_key: str = Field(..., alias="GEMINI_API_KEY")
    gemini_model: str = "gemini-2.5-flash"
    gemini_embedding_model: str = "gemini-embedding-001"

    # Free-tier rate limits (conservative defaults — tune per your quota)
    gemini_rpm_limit: int = 10              # Requests per minute
    gemini_rpd_limit: int = 1500            # Requests per day
    gemini_tpm_limit: int = 250_000         # Tokens per minute

    # Retry config
    gemini_max_retries: int = 5
    gemini_retry_min_wait: float = 1.0      # seconds
    gemini_retry_max_wait: float = 60.0     # seconds
    gemini_retry_jitter_max: float = 2.0    # seconds of random jitter

    # ── PostgreSQL ───────────────────────────────────────────────────────────
    postgres_host: str = "postgres"
    postgres_port: int = 5432
    postgres_db: str = "analytics"
    postgres_user: str = "analytics_user"
    postgres_password: str = Field(..., alias="POSTGRES_PASSWORD")
    postgres_pool_size: int = 10
    postgres_max_overflow: int = 20

    @property
    def postgres_dsn(self) -> str:
        return (
            f"postgresql+asyncpg://{self.postgres_user}:{self.postgres_password}"
            f"@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
        )

    # ── Redis ────────────────────────────────────────────────────────────────
    redis_host: str = "redis"
    redis_port: int = 6379
    redis_db: int = 0
    redis_password: str | None = None
    redis_max_connections: int = 50

    # Cache TTLs (seconds)
    cache_ttl_llm_response: int = 3600        # 1 hour
    cache_ttl_embedding: int = 86_400         # 24 hours
    cache_ttl_analysis: int = 1800            # 30 minutes
    cache_ttl_rag_retrieval: int = 900        # 15 minutes
    cache_ttl_stream_partial: int = 300       # 5 minutes

    # ── ChromaDB ─────────────────────────────────────────────────────────────
    chromadb_host: str = "chromadb"
    chromadb_port: int = 8000
    chromadb_collection_conversations: str = "conversations"
    chromadb_collection_intents: str = "intents"

    # ── Embedding ────────────────────────────────────────────────────────────
    embedding_batch_size: int = 50            # Max texts per embed_content call
    embedding_dimension: int = 3072           # gemini-embedding-001 output dim

    # ── LangSmith ────────────────────────────────────────────────────────────
    # Optional. If LANGCHAIN_API_KEY is not set, tracing is silently disabled.
    langsmith_api_key: str | None = None
    langsmith_project: str = "ai-analytics-assistant"
    langsmith_endpoint: str = "https://api.smith.langchain.com"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return singleton settings — cached after first call."""
    return Settings()


# Convenience import: `from core.config import settings`
settings = get_settings()