"""
Backend settings. Mirrors the MCP server config for shared infrastructure
(Postgres, Redis) and adds backend-specific fields (MCP URL, agent settings).
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

    # App
    app_name: str = "AI Analytics Backend"
    app_version: str = "1.0.0"
    debug: bool = False
    environment: str = "production"

    # Backend server
    backend_host: str = "0.0.0.0"
    backend_port: int = 8080

    # MCP server URL — backend calls this for all tool invocations
    mcp_server_url: str = "http://mcp_server:8001/mcp"
    mcp_timeout_seconds: int = 120

    # Gemini (same key as MCP server — both services share it)
    gemini_api_key: str = Field(..., alias="GEMINI_API_KEY")
    gemini_model: str = "gemini-2.5-flash"

    # Gemini rate limits (backend enforces same limits as MCP server)
    gemini_rpm_limit: int = 10
    gemini_rpd_limit: int = 1500
    gemini_tpm_limit: int = 250_000
    gemini_max_retries: int = 5
    gemini_retry_min_wait: float = 1.0
    gemini_retry_max_wait: float = 60.0
    gemini_retry_jitter_max: float = 2.0

    # PostgreSQL
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

    # Redis
    redis_host: str = "redis"
    redis_port: int = 6379
    redis_db: int = 0
    redis_password: str | None = None
    redis_max_connections: int = 50

    # Cache TTLs (seconds)
    cache_ttl_agent_run: int = 1800         # 30 min
    cache_ttl_dashboard: int = 60           # 1 min
    cache_ttl_report: int = 3600            # 1 hour

    # LangGraph agent settings
    agent_max_tool_calls: int = 10          # Max tools the analyst can call per run
    agent_timeout_seconds: int = 180        # Total pipeline timeout
    analyst_temperature: float = 0.1        # Low — analyst should be precise
    summary_temperature: float = 0.3        # Slightly higher for readable prose

    # LangSmith
    langsmith_api_key: str | None = None
    langsmith_project: str = "ai-analytics-assistant"
    langsmith_endpoint: str = "https://api.smith.langchain.com"

    # ChromaDB (needed for the RAG indexer background task)
    chromadb_host: str = "chromadb"
    chromadb_port: int = 8000

    # Log indexing
    index_batch_size: int = 50              # Messages per indexing batch
    index_interval_seconds: int = 60        # How often the background indexer runs


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()


settings = get_settings()