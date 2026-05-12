"""
Settings for the indexer service.
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
    app_name: str = "AI Analytics Indexer"
    environment: str = "production"
    debug: bool = False

    # Gemini
    gemini_api_key: str = Field(..., alias="GEMINI_API_KEY")
    gemini_embedding_model: str = "gemini-embedding-001"
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

    # ChromaDB
    chromadb_host: str = "chromadb"
    chromadb_port: int = 8000
    chromadb_collection_conversations: str = "conversations"

    # Indexer behaviour
    batch_size: int = 50            # Messages to embed per Gemini call
    poll_interval_seconds: int = 10 # How often to drain the queue when idle
    max_queue_age_hours: int = 24   # Reindex messages missed for this long
    embedding_dimension: int = 3072
    embedding_cache_ttl: int = 86_400

    # Redis keys (must match session_service.py)
    index_queue_key: str = "queue:index_messages"
    embedding_cache_prefix: str = "cache:embed:"
    rate_limit_rpm_key: str = "ratelimit:gemini:rpm"
    rate_limit_rpd_key: str = "ratelimit:gemini:rpd"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()


settings = get_settings()