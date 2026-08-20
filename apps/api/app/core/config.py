from functools import lru_cache
from typing import Literal

from pydantic import AnyHttpUrl, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="AIDLENS_",
        env_file=".env",
        extra="ignore",
    )

    app_name: str = "AidLens API"
    env: Literal["development", "test", "production"] = "development"
    database_url: str = Field(description="Async SQLAlchemy PostgreSQL connection URL")
    cors_origins: list[str] = Field(default_factory=lambda: ["http://localhost:3000"])

    archive_base_url: AnyHttpUrl = AnyHttpUrl("https://usaid-archive.aiddata.org")
    archive_request_delay_seconds: float = 0.20
    archive_timeout_seconds: float = 30.0

    embedding_provider: Literal["disabled", "sentence-transformers"] = "disabled"
    embedding_model: str = "BAAI/bge-base-en-v1.5"
    embedding_dimensions: int = 768

    aidranker_provider: Literal["disabled", "sentence-transformers"] = "disabled"
    aidranker_model: str = "models/aidranker-v1.local"
    aidranker_candidate_k: int = Field(default=40, ge=10, le=100)
    aidranker_fail_open: bool = True


@lru_cache
def get_settings() -> Settings:
    return Settings()
