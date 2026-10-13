# Typed runtime config: env vars → Settings → the rest of the app.
# Interview angle: fail fast at startup with clear validation errors
# instead of KeyError deep inside a LangGraph node mid-request.
# Trade-off: pydantic-settings adds a dep; bare os.getenv is fewer
# packages but no types, defaults, or .env loading in one place.

"""Application settings loaded from environment variables and optional `.env`."""

from functools import lru_cache
from typing import Literal

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Single source of truth for knobs the API, graph, and tools share.

    Field names map to env vars (case-insensitive). Nested delimiters are
    unused on purpose — flat names stay easy to set in Docker/K8s.

    LLM (chat) and embeddings are configured separately because they can
    use different providers — e.g., Groq for chat, OpenAI or local for
    embeddings.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",  # ignore unknown env keys so CI vars do not break boot
    )

    # --- API ---
    api_host: str = "0.0.0.0"
    api_port: int = Field(default=8000, ge=1, le=65535)
    cors_origins: str = "*"

    # --- LLM (chat) — Groq via OpenAI-compatible API ---
    # Env: LLM_API_KEY, LLM_BASE_URL, LLM_MODEL (separate from embeddings).
    llm_api_key: SecretStr | None = None
    llm_base_url: str = "https://api.groq.com/openai/v1"
    llm_model: str = "openai/gpt-oss-120b"
    llm_temperature: float = Field(default=0.0, ge=0.0, le=2.0)

    # --- Embeddings — OpenAI or local (not Groq) ---
    # Env: EMBEDDING_PROVIDER, EMBEDDING_API_KEY, EMBEDDING_MODEL.
    embedding_provider: Literal["openai", "local"] = "openai"
    embedding_api_key: SecretStr | None = None
    embedding_model: str = "text-embedding-3-small"
    # Used only when embedding_provider == "local" (`[local]` extra).
    local_embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"

    # --- Vector store (Chroma) ---
    chroma_persist_dir: str = "chroma_db"
    chroma_collection: str = "agentflow_docs"

    # --- Tools ---
    web_search_api_key: SecretStr | None = None
    retrieve_top_k: int = Field(default=4, ge=1, le=20)

    # --- Observability ---
    langchain_tracing_v2: bool = False
    langchain_api_key: SecretStr | None = None
    langchain_project: str = "agentflow"

    def cors_origin_list(self) -> list[str]:
        """Split `cors_origins` into a list FastAPI middleware can consume."""
        raw = self.cors_origins.strip()
        if raw == "*":
            return ["*"]
        return [part.strip() for part in raw.split(",") if part.strip()]


@lru_cache
def get_settings() -> Settings:
    """Cached Settings instance — one parse of env per process.

    FastAPI deps and graph nodes should call this instead of constructing
    Settings() repeatedly (re-reads .env and rebuilds the model each time).
    Call `get_settings.cache_clear()` in tests after monkeypatching env.
    """
    return Settings()


__all__ = ["Settings", "get_settings"]