# Provider factories: Settings → chat model / embeddings.
# Interview angle: nodes depend on these helpers, not on Groq/OpenAI SDKs
# directly — swap providers by changing env vars, not by rewriting imports
# in every agent. Trade-off: a thin factory layer vs. constructing clients
# inline (faster to write once, painful when the base URL changes).

"""Build LangChain chat and embedding clients from application settings."""

from __future__ import annotations

from typing import TYPE_CHECKING

from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from agentflow.config import Settings, get_settings

if TYPE_CHECKING:
    from langchain_core.embeddings import Embeddings
    from langchain_core.language_models.chat_models import BaseChatModel


def get_chat_model(settings: Settings | None = None) -> BaseChatModel:
    """Return the chat LLM (Groq OpenAI-compatible by default).

    Uses LLM_API_KEY / LLM_BASE_URL / LLM_MODEL — never the embedding key.
    """
    cfg = settings if settings is not None else get_settings()
    if cfg.llm_api_key is None:
        raise ValueError(
            "LLM_API_KEY is not set. Add it to .env (see .env.example)."
        )
    return ChatOpenAI(
        api_key=cfg.llm_api_key.get_secret_value(),
        base_url=cfg.llm_base_url,
        model=cfg.llm_model,
        temperature=cfg.llm_temperature,
    )


def get_embeddings(settings: Settings | None = None) -> Embeddings:
    """Return the embedding model (OpenAI or local).

    Separate from the chat client so Groq can own chat while OpenAI/local
    own vectors — mixing keys here is a common production footgun.
    """
    cfg = settings if settings is not None else get_settings()
    if cfg.embedding_provider == "local":
        return _local_embeddings(cfg)
    if cfg.embedding_api_key is None:
        raise ValueError(
            "EMBEDDING_API_KEY is not set. Add it to .env (see .env.example), "
            "or set EMBEDDING_PROVIDER=local with the [local] extra installed."
        )
    return OpenAIEmbeddings(
        api_key=cfg.embedding_api_key.get_secret_value(),
        model=cfg.embedding_model,
    )


def _local_embeddings(cfg: Settings) -> Embeddings:
    """Lazy-import HuggingFace embeddings so core installs stay slim."""
    try:
        from langchain_huggingface import HuggingFaceEmbeddings
    except ImportError as exc:
        raise ImportError(
            "Local embeddings require the [local] extra: "
            'pip install -e ".[local]"'
        ) from exc
    return HuggingFaceEmbeddings(model_name=cfg.local_embedding_model)


__all__ = ["get_chat_model", "get_embeddings"]
