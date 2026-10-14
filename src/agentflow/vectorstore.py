# Chroma persistence + similarity search over embedded chunks.
# Interview angle: wrap chromadb behind a small API so the retrieve node
# (and later MCP tool) never touch collection names or persist paths —
# those stay in Settings. Trade-off vs langchain-chroma: one less package
# (we already depend on chromadb); we own Document mapping ourselves.

"""Persistent Chroma vector store helpers for RAG retrieval."""

from __future__ import annotations

from typing import Any
from uuid import uuid4

import chromadb
from langchain_core.documents import Document

from agentflow.config import Settings, get_settings
from agentflow.llm import get_embeddings


def get_chroma_client(settings: Settings | None = None) -> chromadb.ClientAPI:
    """Return a PersistentClient rooted at `chroma_persist_dir`."""
    cfg = settings if settings is not None else get_settings()
    return chromadb.PersistentClient(path=cfg.chroma_persist_dir)


def get_collection(
    settings: Settings | None = None,
) -> chromadb.Collection:
    """Get or create the configured document collection."""
    cfg = settings if settings is not None else get_settings()
    client = get_chroma_client(cfg)
    return client.get_or_create_collection(name=cfg.chroma_collection)


def add_documents(
    documents: list[Document],
    *,
    settings: Settings | None = None,
) -> list[str]:
    """Embed and upsert documents; return the ids written.

    Ids come from `Document.id` when set, otherwise a new UUID — stable ids
    matter if you re-ingest the same source without duplicating vectors.
    """
    if not documents:
        return []
    cfg = settings if settings is not None else get_settings()
    collection = get_collection(cfg)
    embeddings = get_embeddings(cfg)

    ids = [doc.id if doc.id else str(uuid4()) for doc in documents]
    texts = [doc.page_content for doc in documents]
    metadatas: list[dict[str, Any]] = [
        dict(doc.metadata) if doc.metadata else {} for doc in documents
    ]
    vectors = embeddings.embed_documents(texts)
    collection.upsert(
        ids=ids,
        documents=texts,
        embeddings=vectors,
        metadatas=metadatas,
    )
    return ids


def similarity_search(
    query: str,
    *,
    k: int | None = None,
    settings: Settings | None = None,
) -> list[Document]:
    """Return the top-k Documents nearest to `query`.

    v1 callers can map `page_content` into `AgentState.context: list[str]`.
    Week 2 can pass these Documents through state for citation metadata.
    """
    cfg = settings if settings is not None else get_settings()
    top_k = k if k is not None else cfg.retrieve_top_k
    collection = get_collection(cfg)
    embeddings = get_embeddings(cfg)
    query_vec = embeddings.embed_query(query)

    result = collection.query(
        query_embeddings=[query_vec],
        n_results=top_k,
        include=["documents", "metadatas"],
    )
    docs_batch = result.get("documents") or [[]]
    meta_batch = result.get("metadatas") or [[]]
    id_batch = result.get("ids") or [[]]

    documents: list[Document] = []
    for text, metadata, doc_id in zip(
        docs_batch[0], meta_batch[0], id_batch[0], strict=False
    ):
        if text is None:
            continue
        documents.append(
            Document(
                page_content=text,
                metadata=dict(metadata) if metadata else {},
                id=doc_id,
            )
        )
    return documents


__all__ = [
    "add_documents",
    "get_chroma_client",
    "get_collection",
    "similarity_search",
]
