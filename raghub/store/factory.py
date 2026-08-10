"""Default vector-store factory."""

from __future__ import annotations

import os

from raghub.config import Settings
from raghub.constants import ENV_RAG_VECTORSTORE_PATH
from raghub.store.base import Store
from raghub.store.sqlite import SqliteStore


def build_store(
    settings: Settings,
    *,
    embedding_dim: int | None = None,
) -> Store:
    """Return the configured vector store.

    The factory always returns a :class:`SqliteStore` pointed at
    ``settings.data_dir / "vectorstore.sqlite"``. The path can be
    overridden via the ``RAG_VECTORSTORE_PATH`` env var. When the
    ``sqlite-vector`` package is installed the store benefits from
    its native ANN + filtering; without it the same SQL tables back
    a NumPy cosine fallback.
    """
    if not isinstance(settings, Settings):
        raise TypeError(f"build_store: expected Settings, got {type(settings).__name__}")
    dim = embedding_dim if embedding_dim is not None else settings.embedding_dim
    override = os.environ.get(ENV_RAG_VECTORSTORE_PATH)
    path = override or str(settings.data_dir / "vectorstore.sqlite")
    return SqliteStore(path=path, embedding_dim=dim)
