"""Tests for the abstract ``BaseVectorStore`` defaults."""

from __future__ import annotations

import pytest

from raghub.vectorstore.base import BaseVectorStore


class _StubStore(BaseVectorStore):
    """Minimal concrete subclass that satisfies the abstract interface."""

    def create_collection(self) -> None:
        return None

    def insert(self, chunks: object, vectors: object) -> None:
        return None

    def upsert(self, chunks: object, vectors: object) -> None:
        return None

    def delete(self, chunk_ids: object) -> None:
        return None

    def delete_document(self, document_id: str) -> None:
        return None

    def delete_version(self, document_id: str, version: int) -> None:
        return None

    def search(
        self,
        *,
        vector: object,
        top_k: int,
        metadata_filter: object = "",
    ) -> list[dict[str, object]]:
        return []

    def hybrid_search(
        self,
        *,
        query: str,
        vector: object,
        top_k: int,
        metadata_filter: object = "",
    ) -> list[dict[str, object]]:
        return []

    def optimize(self) -> None:
        return None

    def health(self) -> dict[str, object]:
        return {"status": "ok"}


def test_keyword_search_default_returns_empty_list() -> None:
    """The default keyword_search returns ``[]`` without any backend logic."""
    store = _StubStore()
    assert store.keyword_search("anything", top_k=10) == []


def test_base_vector_store_cannot_be_instantiated_directly() -> None:
    """``BaseVectorStore`` is abstract and rejects direct instantiation."""
    with pytest.raises(TypeError):
        BaseVectorStore()  # type: ignore[abstract]
