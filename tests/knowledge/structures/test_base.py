"""Tests for ``raghub.knowledge.structures.base.KnowledgeIndex``."""
from __future__ import annotations

from typing import Any

import pytest

from raghub.knowledge.structures.base import KnowledgeIndex
from raghub.models import Chunk, RetrievalHit


class _FakeIndex(KnowledgeIndex):
    """Minimal subclass that records every call for assertion."""

    name = "fake"

    def __init__(self) -> None:
        self.chunks: list[Chunk] = []

    def add_chunks(
        self,
        chunks: list[Chunk],
        vectors: list[list[float]],
    ) -> None:
        self.chunks.extend(chunks)

    def delete_for_document(self, document_id: str) -> int:
        before = len(self.chunks)
        self.chunks = [
            chunk for chunk in self.chunks if chunk.document_id != document_id
        ]
        return before - len(self.chunks)

    def search(self, query: str, top_k: int = 5) -> list[RetrievalHit]:
        return self.chunks[:top_k]


def _chunk(document_id: str = "d1", text: str = "body") -> Chunk:
    return Chunk(
        chunk_id="c1",
        document_id=document_id,
        version=1,
        text=text,
        company="acme",
        owner="me",
    )


def test_knowledge_index_is_abstract() -> None:
    """``KnowledgeIndex`` cannot be instantiated directly."""
    with pytest.raises(TypeError):
        KnowledgeIndex()  # type: ignore[abstract]


def test_default_name_is_knowledge_index() -> None:
    """The default class-level ``name`` is ``"knowledge_index"``."""
    assert KnowledgeIndex.name == "knowledge_index"


def test_subclass_can_override_name() -> None:
    """Subclasses may override the ``name`` class attribute."""
    assert _FakeIndex.name == "fake"


def test_subclass_must_implement_add_chunks() -> None:
    """Subclasses that omit ``add_chunks`` remain abstract."""
    with pytest.raises(TypeError):

        class _Missing(KnowledgeIndex):
            def delete_for_document(self, document_id: str) -> int:
                return 0

            def search(self, query: str, top_k: int = 5) -> list[RetrievalHit]:
                return []

        _Missing()  # type: ignore[abstract]


def test_subclass_must_implement_delete_for_document() -> None:
    """Subclasses that omit ``delete_for_document`` remain abstract."""
    with pytest.raises(TypeError):

        class _Missing(KnowledgeIndex):
            def add_chunks(
                self,
                chunks: list[Chunk],
                vectors: list[list[float]],
            ) -> None:
                pass

            def search(self, query: str, top_k: int = 5) -> list[RetrievalHit]:
                return []

        _Missing()  # type: ignore[abstract]


def test_subclass_must_implement_search() -> None:
    """Subclasses that omit ``search`` remain abstract."""
    with pytest.raises(TypeError):

        class _Missing(KnowledgeIndex):
            def add_chunks(
                self,
                chunks: list[Chunk],
                vectors: list[list[float]],
            ) -> None:
                pass

            def delete_for_document(self, document_id: str) -> int:
                return 0

        _Missing()  # type: ignore[abstract]


def test_add_chunks_records_inputs() -> None:
    """``add_chunks`` receives the supplied chunks and stores them."""
    index = _FakeIndex()
    index.add_chunks([_chunk()], [[1.0, 0.0]])
    assert len(index.chunks) == 1


def test_delete_for_document_returns_count() -> None:
    """``delete_for_document`` returns the number of removed entries."""
    index = _FakeIndex()
    index.add_chunks(
        [
            _chunk(document_id="A"),
            _chunk(document_id="A"),
            _chunk(document_id="B"),
        ],
        [[1.0], [1.0], [1.0]],
    )
    assert index.delete_for_document("A") == 2
    assert len(index.chunks) == 1


def test_search_returns_top_k_results() -> None:
    """``search`` respects the ``top_k`` argument."""
    index = _FakeIndex()
    index.add_chunks(
        [_chunk(text=f"chunk {i}") for i in range(5)],
        [[1.0]] * 5,
    )
    assert len(index.search("query", top_k=2)) == 2


def test_health_reports_chunk_count() -> None:
    """The default ``health`` reports the index name and chunk count."""
    index = _FakeIndex()
    index.add_chunks([_chunk(), _chunk()], [[1.0], [1.0]])
    health = index.health()
    assert health == {"name": "fake", "chunks": 2}


def test_health_handles_empty_index() -> None:
    """An empty index reports ``chunks: 0``."""
    index = _FakeIndex()
    assert index.health() == {"name": "fake", "chunks": 0}


def test_health_uses_class_name_default() -> None:
    """A subclass that does not override ``name`` uses the class default."""

    class _DefaultNamed(KnowledgeIndex):
        def add_chunks(
            self,
            chunks: list[Chunk],
            vectors: list[list[float]],
        ) -> None:
            self.chunks = list(chunks)

        def delete_for_document(self, document_id: str) -> int:
            return 0

        def search(self, query: str, top_k: int = 5) -> list[RetrievalHit]:
            return []

    idx = _DefaultNamed()
    assert idx.health()["name"] == "knowledge_index"


def test_health_reports_zero_when_no_chunks_attribute() -> None:
    """Subclasses without a ``chunks`` attribute default to ``0``."""

    class _NoAttr(KnowledgeIndex):
        def add_chunks(
            self,
            chunks: list[Chunk],
            vectors: list[list[float]],
        ) -> None:
            pass

        def delete_for_document(self, document_id: str) -> int:
            return 0

        def search(self, query: str, top_k: int = 5) -> list[RetrievalHit]:
            return []

    idx = _NoAttr()
    assert idx.health() == {"name": "knowledge_index", "chunks": 0}


def test_subclass_can_override_health() -> None:
    """Subclasses may override ``health`` to report custom data."""

    class _Healthful(KnowledgeIndex):
        def add_chunks(
            self,
            chunks: list[Chunk],
            vectors: list[list[float]],
        ) -> None:
            pass

        def delete_for_document(self, document_id: str) -> int:
            return 0

        def search(self, query: str, top_k: int = 5) -> list[RetrievalHit]:
            return []

        def health(self) -> dict[str, Any]:
            return {"name": "custom", "version": 2}

    idx = _Healthful()
    assert idx.health() == {"name": "custom", "version": 2}