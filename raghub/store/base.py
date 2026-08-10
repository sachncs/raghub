"""Vector-store contracts and metadata helpers.

This module defines the abstract :class:`Store` contract, the
:class:`MemoryVectorRecord` payload, and the ``matches_metadata_*``
helpers used to apply metadata pre-filters to in-memory records.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from raghub.models import Chunk


def matches_metadata_dict(record: MemoryVectorRecord, filters: dict[str, Any]) -> bool:
    """Return whether ``record.chunk`` satisfies every entry in ``filters``.

    When ``expected`` is a list, the chunk's field must equal one of
    the list's elements (semantics used by RBAC filters where
    ``{"company": ["Apple", "Microsoft"]}`` means "any of these
    companies is acceptable").
    """
    chunk = record.chunk
    for key, expected in filters.items():
        if not hasattr(chunk, key):
            return False
        actual = getattr(chunk, key)
        if isinstance(expected, list):
            if actual not in expected:
                return False
        elif actual != expected:
            return False
    return True


def matches_metadata_string(record: MemoryVectorRecord, filter_string: str) -> bool:
    """Apply the legacy SQL-style filter string against ``record.chunk``."""
    if not filter_string:
        return True
    for clause in filter_string.split(" AND "):
        clause = clause.strip()
        if not clause:
            continue
        if " IN " in clause:
            prefix, _, values_part = clause.partition(" IN ")
            prefix = prefix.strip()
            values_part = values_part.strip().rstrip(")").lstrip("(")
            valid_values = {v.strip().strip("'") for v in values_part.split(",") if v.strip()}
            attr_name = prefix.split(".")[-1]
            if not hasattr(record.chunk, attr_name):
                return False
            if getattr(record.chunk, attr_name) not in valid_values:
                return False
        elif "=" in clause:
            key, _, value = clause.partition("=")
            attr_name = key.strip().split(".")[-1]
            expected = value.strip().strip("'")
            if not hasattr(record.chunk, attr_name):
                return False
            if str(getattr(record.chunk, attr_name)) != expected:
                return False
    return True


@dataclass
class MemoryVectorRecord:
    """A single chunk + its precomputed embedding vector."""

    chunk: Chunk
    vector: list[float]


class Store(ABC):
    """Abstract base for every vector-store adapter."""

    @abstractmethod
    def create_collection(self) -> None:
        """Create the underlying collection when missing."""

    @abstractmethod
    def insert(self, chunks: Sequence[Chunk], vectors: Sequence[list[float]]) -> int:
        """Insert chunks paired with their precomputed embedding vectors.

        Returns:
            The number of rows written. Should equal ``len(chunks)``
            on success; backends that dedup by primary key may return
            a smaller value when the same ``chunk_id`` appears twice.

        """

    @abstractmethod
    def upsert(self, chunks: Sequence[Chunk], vectors: Sequence[list[float]]) -> int:
        """Insert-or-update chunks by primary key.

        Returns:
            The number of rows written (inserts + updates).

        """

    @abstractmethod
    def delete(self, chunk_ids: Sequence[str]) -> None:
        """Delete chunks by ``chunk_id``."""

    @abstractmethod
    def delete_document(self, document_id: str) -> None:
        """Delete every chunk that belongs to ``document_id``."""

    @abstractmethod
    def delete_version(self, document_id: str, version: int) -> None:
        """Delete every chunk that belongs to one ``(document_id, version)`` pair."""

    @abstractmethod
    def search(
        self, *, vector: Sequence[float], top_k: int, metadata_filter: str | dict[str, Any] = ""
    ) -> list[dict[str, Any]]:
        """Return the ``top_k`` hits closest to ``vector`` after pre-filtering."""

    @abstractmethod
    def hybrid_search(
        self,
        *,
        query: str,
        vector: Sequence[float],
        top_k: int,
        metadata_filter: str | dict[str, Any] = "",
    ) -> list[dict[str, Any]]:
        """Combine vector search with BM25 keyword search."""

    @abstractmethod
    def optimize(self) -> None:
        """Persist any in-memory state to durable storage."""

    @abstractmethod
    def keyword_search(self, query: str, top_k: int) -> list[dict[str, Any]]:
        """Run BM25 keyword search alone."""

    @abstractmethod
    def health(self) -> dict[str, Any]:
        """Return a liveness report for the ``/health`` endpoint."""
