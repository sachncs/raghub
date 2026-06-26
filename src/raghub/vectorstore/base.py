"""Vector store base and utility types."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import Any

from raghub.models import ChunkRecord


class BaseVectorStore(ABC):
    """Abstract vector database."""

    @abstractmethod
    def create_collection(self) -> None:
        """Create or open the collection."""

    @abstractmethod
    def insert(self, chunks: Sequence[ChunkRecord], vectors: Sequence[list[float]]) -> None:
        """Insert chunks."""

    @abstractmethod
    def upsert(self, chunks: Sequence[ChunkRecord], vectors: Sequence[list[float]]) -> None:
        """Upsert chunks."""

    @abstractmethod
    def delete(self, chunk_ids: Sequence[str]) -> None:
        """Delete chunks."""

    @abstractmethod
    def delete_document(self, document_id: str) -> None:
        """Delete all chunks for a document."""

    @abstractmethod
    def delete_version(self, document_id: str, version: int) -> None:
        """Delete one version."""

    @abstractmethod
    def search(self, *, vector: Sequence[float], top_k: int, metadata_filter: str) -> list[dict[str, Any]]:
        """Search by vector with a metadata filter."""

    @abstractmethod
    def hybrid_search(
        self,
        *,
        query: str,
        vector: Sequence[float],
        top_k: int,
        metadata_filter: str,
    ) -> list[dict[str, Any]]:
        """Hybrid search."""

    @abstractmethod
    def optimize(self) -> None:
        """Optimize internal indexes."""

    @abstractmethod
    def health(self) -> dict[str, Any]:
        """Return health information."""
