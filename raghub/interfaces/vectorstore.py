"""Vector store contracts.

Structural type for the project's vector database. Concrete
implementations include the in-memory and zvec backends in
:mod:`raghub.vectorstore`.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Protocol

from raghub.models import ChunkRecord


class VectorStore(Protocol):
    """Vector database contract."""

    def create_collection(self) -> None:
        """Create or open the backing collection."""

    def insert(self, chunks: Sequence[ChunkRecord], vectors: Sequence[list[float]]) -> None:
        """Insert records.

        Args:
            chunks: Chunk metadata. ``chunks[i]`` corresponds to
                ``vectors[i]``.
            vectors: Parallel list of embedding vectors.
        """

    def upsert(self, chunks: Sequence[ChunkRecord], vectors: Sequence[list[float]]) -> None:
        """Insert or update records.

        Args:
            chunks: Chunk metadata. ``chunks[i]`` corresponds to
                ``vectors[i]``.
            vectors: Parallel list of embedding vectors.
        """

    def delete(self, chunk_ids: Sequence[str]) -> None:
        """Delete chunks by id.

        Args:
            chunk_ids: Chunk ids to remove.
        """

    def delete_document(self, document_id: str) -> None:
        """Delete all chunks for a document.

        Args:
            document_id: The document id.
        """

    def delete_version(self, document_id: str, version: int) -> None:
        """Delete a specific version of a document.

        Args:
            document_id: The document id.
            version: The version number.
        """

    def search(
        self, *, vector: Sequence[float], top_k: int, metadata_filter: str | dict = ""
    ) -> list[dict[str, Any]]:
        """Run filtered vector search.

        Args:
            vector: Query embedding.
            top_k: Maximum number of results.
            metadata_filter: Backend-specific metadata filter
                expression (e.g. a SQL fragment for the in-memory
                backend, or a dict for the RBAC layer).

        Returns:
            A list of hit dicts in backend-native shape.
        """

    def hybrid_search(
        self,
        *,
        query: str,
        vector: Sequence[float],
        top_k: int,
        metadata_filter: str | dict = "",
    ) -> list[dict[str, Any]]:
        """Run hybrid (vector + keyword) search.

        Args:
            query: Raw query string used for the keyword channel.
            vector: Query embedding for the vector channel.
            top_k: Maximum number of results.
            metadata_filter: Backend-specific metadata filter
                expression.

        Returns:
            A list of hit dicts in backend-native shape.
        """

    def optimize(self) -> None:
        """Optimize the index (e.g. flush + compact)."""

    def health(self) -> dict[str, Any]:
        """Return backend health.

        Returns:
            A status dict; ``{"status": "ok"}`` on success.
        """

    def keyword_search(self, query: str, top_k: int) -> list[dict[str, Any]]:
        """Run keyword-only search.

        Args:
            query: The query string.
            top_k: Maximum number of results.

        Returns:
            A list of hit dicts in backend-native shape.
        """
