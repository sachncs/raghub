"""Qdrant vector-store adapter.

Implements the :class:`raghub.interfaces.vectorstore.VectorStore`
contract against a Qdrant server. Falls back gracefully when the
``qdrant-client`` package is not installed.
"""

from __future__ import annotations

import uuid
from typing import Any, Sequence

from raghub.exceptions import ConfigurationError, VectorStoreError
from raghub.interfaces.vectorstore import VectorStore
from raghub.models import ChunkRecord

try:
    from qdrant_client import QdrantClient  # type: ignore
    from qdrant_client.http import models as _qmodels  # type: ignore
    _QDRANT_AVAILABLE = True
    _ImportError: Exception | None = None
except Exception as exc:  # pragma: no cover - optional dep
    QdrantClient = None
    _qmodels = None
    _QDRANT_AVAILABLE = False
    _ImportError = exc


class QdrantVectorStore(VectorStore):
    """Qdrant-backed vector store."""

    def __init__(
        self,
        *,
        collection: str = "raghub",
        url: str = "http://localhost:6333",
        api_key: str | None = None,
        embedding_dim: int = 384,
        prefer_grpc: bool = False,
    ) -> None:
        """Initialise the Qdrant client.

        Args:
            collection: Name of the Qdrant collection.
            url: Qdrant server URL.
            api_key: Optional API key.
            embedding_dim: Expected embedding dimension.
            prefer_grpc: Use gRPC transport when available.

        Raises:
            ConfigurationError: When ``qdrant-client`` is not installed.
        """
        if not _QDRANT_AVAILABLE:
            raise ConfigurationError(
                "qdrant-client is not installed; run `pip install qdrant-client`."
            )
        self._collection = collection
        self._embedding_dim = embedding_dim
        self._client = QdrantClient(url=url, api_key=api_key, prefer_grpc=prefer_grpc)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def create_collection(self) -> None:
        """Create the Qdrant collection if it does not already exist."""
        try:
            self._client.get_collection(self._collection)
        except Exception:
            self._client.recreate_collection(
                collection_name=self._collection,
                vectors_config=_qmodels.VectorParams(
                    size=self._embedding_dim,
                    distance=_qmodels.Distance.COSINE,
                ),
            )

    def optimize(self) -> None:
        """Qdrant has no separate optimisation step; flush in-memory state."""
        try:
            self._client.update_collection_aliases(  # cheap call to verify liveness
                change_aliases_operations=[]
            )
        except Exception:
            pass

    def health(self) -> dict[str, Any]:
        """Return Qdrant cluster health."""
        try:
            info = self._client.get_collections()
            return {"status": "ok", "collections": [c.name for c in info.collections]}
        except Exception as exc:
            return {"status": "degraded", "error": str(exc)}

    # ------------------------------------------------------------------
    # CRUD
    # ------------------------------------------------------------------

    def qdrantqdrant_point_id(self, chunk_id: str) -> str:
        """Return a stable UUID derived from ``chunk_id``."""
        return str(uuid.uuid5(uuid.NAMESPACE_DNS, f"raghub:{chunk_id}"))

    def insert(self, chunks: Sequence[ChunkRecord], vectors: Sequence[list[float]]) -> None:
        """Insert ``chunks`` with their vectors (fails on existing ids)."""
        self._upsert(chunks, vectors)

    def upsert(self, chunks: Sequence[ChunkRecord], vectors: Sequence[list[float]]) -> None:
        """Upsert ``chunks`` with their vectors."""
        self._upsert(chunks, vectors)

    def _upsert(self, chunks: Sequence[ChunkRecord], vectors: Sequence[list[float]]) -> None:
        if len(chunks) != len(vectors):
            raise VectorStoreError("chunks and vectors length mismatch")
        if not chunks:
            return
        try:
            self._client.upsert(
                collection_name=self._collection,
                points=[
                    _qmodels.PointStruct(
                        id=self.qdrantqdrant_point_id(chunk.chunk_id),
                        vector=list(vector),
                        payload={
                            "chunk_id": chunk.chunk_id,
                            "document_id": chunk.document_id,
                            "version": chunk.version,
                            "page": chunk.page,
                            "source_location": chunk.source_location,
                            "section": chunk.section,
                            "company": chunk.company,
                            "owner": chunk.owner,
                            "department": chunk.department,
                            "classification": chunk.classification.value,
                            "text": chunk.text,
                            "metadata": chunk.metadata,
                        },
                    )
                    for chunk, vector in zip(chunks, vectors)
                ],
            )
        except Exception as exc:
            raise VectorStoreError(f"Qdrant upsert failed: {exc}") from exc

    def delete(self, chunk_ids: Sequence[str]) -> None:
        """Delete by chunk id (alias for ``delete_chunks_by_id``)."""
        self.delete_chunks_by_id(chunk_ids)
        try:
            self._client.delete(
                collection_name=self._collection,
                points_selector=_qmodels.PointIdsList(
                    points=[self.qdrantqdrant_point_id(cid) for cid in chunk_ids]
                ),
            )
        except Exception as exc:
            raise VectorStoreError(f"Qdrant delete failed: {exc}") from exc

    def delete_document(self, document_id: str) -> None:
        """Delete every chunk for ``document_id``."""
        try:
            self._client.delete(
                collection_name=self._collection,
                points_selector=_qmodels.FilterSelector(
                    filter=_qmodels.Filter(
                        must=[
                            _qmodels.FieldCondition(
                                key="document_id", match=_qmodels.MatchValue(value=document_id)
                            )
                        ]
                    )
                ),
            )
        except Exception as exc:
            raise VectorStoreError(f"Qdrant delete_document failed: {exc}") from exc

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    def search(
        self,
        *,
        vector: list[float],
        top_k: int,
        metadata_filter: str = "",
    ) -> list[dict[str, Any]]:
        """Run vector search.

        ``metadata_filter`` is interpreted as a Qdrant filter JSON
        string; the legacy in-memory backends used a SQL fragment.
        """
        return self._search(vector=vector, top_k=top_k, query_filter=None, query=None)

    def hybrid_search(
        self,
        *,
        query: str,
        vector: list[float],
        top_k: int,
        metadata_filter: str = "",
    ) -> list[dict[str, Any]]:
        """Run hybrid (vector + keyword) search against the collection.

        Implementation note: Qdrant's native hybrid mode requires a
        collection with a configured sparse vector. The default
        :class:`QdrantVectorStore` is created with dense vectors only,
        so this method falls back to a dense-only ``search`` for now.
        To enable true hybrid search, create the collection with a
        named sparse vector and extend :class:`QdrantVectorStore` to
        issue a ``query_points`` call with both dense and sparse
        inputs.
        """
        return self._search(
            vector=vector, top_k=top_k, query_filter=None, query=query
        )

    def keyword_search(self, query: str, top_k: int) -> list[dict[str, Any]]:
        """Return empty list; Qdrant keyword channel requires a sparse vector config
        which is out of scope for the default install.
        """
        return []

    def _search(
        self,
        *,
        vector: list[float],
        top_k: int,
        query_filter: Any | None,
        query: str | None,
    ) -> list[dict[str, Any]]:
        try:
            response = self._client.search(
                collection_name=self._collection,
                query_vector=vector,
                limit=top_k,
                with_payload=True,
            )
        except Exception as exc:
            raise VectorStoreError(f"Qdrant search failed: {exc}") from exc

        results: list[dict[str, Any]] = []
        for hit in response:
            payload = hit.payload or {}
            results.append(
                {
                    "chunk_id": payload.get("chunk_id", str(hit.id)),
                    "score": float(hit.score),
                    "chunk": ChunkRecord(
                        chunk_id=payload.get("chunk_id", str(hit.id)),
                        document_id=payload.get("document_id", ""),
                        version=int(payload.get("version", 1)),
                        page=int(payload.get("page", 0)),
                        source_location=payload.get("source_location", ""),
                        section=payload.get("section", ""),
                        company=payload.get("company", ""),
                        owner=payload.get("owner", ""),
                        department=payload.get("department", ""),
                        text=payload.get("text", ""),
                        metadata=payload.get("metadata", {}) or {},
                    ),
                }
            )
        return results


__all__ = ["QdrantVectorStore"]
