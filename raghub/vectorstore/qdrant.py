"""Qdrant vector-store adapter.

Implements the :class:`raghub.interfaces.vectorstore.VectorStore`
contract against a Qdrant server. Falls back gracefully when the
``qdrant-client`` package is not installed.
"""

from __future__ import annotations

import uuid
from collections.abc import Sequence
from contextlib import suppress
from typing import Any

from raghub.exceptions import ConfigurationError, VectorStoreError
from raghub.interfaces.vectorstore import VectorStore
from raghub.models import ChunkRecord

QdrantClient: Any
qmodels: Any

try:
    from qdrant_client import QdrantClient
    from qdrant_client.http import models as qmodels

    QDRANT_AVAILABLE = True
    OptionalImportError: Exception | None = None
except Exception as exc:  # pragma: no cover - optional dep
    QdrantClient = None
    qmodels = None
    QDRANT_AVAILABLE = False
    OptionalImportError = exc


class QdrantVectorStore(VectorStore):
    """Qdrant-backed vector store.

    Transport choice — gRPC vs HTTP:
        The client defaults to HTTP (``prefer_grpc=False``) for broad
        compatibility. HTTP is simpler to debug, works over any
        network (including load balancers that don't support gRPC),
        and avoids the dependency on ``grpcio``. The trade-off is
        latency: gRPC uses persistent connections and protobuf
        serialisation (roughly 10-30% faster for high-throughput
        workloads). When latency matters, switch Qdrant to its gRPC
        port (usually 6334) and set ``prefer_grpc=True``.
    """

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
            prefer_grpc: Use gRPC transport when available (see class
                docstring for trade-offs).

        Raises:
            ConfigurationError: When ``qdrant-client`` is not installed.
        """
        if not QDRANT_AVAILABLE:
            raise ConfigurationError(
                "qdrant-client is not installed; run `pip install qdrant-client`."
            )
        self.collection = collection
        self.embedding_dim = embedding_dim
        self.client = QdrantClient(url=url, api_key=api_key, prefer_grpc=prefer_grpc)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def create_collection(self) -> None:
        """Create the Qdrant collection if it does not already exist."""
        try:
            self.client.get_collection(self.collection)
            return
        except Exception as exc:
            if getattr(exc, "status_code", None) != 404:
                raise VectorStoreError(f"Qdrant collection lookup failed: {exc}") from exc
        try:
            self.client.create_collection(
                collection_name=self.collection,
                vectors_config=qmodels.VectorParams(
                    size=self.embedding_dim,
                    distance=qmodels.Distance.COSINE,
                ),
            )
        except Exception as exc:
            raise VectorStoreError(f"Qdrant collection creation failed: {exc}") from exc

    def optimize(self) -> None:
        """Qdrant has no separate optimisation step; flush in-memory state."""
        with suppress(Exception):
            self.client.update_collection_aliases(change_aliases_operations=[])

    def delete_version(self, document_id: str, version: int) -> None:
        """Delete a specific version of a document.

        Args:
            document_id: The document id.
            version: The version number.
        """
        try:
            self.client.delete(
                collection_name=self.collection,
                points_selector=qmodels.FilterSelector(
                    filter=qmodels.Filter(
                        must=[
                            qmodels.FieldCondition(
                                key="document_id", match=qmodels.MatchValue(value=document_id)
                            ),
                            qmodels.FieldCondition(
                                key="version", match=qmodels.MatchValue(value=version)
                            ),
                        ]
                    )
                ),
            )
        except Exception as exc:
            raise VectorStoreError(f"Qdrant delete_version failed: {exc}") from exc

    def health(self) -> dict[str, Any]:
        """Return Qdrant cluster health."""
        try:
            info = self.client.get_collections()
            return {"status": "ok", "collections": [c.name for c in info.collections]}
        except Exception as exc:
            return {"status": "degraded", "error": str(exc)}

    # ------------------------------------------------------------------
    # CRUD
    # ------------------------------------------------------------------

    def qdrant_point_id(self, chunk_id: str) -> str:
        """Return a stable UUID derived from ``chunk_id``."""
        return str(uuid.uuid5(uuid.NAMESPACE_DNS, f"raghub:{chunk_id}"))

    def insert(self, chunks: Sequence[ChunkRecord], vectors: Sequence[list[float]]) -> None:
        """Insert ``chunks`` with their vectors (fails on existing ids)."""
        self.upsert(chunks, vectors)

    def upsert(self, chunks: Sequence[ChunkRecord], vectors: Sequence[list[float]]) -> None:
        """Upsert ``chunks`` with their vectors."""
        if len(chunks) != len(vectors):
            raise VectorStoreError("chunks and vectors length mismatch")
        if not chunks:
            return
        try:
            self.client.upsert(
                collection_name=self.collection,
                points=[
                    qmodels.PointStruct(
                        id=self.qdrant_point_id(chunk.chunk_id),
                        vector=list(vector),
                        payload=chunk.model_dump(mode="json"),
                    )
                    for chunk, vector in zip(chunks, vectors, strict=True)
                ],
            )
        except Exception as exc:
            raise VectorStoreError(f"Qdrant upsert failed: {exc}") from exc

    def delete(self, chunk_ids: Sequence[str]) -> None:
        """Delete by chunk id."""
        try:
            self.client.delete(
                collection_name=self.collection,
                points_selector=qmodels.PointIdsList(
                    points=[self.qdrant_point_id(cid) for cid in chunk_ids]
                ),
            )
        except Exception as exc:
            raise VectorStoreError(f"Qdrant delete failed: {exc}") from exc

    def delete_document(self, document_id: str) -> None:
        """Delete every chunk for ``document_id``."""
        try:
            self.client.delete(
                collection_name=self.collection,
                points_selector=qmodels.FilterSelector(
                    filter=qmodels.Filter(
                        must=[
                            qmodels.FieldCondition(
                                key="document_id", match=qmodels.MatchValue(value=document_id)
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
        vector: Sequence[float],
        top_k: int,
        metadata_filter: str | dict = "",
    ) -> list[dict[str, Any]]:
        """Run vector search with a canonical metadata filter."""
        if metadata_filter == "":
            query_filter = None
        elif not isinstance(metadata_filter, dict):
            raise VectorStoreError("Qdrant metadata_filter must be a dict or empty string")
        else:
            conditions = []
            for key, value in metadata_filter.items():
                if key not in {"company", "document_id"}:
                    raise VectorStoreError(f"Unsupported Qdrant metadata filter field: {key}")
                if isinstance(value, str):
                    match = qmodels.MatchValue(value=value)
                elif isinstance(value, list) and all(isinstance(item, str) for item in value):
                    match = qmodels.MatchAny(any=value)
                else:
                    raise VectorStoreError(f"Unsupported Qdrant metadata filter value for {key}")
                conditions.append(qmodels.FieldCondition(key=key, match=match))
            query_filter = qmodels.Filter(must=conditions) if conditions else None

        try:
            query_points = getattr(self.client, "query_points", None)
            if query_points is None:
                response = self.client.search(
                    collection_name=self.collection,
                    query_vector=list(vector),
                    query_filter=query_filter,
                    limit=top_k,
                    with_payload=True,
                )
                hits = response
            else:
                response = query_points(
                    collection_name=self.collection,
                    query=list(vector),
                    query_filter=query_filter,
                    limit=top_k,
                    with_payload=True,
                )
                hits = getattr(response, "points", response)
        except Exception as exc:
            raise VectorStoreError(f"Qdrant search failed: {exc}") from exc

        results: list[dict[str, Any]] = []
        for hit in hits:
            try:
                chunk = ChunkRecord.model_validate(hit.payload)
            except Exception as exc:
                raise VectorStoreError(
                    f"Qdrant search returned invalid payload for point {hit.id}"
                ) from exc
            results.append(
                {
                    "chunk_id": chunk.chunk_id,
                    "score": float(hit.score),
                    "chunk": chunk,
                }
            )
        return results

    def hybrid_search(
        self,
        *,
        query: str,
        vector: Sequence[float],
        top_k: int,
        metadata_filter: str | dict = "",
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
        return self.search(
            vector=vector,
            top_k=top_k,
            metadata_filter=metadata_filter,
        )

    def keyword_search(self, query: str, top_k: int) -> list[dict[str, Any]]:
        """Return empty list; Qdrant keyword channel requires a sparse vector config
        which is out of scope for the default install.
        """
        return []


__all__ = ["QdrantVectorStore"]
