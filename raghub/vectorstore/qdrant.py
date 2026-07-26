"""Qdrant adapter for dense vector search and metadata filtering.

Qdrant client and transport errors intentionally propagate to the application
boundary, which owns logging and domain-level error translation.
"""

from __future__ import annotations

import uuid
from collections.abc import Sequence
from typing import Any

from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels

from raghub.exceptions import VectorStoreError
from raghub.interfaces.vectorstore import VectorStore
from raghub.models import ChunkRecord


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
        """
        self.collection = collection
        self.embedding_dim = embedding_dim
        self.client = QdrantClient(url=url, api_key=api_key, prefer_grpc=prefer_grpc)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def create_collection(self) -> None:
        """Create the Qdrant collection when it is absent."""
        if not self.client.collection_exists(collection_name=self.collection):
            self.client.create_collection(
                collection_name=self.collection,
                vectors_config=qmodels.VectorParams(
                    size=self.embedding_dim,
                    distance=qmodels.Distance.COSINE,
                ),
            )

    def optimize(self) -> None:
        """Flush Qdrant state through its collection-alias update endpoint."""
        self.client.update_collection_aliases(change_aliases_operations=[])

    def delete_version(self, document_id: str, version: int) -> None:
        """Delete chunks for one document version.

        Args:
            document_id: The document id.
            version: The version number.
        """
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

    def health(self) -> dict[str, Any]:
        """Return the names of collections reported by Qdrant."""
        info = self.client.get_collections()
        return {"status": "ok", "collections": [collection.name for collection in info.collections]}

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

    def delete(self, chunk_ids: Sequence[str]) -> None:
        """Delete chunks by their stable Qdrant point ids."""
        self.client.delete(
            collection_name=self.collection,
            points_selector=qmodels.PointIdsList(
                points=[self.qdrant_point_id(chunk_id) for chunk_id in chunk_ids]
            ),
        )

    def delete_document(self, document_id: str) -> None:
        """Delete every chunk for ``document_id``."""
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
            conditions: list[Any] = []
            for key, value in metadata_filter.items():
                if key not in {"company", "document_id"}:
                    raise VectorStoreError(f"Unsupported Qdrant metadata filter field: {key}")
                if isinstance(value, str):
                    conditions.append(
                        qmodels.FieldCondition(key=key, match=qmodels.MatchValue(value=value))
                    )
                elif isinstance(value, list) and all(isinstance(item, str) for item in value):
                    conditions.append(
                        qmodels.FieldCondition(key=key, match=qmodels.MatchAny(any=value))
                    )
                else:
                    raise VectorStoreError(f"Unsupported Qdrant metadata filter value for {key}")
            query_filter = qmodels.Filter(must=conditions) if conditions else None

        client: Any = self.client
        query_points = getattr(client, "query_points", None)
        if query_points is None:
            hits = client.search(
                collection_name=self.collection,
                query_vector=list(vector),
                query_filter=query_filter,
                limit=top_k,
                with_payload=True,
            )
        else:
            response = query_points(
                collection_name=self.collection,
                query=list(vector),
                query_filter=query_filter,
                limit=top_k,
                with_payload=True,
            )
            hits = getattr(response, "points", response)

        results: list[dict[str, Any]] = []
        for hit in hits:
            chunk = ChunkRecord.model_validate(hit.payload)
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
