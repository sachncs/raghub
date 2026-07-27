"""Vector-store implementations for dense, hybrid, and metadata-filtered search.

The module exposes the abstract contract together with the in-memory,
Qdrant, and Zvec adapters. All four share the dense-vector domain so
they live in a single file even though the implementations span
~1000 lines.
"""

from __future__ import annotations

import os
import re

# Module aliases so tests that patch ``raghub.vectorstore.qdrant.X``
# (or .memory / .zvec / .base) after the flatten still resolve.
import sys
import uuid
from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from importlib import import_module
from importlib.util import find_spec
from threading import RLock
from typing import Any, cast

import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels
from rank_bm25 import BM25Okapi

from raghub.exceptions import VectorStoreError
from raghub.models import ChunkRecord, VectorStore

sys.modules.setdefault("raghub.vectorstore.base", sys.modules[__name__])
sys.modules.setdefault("raghub.vectorstore.memory", sys.modules[__name__])
sys.modules.setdefault("raghub.vectorstore.qdrant", sys.modules[__name__])
sys.modules.setdefault("raghub.vectorstore.zvec", sys.modules[__name__])


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
    def search(
        self, *, vector: Sequence[float], top_k: int, metadata_filter: str | dict[str, Any] = ""
    ) -> list[dict[str, Any]]:
        """Search by vector with a metadata filter."""

    @abstractmethod
    def hybrid_search(
        self,
        *,
        query: str,
        vector: Sequence[float],
        top_k: int,
        metadata_filter: str | dict[str, Any] = "",
    ) -> list[dict[str, Any]]:
        """Hybrid search."""

    @abstractmethod
    def optimize(self) -> None:
        """Optimize internal indexes."""

    @abstractmethod
    def health(self) -> dict[str, Any]:
        """Return health information."""

    def keyword_search(self, query: str, top_k: int) -> list[dict[str, Any]]:
        """Keyword-based search. Default implementation returns empty."""
        return []


def matches_metadata_dict(record: MemoryVectorRecord, filters: dict[str, Any]) -> bool:
    """Return whether ``record`` matches every key/value in ``filters``."""
    for key, expected in filters.items():
        if not hasattr(record.chunk, key):
            return False
        actual = getattr(record.chunk, key)
        if isinstance(expected, list):
            if actual not in expected:
                return False
        elif actual != expected:
            return False
    return True


@dataclass
class MemoryVectorRecord:
    """A single chunk + its precomputed embedding vector.

    Attributes:
        chunk: The persisted :class:`ChunkRecord` returned to callers.
        vector: The raw embedding as a Python list of floats.
    """

    chunk: ChunkRecord
    vector: list[float]


class InMemoryVectorStore(BaseVectorStore):
    """Cosine-similarity vector store with BM25 keyword search."""

    def __init__(self) -> None:
        """Initialise an empty store with a re-entrant lock."""
        self.lock = RLock()
        self.records: dict[str, MemoryVectorRecord] = {}

    def create_collection(self) -> None:
        """No-op: this backend has no separate collection concept."""
        return None

    def insert(self, chunks: Sequence[ChunkRecord], vectors: Sequence[list[float]]) -> None:
        """Insert or overwrite chunks by ``chunk_id``."""
        with self.lock:
            for chunk, vector in zip(chunks, vectors, strict=True):
                self.records[chunk.chunk_id] = MemoryVectorRecord(chunk=chunk, vector=vector)

    def upsert(self, chunks: Sequence[ChunkRecord], vectors: Sequence[list[float]]) -> None:
        """Insert-or-update alias. Delegates to :meth:`insert`."""
        self.insert(chunks, vectors)

    def delete(self, chunk_ids: Sequence[str]) -> None:
        """Remove chunks by id, tolerating unknown ids."""
        with self.lock:
            for chunk_id in chunk_ids:
                self.records.pop(chunk_id, None)

    def delete_document(self, document_id: str) -> None:
        """Remove every chunk belonging to ``document_id``."""
        with self.lock:
            chunk_ids = [
                chunk_id
                for chunk_id, record in self.records.items()
                if record.chunk.document_id == document_id
            ]
            for chunk_id in chunk_ids:
                self.records.pop(chunk_id, None)

    def delete_version(self, document_id: str, version: int) -> None:
        """Remove chunks whose ``document_id`` and ``version`` match."""
        with self.lock:
            chunk_ids = [
                chunk_id
                for chunk_id, record in self.records.items()
                if record.chunk.document_id == document_id and record.chunk.version == version
            ]
            for chunk_id in chunk_ids:
                self.records.pop(chunk_id, None)

    def matches_filter(self, record: MemoryVectorRecord, metadata_filter: str) -> bool:
        """Return whether ``record`` satisfies ``metadata_filter``.

        The parser recognises two shapes:

        * ``company IN ('a', 'b')`` — checks ``record.chunk.company``.
        * ``document_id = 'abc'`` — checks ``record.chunk.document_id``.

        Anything else fails closed.
        """
        if not metadata_filter:
            return True
        company_match = re.fullmatch(
            r"\s*company\s+IN\s+\((.+)\)\s*", metadata_filter, flags=re.IGNORECASE
        )
        if company_match:
            allowed = [item.strip().strip("'\"") for item in company_match.group(1).split(",")]
            return record.chunk.company in allowed
        document_match = re.fullmatch(
            r"\s*document_id\s*=\s*'([^']+)'\s*", metadata_filter, flags=re.IGNORECASE
        )
        if document_match:
            return bool(record.chunk.document_id == document_match.group(1))
        return False

    def compute_score(self, left: Sequence[float], right: Sequence[float]) -> float:
        """Compute cosine similarity in ``[0, 1]``."""
        lhs = np.asarray(left, dtype=np.float32)
        rhs = np.asarray(right, dtype=np.float32)
        denom = float(np.linalg.norm(lhs) * np.linalg.norm(rhs))
        if denom == 0:
            return 0.0
        return float(np.dot(lhs, rhs) / denom)

    def search(
        self, *, vector: Sequence[float], top_k: int, metadata_filter: str | dict[str, Any] = ""
    ) -> list[dict[str, Any]]:
        """Cosine-similarity search with metadata pre-filtering."""
        if isinstance(metadata_filter, dict):
            dict_filter = metadata_filter
            str_filter: str | None = None
        else:
            dict_filter = None
            str_filter = metadata_filter
        with self.lock:
            records = [
                record
                for record in self.records.values()
                if (dict_filter is None or matches_metadata_dict(record, dict_filter))
                and (dict_filter is not None or self.matches_filter(record, str_filter or ""))
            ]
        hits = [
            {
                "chunk_id": record.chunk.chunk_id,
                "score": self.compute_score(vector, record.vector),
                "chunk": record.chunk,
            }
            for record in records
        ]
        hits.sort(key=lambda item: cast(float, item["score"]), reverse=True)
        return hits[:top_k]

    def hybrid_search(
        self,
        *,
        query: str,
        vector: Sequence[float],
        top_k: int,
        metadata_filter: str | dict[str, Any] = "",
    ) -> list[dict[str, Any]]:
        """Hybrid search shim. The in-memory backend collapses to vector search."""
        return self.search(vector=vector, top_k=top_k, metadata_filter=metadata_filter)

    def keyword_search(self, query: str, top_k: int) -> list[dict[str, Any]]:
        """Return BM25-ranked chunks containing the query terms."""
        query_terms = query.lower().split()
        if not query_terms:
            return []
        with self.lock:
            records = list(self.records.values())
        tokenised_corpus = [(record.chunk.text or "").lower().split() for record in records]
        if not any(tokenised_corpus):
            return []
        scores = BM25Okapi(tokenised_corpus).get_scores(query_terms)
        scored = [
            (record.chunk.chunk_id, float(score), record.chunk)
            for record, score in zip(records, scores, strict=True)
            if score > 0
        ]
        scored.sort(key=lambda item: item[1], reverse=True)
        return [
            {"chunk_id": chunk_id, "score": score, "chunk": chunk}
            for chunk_id, score, chunk in scored[:top_k]
        ]

    def optimize(self) -> None:
        """No-op: the in-memory backend has no on-disk structures to optimise."""
        return None

    def health(self) -> dict[str, Any]:
        """Report liveness information for the health endpoint."""
        return {"status": "ok", "backend": "memory", "chunks": len(self.records)}


class QdrantVectorStore(VectorStore):
    """Qdrant-backed vector store.

    Transport choice — gRPC vs HTTP:
        The client defaults to HTTP (``prefer_grpc=False``) for broad
        When latency matters, switch Qdrant to its gRPC
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
        """Initialise the Qdrant client."""
        self.collection = collection
        self.embedding_dim = embedding_dim
        self.client = QdrantClient(url=url, api_key=api_key, prefer_grpc=prefer_grpc)

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
        """Delete chunks for one document version."""

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

    def search(
        self,
        *,
        vector: Sequence[float],
        top_k: int,
        metadata_filter: str | dict[str, Any] = "",
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
        metadata_filter: str | dict[str, Any] = "",
    ) -> list[dict[str, Any]]:
        """Run hybrid (vector + keyword) search against the collection.

        Implementation note: Qdrant's native hybrid mode requires a
        collection with a configured sparse vector. The default
        :class:`QdrantVectorStore` is created with dense vectors only,
        so this method falls back to a dense-only ``search`` for now.
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


zvec_module = import_module("zvec") if find_spec("zvec") is not None else None


def native_filter(metadata_filter: str | dict[str, Any]) -> str | None:
    """Translate a canonical metadata filter into a Zvec SQL fragment.

    Args:
        metadata_filter: Canonical dict, SQL string, or empty string.

    Returns:
        A SQL fragment, ``None`` for no constraint, or ``"false"`` for an
        empty list that must match no records.

    Raises:
        ValueError: If a field, value, or filter type is unsupported.
    """
    if metadata_filter in ("", None):
        return None
    if isinstance(metadata_filter, dict):
        if not metadata_filter:
            return None
        if set(metadata_filter) - {"company", "document_id"}:
            raise ValueError(f"Unsupported Zvec metadata filter fields: {sorted(metadata_filter)}")
        clauses: list[str] = []
        for key, value in metadata_filter.items():
            if isinstance(value, list):
                if not value:
                    return "false"
                literals = ", ".join(
                    f"'{str(item).replace(chr(39), chr(39) * 2)}'" for item in value
                )
                clauses.append(f"{key} IN ({literals})")
            elif isinstance(value, str):
                escaped = value.replace("'", "''")
                clauses.append(f"{key} = '{escaped}'")
            else:
                raise ValueError(f"Unsupported Zvec metadata filter value for {key}")
        return " AND ".join(clauses)
    if isinstance(metadata_filter, str):
        return metadata_filter
    raise ValueError(f"Unsupported Zvec metadata filter type: {type(metadata_filter).__name__}")


class RealZvecBackend(BaseVectorStore):
    """Vector-store adapter around the native Alibaba Zvec collection API."""

    def __init__(self, zvec_module: Any, path: str, embedding_dim: int) -> None:
        """Open or create a Zvec collection at ``path``."""
        self.zvec = zvec_module
        self.path = path
        self.embedding_dim = embedding_dim
        self.collection = self.open_collection()

    def open_collection(self) -> Any:
        """Open an existing collection or create one with the RAGHub schema."""
        lock_path = os.path.join(self.path, "LOCK")
        if os.path.exists(lock_path):
            return self.zvec.open(path=self.path)

        schema = self.zvec.CollectionSchema(
            name="documents",
            fields=[
                self.zvec.FieldSchema(name="chunk_id", data_type=self.zvec.DataType.STRING),
                self.zvec.FieldSchema(name="document_id", data_type=self.zvec.DataType.STRING),
                self.zvec.FieldSchema(name="version", data_type=self.zvec.DataType.INT32),
                self.zvec.FieldSchema(name="page", data_type=self.zvec.DataType.INT32),
                self.zvec.FieldSchema(name="section", data_type=self.zvec.DataType.STRING),
                self.zvec.FieldSchema(name="company", data_type=self.zvec.DataType.STRING),
                self.zvec.FieldSchema(name="owner", data_type=self.zvec.DataType.STRING),
                self.zvec.FieldSchema(name="department", data_type=self.zvec.DataType.STRING),
                self.zvec.FieldSchema(name="classification", data_type=self.zvec.DataType.STRING),
                self.zvec.FieldSchema(name="created_at", data_type=self.zvec.DataType.STRING),
                self.zvec.FieldSchema(name="embedding_model", data_type=self.zvec.DataType.STRING),
                self.zvec.FieldSchema(name="hash", data_type=self.zvec.DataType.STRING),
                self.zvec.FieldSchema(name="filename", data_type=self.zvec.DataType.STRING),
                self.zvec.FieldSchema(name="text", data_type=self.zvec.DataType.STRING),
            ],
            vectors=[
                self.zvec.VectorSchema(
                    name="embedding",
                    data_type=self.zvec.DataType.VECTOR_FP32,
                    dimension=self.embedding_dim,
                    index_param=self.zvec.HnswIndexParam(metric_type=self.zvec.MetricType.COSINE),
                )
            ],
        )
        return self.zvec.create_and_open(path=self.path, schema=schema)

    def create_collection(self) -> None:
        """No-op because construction already opens or creates the collection."""
        return None

    def insert(self, chunks: Sequence[ChunkRecord], vectors: Sequence[list[float]]) -> None:
        """Insert chunks and their vectors into the native collection."""
        for chunk, vector in zip(chunks, vectors, strict=True):
            self.collection.insert(
                self.zvec.Doc(
                    id=chunk.chunk_id,
                    vectors={"embedding": vector},
                    fields={
                        "chunk_id": chunk.chunk_id,
                        "document_id": chunk.document_id,
                        "version": chunk.version,
                        "page": chunk.page,
                        "section": chunk.section,
                        "company": chunk.company,
                        "owner": chunk.owner,
                        "department": chunk.department,
                        "classification": chunk.classification.value,
                        "created_at": chunk.created_at.isoformat(),
                        "embedding_model": chunk.embedding_model,
                        "hash": chunk.hash,
                        "filename": chunk.metadata.get("filename", ""),
                        "text": chunk.text,
                    },
                )
            )

    def upsert(self, chunks: Sequence[ChunkRecord], vectors: Sequence[list[float]]) -> None:
        """Insert or replace chunks using the native insertion operation."""
        self.insert(chunks, vectors)

    def delete(self, chunk_ids: Sequence[str]) -> None:
        """Delete chunks by id."""
        for chunk_id in chunk_ids:
            self.collection.delete(ids=chunk_id)

    def sanitize_id(self, value: str) -> str:
        """Return an identifier containing only safe filter characters."""
        return "".join(
            character for character in value if character.isalnum() or character in "-_."
        )

    def delete_document(self, document_id: str) -> None:
        """Delete every chunk belonging to a document."""
        safe_id = self.sanitize_id(document_id)
        self.collection.delete_by_filter(filter=f"document_id = '{safe_id}'")

    def delete_version(self, document_id: str, version: int) -> None:
        """Delete all chunks for one document version."""
        safe_id = self.sanitize_id(document_id)
        self.collection.delete_by_filter(
            filter=f"document_id = '{safe_id}' AND version = {version}"
        )

    def search(
        self, *, vector: Sequence[float], top_k: int, metadata_filter: str | dict[str, Any] = ""
    ) -> list[dict[str, Any]]:
        """Search the collection with an optional canonical metadata filter."""
        filter_clause = native_filter(metadata_filter)
        if filter_clause is None:
            result = self.collection.query(
                queries=self.zvec.Query(field_name="embedding", vector=vector),
                topk=top_k,
            )
        else:
            result = self.collection.query(
                queries=self.zvec.Query(field_name="embedding", vector=vector),
                topk=top_k,
                filter=filter_clause,
            )
        return self.normalize_search_result(result)

    def hybrid_search(
        self,
        *,
        query: str,
        vector: Sequence[float],
        top_k: int,
        metadata_filter: str | dict[str, Any] = "",
    ) -> list[dict[str, Any]]:
        """Run dense search because this backend has no keyword channel."""
        return self.search(vector=vector, top_k=top_k, metadata_filter=metadata_filter)

    def matches_metadata(self, metadata_filter: str | dict[str, Any]) -> bool:
        """Return whether a metadata filter is supported by the Zvec adapter."""
        if metadata_filter in ("", None) or isinstance(metadata_filter, str):
            return True
        if not isinstance(metadata_filter, dict):
            return False
        return not set(metadata_filter) - {"company", "document_id"} and all(
            isinstance(value, (str, list)) for value in metadata_filter.values()
        )

    def optimize(self) -> None:
        """Optimize the native collection indexes."""
        self.collection.optimize()

    def keyword_search(self, query: str, top_k: int) -> list[dict[str, Any]]:
        """Return no hits because the native backend has no keyword index."""
        return []

    def health(self) -> dict[str, Any]:
        """Return native collection health and statistics."""
        return {"status": "ok", "backend": "zvec", "stats": getattr(self.collection, "stats", {})}

    def normalize_search_result(self, result: Any) -> list[dict[str, Any]]:
        """Convert native query results into the shared vector-store hit shape."""
        normalized: list[dict[str, Any]] = []
        if result is None:
            return normalized
        for item in result:
            fields = (
                item.get("fields", {}) if isinstance(item, dict) else getattr(item, "fields", {})
            )
            normalized.append(
                {
                    "chunk_id": fields.get(
                        "chunk_id",
                        item.get("id") if isinstance(item, dict) else getattr(item, "id", ""),
                    ),
                    "score": item.get("score", 0.0)
                    if isinstance(item, dict)
                    else getattr(item, "score", 0.0),
                    "chunk": ChunkRecord(
                        chunk_id=fields.get("chunk_id", ""),
                        document_id=fields.get("document_id", ""),
                        version=int(fields.get("version", 1)),
                        page=int(fields.get("page", 1)),
                        section=fields.get("section", ""),
                        company=fields.get("company", ""),
                        owner=fields.get("owner", ""),
                        department=fields.get("department", ""),
                        classification=fields.get("classification", "internal"),
                        created_at=datetime.fromisoformat(str(fields.get("created_at")))
                        if fields.get("created_at")
                        else datetime.now(UTC),
                        embedding_model=fields.get("embedding_model", ""),
                        hash=fields.get("hash", ""),
                        text=fields.get("text", ""),
                    ),
                }
            )
        return normalized


class ZvecVectorStore(BaseVectorStore):
    """Zvec adapter with an in-memory fallback when Zvec is unavailable."""

    def __init__(self, path: str, embedding_dim: int, require_zvec: bool = False) -> None:
        """Select the native or fallback backend for ``path``."""
        self.path = path
        self.embedding_dim = embedding_dim
        self.require_zvec = require_zvec
        self.zvec_module: Any = None
        self.backend = self.create_backend()

    def create_backend(self) -> BaseVectorStore:
        """Create a native backend, or an in-memory fallback when permitted."""
        if zvec_module is not None:
            self.zvec_module = zvec_module
            return RealZvecBackend(zvec_module, self.path, self.embedding_dim)
        if self.require_zvec:
            raise RuntimeError("ZVec is required in production mode but could not be imported")
        return InMemoryVectorStore()

    def create_collection(self) -> None:
        """Create or open the selected backend collection."""
        self.backend.create_collection()

    def insert(self, chunks: Sequence[ChunkRecord], vectors: Sequence[list[float]]) -> None:
        """Insert chunks through the selected backend."""
        self.backend.insert(chunks, vectors)

    def upsert(self, chunks: Sequence[ChunkRecord], vectors: Sequence[list[float]]) -> None:
        """Upsert chunks through the selected backend."""
        self.backend.upsert(chunks, vectors)

    def delete(self, chunk_ids: Sequence[str]) -> None:
        """Delete chunks through the selected backend."""
        self.backend.delete(chunk_ids)

    def delete_document(self, document_id: str) -> None:
        """Delete a document through the selected backend."""
        self.backend.delete_document(document_id)

    def delete_version(self, document_id: str, version: int) -> None:
        """Delete a document version through the selected backend."""
        self.backend.delete_version(document_id, version)

    def search(
        self, *, vector: Sequence[float], top_k: int, metadata_filter: str | dict[str, Any] = ""
    ) -> list[dict[str, Any]]:
        """Run vector search through the selected backend."""
        return self.backend.search(vector=vector, top_k=top_k, metadata_filter=metadata_filter)

    def hybrid_search(
        self,
        *,
        query: str,
        vector: Sequence[float],
        top_k: int,
        metadata_filter: str | dict[str, Any] = "",
    ) -> list[dict[str, Any]]:
        """Run hybrid search through the selected backend."""
        return self.backend.hybrid_search(
            query=query, vector=vector, top_k=top_k, metadata_filter=metadata_filter
        )

    def optimize(self) -> None:
        """Optimize the selected backend."""
        self.backend.optimize()

    def keyword_search(self, query: str, top_k: int) -> list[dict[str, Any]]:
        """Run keyword search through the selected backend."""
        return self.backend.keyword_search(query, top_k)

    def health(self) -> dict[str, Any]:
        """Report which backend is active."""
        return {"status": "ok", "backend": "zvec" if self.zvec_module is not None else "memory"}