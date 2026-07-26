"""Zvec vector-store adapter with a thread-safe in-memory fallback.

The module translates canonical metadata filters to Zvec expressions, wraps
the native collection API, and delegates to the in-memory backend when Zvec
is unavailable and production mode does not require it.
"""

from __future__ import annotations

import os
from collections.abc import Sequence
from datetime import UTC, datetime
from importlib import import_module
from importlib.util import find_spec
from typing import Any

from raghub.models import ChunkRecord
from raghub.vectorstore.base import BaseVectorStore
from raghub.vectorstore.memory import InMemoryVectorStore

zvec_module = import_module("zvec") if find_spec("zvec") is not None else None


def native_filter(metadata_filter: str | dict) -> str | None:
    """Translate a canonical metadata filter into a Zvec SQL fragment.

    Args:
        metadata_filter: Canonical dict, legacy string, or empty string.

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
        self, *, vector: Sequence[float], top_k: int, metadata_filter: str | dict = ""
    ) -> list[dict[str, Any]]:
        """Search the collection with an optional canonical metadata filter."""
        filter_clause = self.native_filter(metadata_filter)
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

    @staticmethod
    def native_filter(metadata_filter: str | dict) -> str | None:
        """Translate a canonical metadata filter into a Zvec SQL fragment."""
        return native_filter(metadata_filter)

    def hybrid_search(
        self,
        *,
        query: str,
        vector: Sequence[float],
        top_k: int,
        metadata_filter: str | dict = "",
    ) -> list[dict[str, Any]]:
        """Run dense search because this backend has no keyword channel."""
        return self.search(vector=vector, top_k=top_k, metadata_filter=metadata_filter)

    def matches_metadata(self, metadata_filter: str | dict) -> bool:
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
        self, *, vector: Sequence[float], top_k: int, metadata_filter: str | dict = ""
    ) -> list[dict[str, Any]]:
        """Run vector search through the selected backend."""
        return self.backend.search(vector=vector, top_k=top_k, metadata_filter=metadata_filter)

    def hybrid_search(
        self,
        *,
        query: str,
        vector: Sequence[float],
        top_k: int,
        metadata_filter: str | dict = "",
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


__all__ = ["RealZvecBackend", "ZvecVectorStore", "native_filter"]
