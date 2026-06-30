"""Alibaba Zvec adapter."""

from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime, timezone
from typing import Any

from raghub.models import ChunkRecord
from raghub.vectorstore.base import BaseVectorStore
from raghub.vectorstore.memory import InMemoryVectorStore


class RealZvecBackend(BaseVectorStore):
    """Adapter around the Alibaba Zvec Python API."""

    def __init__(self, zvec_module: Any, path: str, embedding_dim: int) -> None:
        self.zvec = zvec_module
        self.path = path
        self.embedding_dim = embedding_dim
        self.collection = self.open_collection()

    def open_collection(self) -> Any:
        import os

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
        return None

    def insert(self, chunks: Sequence[ChunkRecord], vectors: Sequence[list[float]]) -> None:
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
        self.insert(chunks, vectors)

    def delete(self, chunk_ids: Sequence[str]) -> None:
        for chunk_id in chunk_ids:
            self.collection.delete(ids=chunk_id)

    def sanitize_id(self, value: str) -> str:
        return "".join(c for c in value if c.isalnum() or c in "-_.")

    def delete_document(self, document_id: str) -> None:
        safe_id = self.sanitize_id(document_id)
        self.collection.delete_by_filter(filter=f"document_id = '{safe_id}'")

    def delete_version(self, document_id: str, version: int) -> None:
        safe_id = self.sanitize_id(document_id)
        self.collection.delete_by_filter(filter=f"document_id = '{safe_id}' AND version = {version}")

    def search(self, *, vector: Sequence[float], top_k: int, metadata_filter: str) -> list[dict[str, Any]]:
        result = self.collection.query(
            queries=self.zvec.Query(field_name="embedding", vector=vector),
            topk=top_k,
            filter=metadata_filter,
        )
        return self.normalize_search_result(result)

    def hybrid_search(
        self,
        *,
        query: str,
        vector: Sequence[float],
        top_k: int,
        metadata_filter: str,
    ) -> list[dict[str, Any]]:
        return self.search(vector=vector, top_k=top_k, metadata_filter=metadata_filter)

    def optimize(self) -> None:
        self.collection.optimize()

    def keyword_search(self, query: str, top_k: int) -> list[dict[str, Any]]:
        return []

    def health(self) -> dict[str, Any]:
        return {"status": "ok", "backend": "zvec", "stats": getattr(self.collection, "stats", {})}

    def normalize_search_result(self, result: Any) -> list[dict[str, Any]]:
        normalized: list[dict[str, Any]] = []
        if result is None:
            return normalized
        for item in result:
            fields = item.get("fields", {}) if isinstance(item, dict) else getattr(item, "fields", {})
            normalized.append(
                {
                    "chunk_id": fields.get("chunk_id", item.get("id") if isinstance(item, dict) else getattr(item, "id", "")),
                    "score": item.get("score", 0.0) if isinstance(item, dict) else getattr(item, "score", 0.0),
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
                        else datetime.now(timezone.utc),
                        embedding_model=fields.get("embedding_model", ""),
                        hash=fields.get("hash", ""),
                        text=fields.get("text", ""),
                    ),
                }
            )
        return normalized


class ZvecVectorStore(BaseVectorStore):
    """Zvec adapter with in-memory fallback when the dependency is unavailable."""

    def __init__(self, path: str, embedding_dim: int, require_zvec: bool = False) -> None:
        self.path = path
        self.embedding_dim = embedding_dim
        self.require_zvec = require_zvec
        self.backend: BaseVectorStore
        self.zvec_module: Any = None
        self.backend = self.create_backend()

    def create_backend(self) -> BaseVectorStore:
        try:
            import zvec

            self.zvec_module = zvec
            return RealZvecBackend(zvec, self.path, self.embedding_dim)
        except ImportError as exc:
            if self.require_zvec:
                raise RuntimeError(
                    "ZVec is required in production mode but could not be imported"
                ) from exc
            return InMemoryVectorStore()

    def create_collection(self) -> None:
        self.backend.create_collection()

    def insert(self, chunks: Sequence[ChunkRecord], vectors: Sequence[list[float]]) -> None:
        self.backend.insert(chunks, vectors)

    def upsert(self, chunks: Sequence[ChunkRecord], vectors: Sequence[list[float]]) -> None:
        self.backend.upsert(chunks, vectors)

    def delete(self, chunk_ids: Sequence[str]) -> None:
        self.backend.delete(chunk_ids)

    def delete_document(self, document_id: str) -> None:
        self.backend.delete_document(document_id)

    def delete_version(self, document_id: str, version: int) -> None:
        self.backend.delete_version(document_id, version)

    def search(self, *, vector: Sequence[float], top_k: int, metadata_filter: str) -> list[dict[str, Any]]:
        return self.backend.search(vector=vector, top_k=top_k, metadata_filter=metadata_filter)

    def hybrid_search(
        self,
        *,
        query: str,
        vector: Sequence[float],
        top_k: int,
        metadata_filter: str,
    ) -> list[dict[str, Any]]:
        return self.backend.hybrid_search(query=query, vector=vector, top_k=top_k, metadata_filter=metadata_filter)

    def optimize(self) -> None:
        self.backend.optimize()

    def keyword_search(self, query: str, top_k: int) -> list[dict[str, Any]]:
        return self.backend.keyword_search(query, top_k)

    def health(self) -> dict[str, Any]:
        return {"status": "ok", "backend": "zvec" if self.zvec_module is not None else "memory"}
