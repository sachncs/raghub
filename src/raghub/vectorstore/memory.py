"""In-memory vector store for local development and tests."""

from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Sequence
from threading import RLock
import re
from typing import Any, cast

import numpy as np

from raghub.models import ChunkRecord
from raghub.vectorstore.base import BaseVectorStore


@dataclass
class MemoryVectorRecord:
    chunk: ChunkRecord
    vector: list[float]


class InMemoryVectorStore(BaseVectorStore):
    """Simple cosine similarity vector store."""

    def __init__(self) -> None:
        self._lock = RLock()
        self._records: dict[str, MemoryVectorRecord] = {}

    def create_collection(self) -> None:
        return None

    def insert(self, chunks: Sequence[ChunkRecord], vectors: Sequence[list[float]]) -> None:
        with self._lock:
            for chunk, vector in zip(chunks, vectors, strict=True):
                self._records[chunk.chunk_id] = MemoryVectorRecord(chunk=chunk, vector=vector)

    def upsert(self, chunks: Sequence[ChunkRecord], vectors: Sequence[list[float]]) -> None:
        self.insert(chunks, vectors)

    def delete(self, chunk_ids: Sequence[str]) -> None:
        with self._lock:
            for chunk_id in chunk_ids:
                self._records.pop(chunk_id, None)

    def delete_document(self, document_id: str) -> None:
        with self._lock:
            chunk_ids = [chunk_id for chunk_id, record in self._records.items() if record.chunk.document_id == document_id]
            for chunk_id in chunk_ids:
                self._records.pop(chunk_id, None)

    def delete_version(self, document_id: str, version: int) -> None:
        with self._lock:
            chunk_ids = [
                chunk_id
                for chunk_id, record in self._records.items()
                if record.chunk.document_id == document_id and record.chunk.version == version
            ]
            for chunk_id in chunk_ids:
                self._records.pop(chunk_id, None)

    def _matches(self, record: MemoryVectorRecord, metadata_filter: str) -> bool:
        if not metadata_filter:
            return True
        company_match = re.search(r"company\s+IN\s+\((.+)\)", metadata_filter, flags=re.IGNORECASE)
        if company_match:
            allowed = [item.strip().strip("'\"") for item in company_match.group(1).split(",")]
            return record.chunk.company in allowed
        document_match = re.search(r"document_id\s*=\s*'([^']+)'", metadata_filter, flags=re.IGNORECASE)
        if document_match:
            return record.chunk.document_id == document_match.group(1)
        return True

    def _score(self, left: Sequence[float], right: Sequence[float]) -> float:
        lhs = np.asarray(left, dtype=np.float32)
        rhs = np.asarray(right, dtype=np.float32)
        denom = float(np.linalg.norm(lhs) * np.linalg.norm(rhs))
        if denom == 0:
            return 0.0
        return float(np.dot(lhs, rhs) / denom)

    def search(self, *, vector: Sequence[float], top_k: int, metadata_filter: str) -> list[dict[str, Any]]:
        with self._lock:
            records = [record for record in self._records.values() if self._matches(record, metadata_filter)]
        hits = [
            {
                "chunk_id": record.chunk.chunk_id,
                "score": self._score(vector, record.vector),
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
        metadata_filter: str,
    ) -> list[dict[str, Any]]:
        return self.search(vector=vector, top_k=top_k, metadata_filter=metadata_filter)

    def optimize(self) -> None:
        return None

    def health(self) -> dict[str, Any]:
        return {"status": "ok", "backend": "memory", "chunks": len(self._records)}
