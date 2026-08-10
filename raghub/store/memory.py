"""In-process vector store (cosine + BM25, no external dependency)."""

from __future__ import annotations

from collections.abc import Sequence
from threading import RLock
from typing import Any

import numpy as np
from rank_bm25 import BM25Okapi

from raghub.constants import RRF_K
from raghub.errors import VectorStoreError
from raghub.models import Chunk
from raghub.store.base import (
    MemoryVectorRecord,
    Store,
    matches_metadata_dict,
    matches_metadata_string,
)


class MemoryStore(Store):
    """Cosine-similarity vector store with BM25 keyword search."""

    def __init__(self, embedding_dim: int) -> None:
        """Initialise an empty store with a re-entrant lock.

        Args:
            embedding_dim: Dimensionality of vectors that will be
                inserted. Mismatched dimensions raise
                :class:`VectorStoreError` on insert.

        """
        self.embedding_dim = embedding_dim
        self.lock = RLock()
        self.records: dict[str, MemoryVectorRecord] = {}
        self.bm25: BM25Okapi | None = None

    @staticmethod
    def create_collection() -> None:
        """No-op: this backend has no separate collection concept."""
        return None

    def insert(self, chunks: Sequence[Chunk], vectors: Sequence[list[float]]) -> int:
        """Insert or overwrite chunks by ``chunk_id``.

        The BM25 index is *not* rebuilt on every insert — that would
        make the per-chunk cost O(N) and turn a 1500-chunk ingest into
        a ~1M iteration job. Call :meth:`rebuild_index` once after a
        batch insert (the directory ingest path does this for you).

        Returns:
            Number of chunks the caller submitted (equals ``len(chunks)``).
            Dedup is silent — a re-inserted ``chunk_id`` overwrites the
            existing record but is still counted as one submitted chunk.

        Raises:
            VectorStoreError: When a vector's dimension does not match
                ``self.embedding_dim``.

        """
        for vector in vectors:
            if len(vector) != self.embedding_dim:
                raise VectorStoreError(
                    f"vector dimension mismatch: expected {self.embedding_dim}, got {len(vector)}"
                )
        for chunk in chunks:
            chunk.verify()
        with self.lock:
            for chunk, vector in zip(chunks, vectors, strict=True):
                self.records[chunk.id] = MemoryVectorRecord(chunk=chunk, vector=vector)
        return len(chunks)

    def rebuild_index(self) -> None:
        """Rebuild the BM25 index over the current record set."""
        self.rebuild_bm25()

    def upsert(self, chunks: Sequence[Chunk], vectors: Sequence[list[float]]) -> int:
        """Insert-or-update alias. Delegates to :meth:`insert`."""
        return self.insert(chunks, vectors)

    def delete(self, chunk_ids: Sequence[str]) -> None:
        """Delete chunks by ``chunk_id``."""
        with self.lock:
            for cid in chunk_ids:
                self.records.pop(cid, None)
        self.rebuild_bm25()

    def delete_document(self, document_id: str) -> None:
        """Delete every chunk that belongs to ``document_id``."""
        with self.lock:
            stale = [
                cid for cid, rec in self.records.items() if rec.chunk.document_id == document_id
            ]
        self.delete(stale)

    def delete_version(self, document_id: str, version: int) -> None:
        """Delete every chunk that belongs to one ``(document_id, version)`` pair."""
        with self.lock:
            stale = [
                cid
                for cid, rec in self.records.items()
                if rec.chunk.document_id == document_id and rec.chunk.version == version
            ]
        self.delete(stale)

    def rebuild_bm25(self) -> None:
        """Recompute the BM25 index from the current record set."""
        with self.lock:
            if not self.records:
                self.bm25 = None
                return
            corpus = [rec.chunk.text.split() for rec in self.records.values()]
        self.bm25 = BM25Okapi(corpus)

    @staticmethod
    def materialize(hits: list[tuple[MemoryVectorRecord, float]]) -> list[dict[str, Any]]:
        """Convert internal hit tuples into the public dict shape."""
        out: list[dict[str, Any]] = []
        for rec, score in hits:
            out.append(
                {
                    "chunk": rec.chunk,
                    "score": score,
                    "chunk_id": rec.chunk.id,
                    "document_id": rec.chunk.document_id,
                    "version": rec.chunk.version,
                }
            )
        return out

    @staticmethod
    def compute_score(left: Sequence[float], right: Sequence[float]) -> float:
        """Compute cosine similarity in ``[0, 1]``."""
        lhs = np.asarray(left, dtype=np.float32)
        rhs = np.asarray(right, dtype=np.float32)
        denom = float(np.linalg.norm(lhs) * np.linalg.norm(rhs))
        if denom == 0:
            return 0.0
        return float(np.dot(lhs, rhs) / denom)

    def search(
        self,
        *,
        vector: Sequence[float],
        top_k: int,
        metadata_filter: str | dict[str, Any] = "",
        tenant_id: str | None = None,
    ) -> list[dict[str, Any]]:
        """Cosine-similarity search with metadata and tenant pre-filtering.

        Args:
            vector: Query embedding.
            top_k: Number of hits to return.
            metadata_filter: Optional metadata filter (string SQL or dict).
            tenant_id: Optional explicit tenant id; if ``None`` and a
                tenant context is bound via :func:`set_current`,
                that tenant id is used. Records whose ``chunk.tenant_id``
                does not match are excluded.

        """
        if isinstance(metadata_filter, dict):
            dict_filter = metadata_filter
            str_filter: str | None = None
        else:
            dict_filter = None
            str_filter = metadata_filter
        # Tier 2 Item 9: tenant_id isolation
        from raghub.tenants.isolation import RowLevel

        effective_tenant = (
            RowLevel()
            .apply_to_kwargs({} if tenant_id is None else {"tenant_id": tenant_id})
            .get("tenant_id")
        )
        with self.lock:
            records = [
                record
                for record in self.records.values()
                if (
                    effective_tenant is None
                    or getattr(record.chunk, "tenant_id", None) == effective_tenant
                )
                and (dict_filter is None or matches_metadata_dict(record, dict_filter))
                and (dict_filter is not None or matches_metadata_string(record, str_filter or ""))
            ]
            scored = [(rec, self.compute_score(vector, rec.vector)) for rec in records]
        scored.sort(key=lambda pair: pair[1], reverse=True)
        return self.materialize(scored[:top_k])

    def hybrid_search(
        self,
        *,
        query: str,
        vector: Sequence[float],
        top_k: int,
        metadata_filter: str | dict[str, Any] = "",
        tenant_id: str | None = None,
    ) -> list[dict[str, Any]]:
        """Combine cosine + BM25 via reciprocal-rank fusion."""
        if isinstance(metadata_filter, dict):
            dict_filter = metadata_filter
            str_filter: str | None = None
        else:
            dict_filter = None
            str_filter = metadata_filter
        # Tier 2 Item 9: tenant_id isolation
        from raghub.tenants.isolation import RowLevel

        effective_tenant = (
            RowLevel()
            .apply_to_kwargs({} if tenant_id is None else {"tenant_id": tenant_id})
            .get("tenant_id")
        )
        with self.lock:
            records = [
                rec
                for rec in self.records.values()
                if (
                    effective_tenant is None
                    or getattr(rec.chunk, "tenant_id", None) == effective_tenant
                )
                and (dict_filter is None or matches_metadata_dict(rec, dict_filter))
                and (dict_filter is not None or matches_metadata_string(rec, str_filter or ""))
            ]
            ids = [rec.chunk.id for rec in records]
            if not ids:
                return []
            dense_scores = np.array([self.compute_score(vector, rec.vector) for rec in records])
            if self.bm25 is not None:
                bm25_scores = np.array(self.bm25.get_scores(query.split()))
            else:
                bm25_scores = np.zeros_like(dense_scores)
            dense_ranks = (-dense_scores).argsort().argsort()
            bm25_ranks = (-bm25_scores).argsort().argsort()
            rrf = 1.0 / (RRF_K + dense_ranks) + 1.0 / (RRF_K + bm25_ranks)
            order = np.argsort(-rrf)[:top_k]
        return self.materialize([(records[i], float(rrf[i])) for i in order])

    @staticmethod
    def optimize() -> None:
        """No-op for the in-process backend."""
        return None

    def keyword_search(self, query: str, top_k: int) -> list[dict[str, Any]]:
        """Pure BM25 keyword search."""
        with self.lock:
            if not self.records or self.bm25 is None:
                return []
            chunk_ids = list(self.records.keys())
            scores = self.bm25.get_scores(query.split())
            order = np.argsort(-scores)[:top_k]
            return [
                {
                    "chunk": self.records[chunk_ids[i]].chunk,
                    "score": float(scores[i]),
                    "chunk_id": chunk_ids[i],
                    "document_id": self.records[chunk_ids[i]].chunk.document_id,
                    "version": self.records[chunk_ids[i]].chunk.version,
                }
                for i in order
            ]

    def health(self) -> dict[str, Any]:
        """Return liveness information for the health endpoint."""
        return {"status": "ok", "backend": "memory", "chunks": len(self.records)}
