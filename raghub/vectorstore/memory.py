"""Thread-safe in-memory vector store for tests, demos, and small workloads.

The adapter performs an ``O(n)`` cosine-similarity scan with metadata
filtering and BM25 keyword search. It is not intended for high-throughput
production workloads; use :class:`raghub.vectorstore.zvec.ZvecVectorStore`
or another persistent backend at scale.

Concurrency:
    Mutations and snapshot reads are protected by a re-entrant lock, so a
    store can be shared safely across threads.

Security:
    Legacy string filters accept only the ``company IN (...)`` and
    ``document_id = '...'`` forms. Unknown string filters fail closed.
    Canonical dict filters match :class:`ChunkRecord` fields directly.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass
from threading import RLock
from typing import Any, cast

import numpy as np
from rank_bm25 import BM25Okapi

from raghub.models import ChunkRecord
from raghub.vectorstore.base import BaseVectorStore


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
        vector: The raw embedding as a Python list of floats. Kept as
            ``list[float]`` rather than ``ndarray`` so the dataclass stays
            picklable for tests that roundtrip the store.
    """

    chunk: ChunkRecord
    vector: list[float]


class InMemoryVectorStore(BaseVectorStore):
    """Cosine-similarity vector store with BM25 keyword search.

    Search is performed by snapshotting the records under the lock,
    computing cosine similarity against the query vector, and returning
    the top ``top_k`` results sorted by descending score.
    """

    def __init__(self) -> None:
        """Initialise an empty store with a re-entrant lock."""
        # ``RLock`` lets nested ``with self.lock:`` blocks (e.g. when a
        # helper method needs to read while another method already holds
        # the lock) work without deadlocking on the same thread.
        self.lock = RLock()
        self.records: dict[str, MemoryVectorRecord] = {}

    def create_collection(self) -> None:
        """No-op: this backend has no separate collection concept."""
        return None

    def insert(self, chunks: Sequence[ChunkRecord], vectors: Sequence[list[float]]) -> None:
        """Insert or overwrite chunks by ``chunk_id``.

        Args:
            chunks: Chunk records to store. Must be the same length as
                ``vectors``.
            vectors: Corresponding embedding vectors.

        Raises:
            ValueError: If ``chunks`` and ``vectors`` differ in length
                (raised by :func:`zip`'s ``strict=True``).
        """
        with self.lock:
            for chunk, vector in zip(chunks, vectors, strict=True):
                self.records[chunk.chunk_id] = MemoryVectorRecord(chunk=chunk, vector=vector)

    def upsert(self, chunks: Sequence[ChunkRecord], vectors: Sequence[list[float]]) -> None:
        """Insert-or-update alias. Delegates to :meth:`insert`."""
        self.insert(chunks, vectors)

    def delete(self, chunk_ids: Sequence[str]) -> None:
        """Remove chunks by id, tolerating unknown ids.

        Args:
            chunk_ids: Chunk ids to remove. Unknown ids are silently
                skipped so retrying a partially-applied deletion is safe.
        """
        with self.lock:
            for chunk_id in chunk_ids:
                self.records.pop(chunk_id, None)

    def delete_document(self, document_id: str) -> None:
        """Remove every chunk belonging to ``document_id``.

        Args:
            document_id: The parent document whose chunks should be purged.
        """
        with self.lock:
            chunk_ids = [
                chunk_id
                for chunk_id, record in self.records.items()
                if record.chunk.document_id == document_id
            ]
            for chunk_id in chunk_ids:
                self.records.pop(chunk_id, None)

    def delete_version(self, document_id: str, version: int) -> None:
        """Remove chunks whose ``document_id`` and ``version`` match.

        Args:
            document_id: The parent document.
            version: The version number to remove; older versions remain.
        """
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

        Args:
            record: The candidate record.
            metadata_filter: A filter expression or empty string.

        Returns:
            ``True`` if the record passes the filter.
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
        """Compute cosine similarity in ``[0, 1]`` (clamped to zero for orthogonal vectors).

        Args:
            left: First vector.
            right: Second vector.

        Returns:
            The cosine similarity, or ``0.0`` if either vector is the
            zero vector (avoids division-by-zero).
        """
        lhs = np.asarray(left, dtype=np.float32)
        rhs = np.asarray(right, dtype=np.float32)
        # Denominator is the product of L2 norms. Zero-norm inputs
        # produce 0.0 rather than NaN so callers don't need defensive
        # checks for empty embeddings.
        denom = float(np.linalg.norm(lhs) * np.linalg.norm(rhs))
        if denom == 0:
            return 0.0
        return float(np.dot(lhs, rhs) / denom)

    def search(
        self, *, vector: Sequence[float], top_k: int, metadata_filter: str | dict = ""
    ) -> list[dict[str, Any]]:
        """Cosine-similarity search with metadata pre-filtering.

        Args:
            vector: Query embedding.
            top_k: Maximum number of hits to return.
            metadata_filter: Either a filter expression string
                (legacy; see :meth:`matches_filter`) or a ``dict``
                keyed by :class:`ChunkRecord` field name. ``dict``
                matches are equality checks.

        Returns:
            A list of hit dicts with keys ``chunk_id``, ``score``,
            ``chunk`` sorted by descending score and trimmed to ``top_k``.
        """
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
        metadata_filter: str | dict = "",
    ) -> list[dict[str, Any]]:
        """Hybrid search shim. The in-memory backend collapses to vector search.

        The keyword channel is intentionally not implemented here because
        the in-memory store is primarily a test fixture. Production hybrid
        fusion lives in :class:`raghub.retrieval.pipeline.RetrievalPipeline`.

        Args:
            query: Raw query text (unused by this backend).
            vector: Query embedding.
            top_k: Maximum number of hits.
            metadata_filter: Filter expression.

        Returns:
            The same hit shape as :meth:`search`.
        """
        return self.search(vector=vector, top_k=top_k, metadata_filter=metadata_filter)

    def keyword_search(self, query: str, top_k: int) -> list[dict[str, Any]]:
        """Return BM25-ranked chunks containing the query terms.

        The corpus is rebuilt from a lock-protected record snapshot on every
        call, which is appropriate for the small workloads supported by this
        adapter.
        """
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
        """Report liveness information for the health endpoint.

        Returns:
            A dict with ``status`` (always ``"ok"`` for this backend),
            ``backend`` identifier, and the current chunk count.
        """
        return {"status": "ok", "backend": "memory", "chunks": len(self.records)}


__all__ = ["InMemoryVectorStore", "MemoryVectorRecord", "matches_metadata_dict"]
