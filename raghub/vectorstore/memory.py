"""In-memory vector store used for development, tests, and tiny workloads.

This module implements a simple dictionary-backed vector store with cosine
similarity search, RBAC-aware metadata filtering, and a naive keyword
fallback. It is **not** a high-throughput system: every query scans the
full record set under a re-entrant lock, so the cost is ``O(n)`` per
search. Use it for tests, demos, and single-tenant deployments of a few
thousand chunks. For production scale use
:class:`raghub.vectorstore.zvec.ZvecVectorStore` or another persistent
backend.

Concurrency:
    The store uses an :class:`threading.RLock` around all mutating and
    snapshot reads. Callers may safely share an instance across threads
    without external locking; the snapshot taken in :meth:`search` is
    consistent because the lock is held while building it.

Security:
    The metadata filter parser understands only the ``company IN (...)``
    and ``document_id = '...'`` shapes emitted by
    :func:`raghub.core.rbac.allowed_company_filter` and the application
    code. Unknown filter fragments are silently ignored (treated as no
    constraint) rather than raising — see :meth:`matches_filter`.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass
from threading import RLock
from typing import Any, cast

import numpy as np

from raghub.models import ChunkRecord
from raghub.vectorstore.base import BaseVectorStore


def matches_metadata_dict(record: "MemoryVectorRecord", filters: dict[str, Any]) -> bool:
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
    """Cosine-similarity vector store with naive keyword fallback.

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
            chunk_ids = [chunk_id for chunk_id, record in self.records.items() if record.chunk.document_id == document_id]
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

        Anything else is treated as **no constraint** (returns ``True``).
        This permissive behaviour makes the store forgiving of feature
        gaps during development; tighten it once the broader filter DSL is
        wired through.

        Args:
            record: The candidate record.
            metadata_filter: A filter expression or empty string.

        Returns:
            ``True`` if the record passes the filter (or there is none).
        """
        if not metadata_filter:
            return True
        # ``company IN (...)``: extract the comma-separated list, strip
        # whitespace and either kind of quote, and test membership.
        company_match = re.search(r"company\s+IN\s+\((.+)\)", metadata_filter, flags=re.IGNORECASE)
        if company_match:
            allowed = [item.strip().strip("'\"") for item in company_match.group(1).split(",")]
            return record.chunk.company in allowed
        # ``document_id = '...'``: simple equality on the literal.
        document_match = re.search(r"document_id\s*=\s*'([^']+)'", metadata_filter, flags=re.IGNORECASE)
        if document_match:
            return record.chunk.document_id == document_match.group(1)
        # Unknown filter fragment — fail open. See class docstring note.
        return True

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

    def search(self, *, vector: Sequence[float], top_k: int, metadata_filter: str | dict = "") -> list[dict[str, Any]]:
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
        """Naive term-frequency keyword search.

        The score for a chunk is ``sum(token_count_in_chunk) /
        len(chunk_text_tokens)`` across all query tokens present in the
        chunk. **This is not BM25/TF-IDF**: there is no IDF weighting, no
        length saturation, and no per-token inverse document frequency.
        The score therefore favours very short chunks that happen to
        contain query terms.

        NOTE: replace with BM25 (or delegate to a real inverted index) when
        persistence and recall-quality requirements demand proper IDF
        weighting.

        Args:
            query: Raw query string.
            top_k: Maximum number of hits.

        Returns:
            A list of hit dicts sorted by descending score. Empty query
            yields an empty list.
        """
        query_terms = query.lower().split()
        if not query_terms:
            return []
        # Snapshot under lock so iteration is safe against concurrent
        # inserts/deletes.
        with self.lock:
            records = list(self.records.values())
        scored: list[tuple[str, float, ChunkRecord]] = []
        for rec in records:
            text = rec.chunk.text.lower()
            text_terms = text.split()
            if not text_terms:
                continue
            # Raw TF / chunk-length: over-counts for short chunks because
            # every query term is divided by the same denominator.
            score = sum(text_terms.count(q) for q in query_terms) / len(text_terms)
            if score > 0:
                scored.append((rec.chunk.chunk_id, score, rec.chunk))
        scored.sort(key=lambda x: x[1], reverse=True)
        return [
            {"chunk_id": cid, "score": s, "chunk": c}
            for cid, s, c in scored[:top_k]
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
