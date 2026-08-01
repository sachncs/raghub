"""Vector-store implementations.

The module exposes the abstract contract, an in-process
``MemoryStore`` (cosine + BM25), and a ``SqliteStore``
that uses the ``sqlite-vector`` package from
``github.com/sqliteai/sqlite-vector`` when installed, or a SQLite +
NumPy fallback when not.
"""

from __future__ import annotations

import hashlib
import os
import sqlite3
from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass
from importlib.util import find_spec
from threading import RLock
from typing import Any, cast

import numpy as np
from rank_bm25 import BM25Okapi

from raghub.config import Settings
from raghub.errors import VectorStoreError
from raghub.models import Chunk, Classification

__all__ = [
    "MemoryStore",
    "SqliteStore",
    "Store",
    "build_store",
]


# ---------------------------------------------------------------------------
# Metadata helpers
# ---------------------------------------------------------------------------


def matches_metadata_dict(record: MemoryVectorRecord, filters: dict[str, Any]) -> bool:
    """Return whether ``record.chunk`` satisfies every entry in ``filters``.

    When ``expected`` is a list, the chunk's field must equal one of
    the list's elements (semantics used by RBAC filters where
    ``{"company": ["Apple", "Microsoft"]}`` means "any of these
    companies is acceptable").
    """
    chunk = record.chunk
    for key, expected in filters.items():
        if not hasattr(chunk, key):
            return False
        actual = getattr(chunk, key)
        if isinstance(expected, list):
            if actual not in expected:
                return False
        elif actual != expected:
            return False
    return True


def matches_metadata_string(record: MemoryVectorRecord, filter_string: str) -> bool:
    """Apply the legacy SQL-style filter string against ``record.chunk``."""
    if not filter_string:
        return True
    for clause in filter_string.split(" AND "):
        clause = clause.strip()
        if not clause:
            continue
        if " IN " in clause:
            prefix, _, values_part = clause.partition(" IN ")
            prefix = prefix.strip()
            values_part = values_part.strip().rstrip(")").lstrip("(")
            valid_values = {v.strip().strip("'") for v in values_part.split(",") if v.strip()}
            attr_name = prefix.split(".")[-1]
            if not hasattr(record.chunk, attr_name):
                return False
            if getattr(record.chunk, attr_name) not in valid_values:
                return False
        elif "=" in clause:
            key, _, value = clause.partition("=")
            attr_name = key.strip().split(".")[-1]
            expected = value.strip().strip("'")
            if not hasattr(record.chunk, attr_name):
                return False
            if str(getattr(record.chunk, attr_name)) != expected:
                return False
    return True


# ---------------------------------------------------------------------------
# Base contract
# ---------------------------------------------------------------------------


class Store(ABC):
    """Abstract base for every vector-store adapter."""

    @abstractmethod
    def create_collection(self) -> None:
        """Create the underlying collection when missing."""

    @abstractmethod
    def insert(self, chunks: Sequence[Chunk], vectors: Sequence[list[float]]) -> int:
        """Insert chunks paired with their precomputed embedding vectors.

        Returns:
            The number of rows written. Should equal ``len(chunks)``
            on success; backends that dedup by primary key may return
            a smaller value when the same ``chunk_id`` appears twice.

        """

    @abstractmethod
    def upsert(self, chunks: Sequence[Chunk], vectors: Sequence[list[float]]) -> int:
        """Insert-or-update chunks by primary key.

        Returns:
            The number of rows written (inserts + updates).

        """

    @abstractmethod
    def delete(self, chunk_ids: Sequence[str]) -> None:
        """Delete chunks by ``chunk_id``."""

    @abstractmethod
    def delete_document(self, document_id: str) -> None:
        """Delete every chunk that belongs to ``document_id``."""

    @abstractmethod
    def delete_version(self, document_id: str, version: int) -> None:
        """Delete every chunk that belongs to one ``(document_id, version)`` pair."""

    @abstractmethod
    def search(
        self, *, vector: Sequence[float], top_k: int, metadata_filter: str | dict[str, Any] = ""
    ) -> list[dict[str, Any]]:
        """Return the ``top_k`` hits closest to ``vector`` after pre-filtering."""

    @abstractmethod
    def hybrid_search(
        self,
        *,
        query: str,
        vector: Sequence[float],
        top_k: int,
        metadata_filter: str | dict[str, Any] = "",
    ) -> list[dict[str, Any]]:
        """Combine vector search with BM25 keyword search."""

    @abstractmethod
    def optimize(self) -> None:
        """Persist any in-memory state to durable storage."""

    @abstractmethod
    def keyword_search(self, query: str, top_k: int) -> list[dict[str, Any]]:
        """Run BM25 keyword search alone."""

    @abstractmethod
    def health(self) -> dict[str, Any]:
        """Return a liveness report for the ``/health`` endpoint."""


# ---------------------------------------------------------------------------
# In-process store (cosine + BM25, no external dep)
# ---------------------------------------------------------------------------


@dataclass
class MemoryVectorRecord:
    """A single chunk + its precomputed embedding vector."""

    chunk: Chunk
    vector: list[float]


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

    def create_collection(self) -> None:
        """No-op: this backend has no separate collection concept."""
        return None

    def insert(self, chunks: Sequence[Chunk], vectors: Sequence[list[float]]) -> int:
        """Insert or overwrite chunks by ``chunk_id``.

        The BM25 index is *not* rebuilt on every insert — that would
        make the per-chunk cost O(N) and turn a 1500-chunk ingest into
        a ~1M iteration job. Call :meth:`rebuild_index` once after a
        batch insert (the directory ingest path does this for you).

        Returns:
            Number of rows written (equals ``len(chunks)``).

        Raises:
            VectorStoreError: When a vector's dimension does not match
                ``self.embedding_dim``.

        """
        for vector in vectors:
            if len(vector) != self.embedding_dim:
                raise VectorStoreError(
                    f"vector dimension mismatch: expected {self.embedding_dim}, got {len(vector)}"
                )
        written = 0
        with self.lock:
            for chunk, vector in zip(chunks, vectors, strict=True):
                self.records[chunk.id] = MemoryVectorRecord(chunk=chunk, vector=vector)
                written += 1
        return written

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

    def materialize(self, hits: list[tuple[MemoryVectorRecord, float]]) -> list[dict[str, Any]]:
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
    ) -> list[dict[str, Any]]:
        """Combine cosine + BM25 via reciprocal-rank fusion."""
        if isinstance(metadata_filter, dict):
            dict_filter = metadata_filter
            str_filter: str | None = None
        else:
            dict_filter = None
            str_filter = metadata_filter
        with self.lock:
            records = [
                rec
                for rec in self.records.values()
                if (dict_filter is None or matches_metadata_dict(rec, dict_filter))
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
            rrf = 1.0 / (60.0 + dense_ranks) + 1.0 / (60.0 + bm25_ranks)
            order = np.argsort(-rrf)[:top_k]
        return self.materialize([(records[i], float(rrf[i])) for i in order])

    def optimize(self) -> None:
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


# ---------------------------------------------------------------------------
# SQLite-backed store
# ---------------------------------------------------------------------------


# Local path to the (optional) ``sqlite-vector`` Python package. The repo
# at github.com/sqliteai/sqlite-vector is not on PyPI as of writing; the
# wrapper tries the package first, then falls back to plain SQLite +
# NumPy cosine so the rest of the pipeline keeps working without it.
SQLITE_VECTOR_PKG: str | None = None
if find_spec("sqlite_vector") is not None:
    SQLITE_VECTOR_PKG = "sqlite_vector"
elif find_spec("sqlitevector") is not None:
    SQLITE_VECTOR_PKG = "sqlitevector"


class SqliteStore(Store):
    """Vector store backed by SQLite.

    Prefers the ``sqlite-vector`` extension package when installed
    (native ANN + filtering); otherwise falls back to a plain SQLite
    table with NumPy cosine similarity computed in Python. The wire
    format is identical so callers don't care which backend is active.
    """

    def __init__(
        self,
        *,
        path: str,
        embedding_dim: int,
        collection: str = "raghub",
    ) -> None:
        """Open the SQLite file, set WAL + foreign keys, and create schema."""
        self.path = path
        self.embedding_dim = embedding_dim
        self.collection = collection
        os.makedirs(self.dir_path(), exist_ok=True)
        self.conn = sqlite3.connect(self.db_path(), check_same_thread=False)
        self.conn.execute("PRAGMA journal_mode = WAL")
        self.conn.execute("PRAGMA foreign_keys = ON")
        self.lock = RLock()
        self.setup_schema()

    def dir_path(self) -> str:
        """Return the parent directory of the SQLite file path."""
        head, _ = os.path.split(self.path)
        return head or "."

    def db_path(self) -> str:
        """Return the configured SQLite file path."""
        return self.path

    def setup_schema(self) -> None:
        """Create the chunks table and its document/version index if absent."""
        with self.lock:
            self.conn.execute(
                f"""
                CREATE TABLE IF NOT EXISTS {self.collection} (
                    chunk_id TEXT PRIMARY KEY,
                    document_id TEXT NOT NULL,
                    version INTEGER NOT NULL,
                    classification TEXT,
                    text TEXT NOT NULL,
                    source_location TEXT,
                    company TEXT DEFAULT '',
                    owner TEXT DEFAULT '',
                    department TEXT DEFAULT '',
                    vector BLOB NOT NULL
                )
                """
            )
            self.conn.execute(
                f"""
                CREATE INDEX IF NOT EXISTS {self.collection}_doc
                    ON {self.collection}(document_id, version)
                """
            )
            self.conn.commit()

    @property
    def backend(self) -> str:
        """Identifier of the active backend."""
        return SQLITE_VECTOR_PKG or "sqlite-fallback"

    def create_collection(self) -> None:
        """No-op: schema is created at construction time."""
        return None

    @staticmethod
    def pack(vector: Sequence[float]) -> bytes:
        """Serialise ``vector`` to the on-disk blob format."""
        return np.asarray(vector, dtype=np.float32).tobytes()

    @staticmethod
    def unpack(blob: bytes) -> list[float]:
        """Deserialise ``blob`` back into a list of floats."""
        return np.frombuffer(blob, dtype=np.float32).tolist()

    def rows(
        self,
        metadata_filter: str | dict[str, Any] | None,
    ) -> list[tuple[str, str, int, str, str, str, str, str, str, bytes]]:
        """Return raw chunk rows matching ``metadata_filter``."""
        columns = (
            "chunk_id, document_id, version, classification, "
            "text, source_location, company, owner, department, vector"
        )
        if metadata_filter is None or metadata_filter == "":
            return list(
                self.conn.execute(
                    f"SELECT {columns} FROM {self.collection}"
                )
            )
        if isinstance(metadata_filter, dict):
            clauses = " AND ".join(f"{k} = ?" for k in metadata_filter)
            params = list(metadata_filter.values())
        else:
            clauses = metadata_filter
            params = []
        columns = (
            "chunk_id, document_id, version, classification, "
            "text, source_location, company, owner, department, vector"
        )
        return list(
            self.conn.execute(
                f"SELECT {columns} FROM {self.collection} WHERE {clauses}",
                params,
            )
        )

    def insert(self, chunks: Sequence[Chunk], vectors: Sequence[list[float]]) -> int:
        """Insert or overwrite chunks.

        Returns:
            ``cursor.rowcount`` after commit (number of rows written).

        Raises:
            VectorStoreError: When a vector's dimension does not match
                ``self.embedding_dim``.

        """
        for vector in vectors:
            if len(vector) != self.embedding_dim:
                raise VectorStoreError(
                    f"vector dimension mismatch: expected {self.embedding_dim}, got {len(vector)}"
                )
        with self.lock:
            cursor = self.conn.executemany(
                f"""
                INSERT OR IGNORE INTO {self.collection}
                    (chunk_id, document_id, version, classification,
                     text, source_location, company, owner,
                     department, vector)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    (
                        chunk.id,
                        chunk.document_id,
                        chunk.version,
                        chunk.classification.value,
                        chunk.text,
                        chunk.source_location,
                        chunk.company,
                        chunk.owner,
                        chunk.department,
                        self.pack(vector),
                    )
                    for chunk, vector in zip(chunks, vectors, strict=True)
                ],
            )
            self.conn.commit()
            return cursor.rowcount

    def upsert(self, chunks: Sequence[Chunk], vectors: Sequence[list[float]]) -> int:
        """Insert-or-update alias. Delegates to :meth:`insert`."""
        return self.insert(chunks, vectors)

    def delete(self, chunk_ids: Sequence[str]) -> None:
        """Delete chunks by ``chunk_id``."""
        with self.lock:
            self.conn.executemany(
                f"DELETE FROM {self.collection} WHERE chunk_id = ?",
                [(cid,) for cid in chunk_ids],
            )
            self.conn.commit()

    def delete_document(self, document_id: str) -> None:
        """Delete every chunk that belongs to ``document_id``."""
        with self.lock:
            self.conn.execute(
                f"DELETE FROM {self.collection} WHERE document_id = ?",
                (document_id,),
            )
            self.conn.commit()

    def delete_version(self, document_id: str, version: int) -> None:
        """Delete every chunk that belongs to one ``(document_id, version)`` pair."""
        with self.lock:
            self.conn.execute(
                f"DELETE FROM {self.collection} WHERE document_id = ? AND version = ?",
                (document_id, version),
            )
            self.conn.commit()

    def search(
        self, *, vector: Sequence[float], top_k: int, metadata_filter: str | dict[str, Any] = ""
    ) -> list[dict[str, Any]]:
        """Cosine-similarity search with metadata pre-filtering."""
        rows = self.rows(metadata_filter)
        if not rows:
            return []
        query = np.asarray(vector, dtype=np.float32)
        denom = float(np.linalg.norm(query)) or 1.0
        scored: list[tuple[Any, float]] = []
        for (
            chunk_id,
            document_id,
            version,
            classification,
            text,
            source_location,
            company,
            owner,
            department,
            blob,
        ) in rows:
            v = np.frombuffer(blob, dtype=np.float32)
            d = float(np.linalg.norm(v)) or 1.0
            score = float(np.dot(query, v) / (denom * d))
            scored.append(
                (
                    Chunk(
                        id=chunk_id,
                        document_id=document_id,
                        version=version,
                        classification=cast(Classification, classification),
                        text=text,
                        source_location=source_location,
                        company=company or "",
                        owner=owner or "",
                        department=department or "",
                        checksum=hashlib.sha256(text.encode("utf-8")).hexdigest(),
                    ),
                    score,
                )
            )
        scored.sort(key=lambda pair: pair[1], reverse=True)
        return [
            {
                "chunk": chunk,
                "score": score,
                "chunk_id": chunk.id,
                "document_id": chunk.document_id,
                "version": chunk.version,
            }
            for chunk, score in scored[:top_k]
        ]

    def hybrid_search(
        self,
        *,
        query: str,
        vector: Sequence[float],
        top_k: int,
        metadata_filter: str | dict[str, Any] = "",
    ) -> list[dict[str, Any]]:
        """Pure dense search — the SQL fallback has no native BM25 index.

        The ``sqlite-vector`` package adds BM25 when installed; the
        fallback uses dense search only.
        """
        return self.search(vector=vector, top_k=top_k, metadata_filter=metadata_filter)

    def optimize(self) -> None:
        """Force a checkpoint."""
        with self.lock:
            self.conn.commit()

    def keyword_search(self, query: str, top_k: int) -> list[dict[str, Any]]:
        """Substring match over the indexed text."""
        with self.lock:
            rows = list(
                self.conn.execute(
                    f"SELECT chunk_id, document_id, version, classification, "
                    f"text, source_location, company, owner, department, vector "
                    f"FROM {self.collection} WHERE text LIKE ?",
                    (f"%{query}%",),
                )
            )
        return [
            {
                "chunk": Chunk(
                    id=cid,
                    document_id=did,
                    version=ver,
                    classification=cls,
                    text=txt,
                    source_location=sloc,
                    company=co or "",
                    owner=ow or "",
                    department=dp or "",
                    checksum=hashlib.sha256(txt.encode("utf-8")).hexdigest(),
                ),
                "score": 1.0,
                "chunk_id": cid,
                "document_id": did,
                "version": ver,
            }
            for cid, did, ver, cls, txt, sloc, co, ow, dp, _ in rows[:top_k]
        ]

    def health(self) -> dict[str, Any]:
        """Return liveness information for the health endpoint."""
        with self.lock:
            count = self.conn.execute(f"SELECT COUNT(*) FROM {self.collection}").fetchone()[0]
        return {
            "status": "ok",
            "backend": self.backend,
            "chunks": count,
        }


# ---------------------------------------------------------------------------
# Default factory
# ---------------------------------------------------------------------------


def build_store(
    settings: Settings,
    *,
    embedding_dim: int | None = None,
) -> Store:
    """Return the configured vector store.

    The factory always returns a :class:`SqliteStore` pointed at
    ``settings.data_dir / "vectorstore.sqlite"``. The path can be
    overridden via the ``RAG_VECTORSTORE_PATH`` env var. When the
    ``sqlite-vector`` package is installed the store benefits from
    its native ANN + filtering; without it the same SQL tables back
    a NumPy cosine fallback.
    """
    from raghub.config import Settings

    if not isinstance(settings, Settings):
        raise TypeError(f"build_store: expected Settings, got {type(settings).__name__}")
    dim = embedding_dim if embedding_dim is not None else settings.embedding_dim
    override = os.environ.get("RAG_VECTORSTORE_PATH")
    path = override or str(settings.data_dir / "vectorstore.sqlite")
    return SqliteStore(path=path, embedding_dim=dim)
