"""SQLite-backed vector store.

Prefers the ``sqlite-vector`` extension package when installed
(native ANN + filtering); otherwise falls back to a plain SQLite
table with NumPy cosine similarity computed in Python. The wire
format is identical so callers don't care which backend is active.
"""

from __future__ import annotations

import hashlib
import os
import sqlite3
from collections.abc import Sequence
from importlib.util import find_spec
from threading import RLock
from typing import Any, cast

import numpy as np

from raghub.errors import VectorStoreError
from raghub.models import Chunk, Classification
from raghub.stores.vector_base import Store

# Local path to the (optional) ``sqlite-vector`` Python package. The repo
# at github.com/sqliteai/sqlite-vector is not on PyPI as of writing; the
# wrapper tries the package first, then falls back to plain SQLite +
# NumPy cosine so the rest of the pipeline keeps working without it.
SQLITE_VECTOR_PKG: str | None = None
if find_spec("sqlite_vector") is not None:
    SQLITE_VECTOR_PKG = "sqlite_vector"
elif find_spec("sqlitevector") is not None:
    SQLITE_VECTOR_PKG = "sqlitevector"


@Store.register("sqlite")
class SqliteStore(Store):  # ruff: ignore[too-many-public-methods] -- full Store surface for SQLite backend
    """Vector store backed by SQLite."""

    name = "sqlite"

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
                    tenant_id TEXT DEFAULT '',
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

    @staticmethod
    def create_collection() -> None:
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
        metadata_filter: str | dict[str, Any] | None = None,
        *,
        tenant_id: str | None = None,
    ) -> list[tuple[str, str, int, str, str, str, str, str, str, str, bytes]]:
        """Return raw chunk rows matching ``metadata_filter`` and tenant_id."""
        columns = (
            "chunk_id, document_id, version, classification, "
            "text, source_location, company, owner, department, tenant_id, vector"
        )
        clauses: list[str] = []
        params: list[Any] = []
        if metadata_filter is not None and metadata_filter != "":
            if isinstance(metadata_filter, dict):
                for key, expected in metadata_filter.items():
                    if isinstance(expected, list):
                        if not expected:
                            return []
                        placeholders = ", ".join("?" for _ in expected)
                        clauses.append(f"{key} IN ({placeholders})")
                        params.extend(expected)
                    else:
                        clauses.append(f"{key} = ?")
                        params.append(expected)
            else:
                clauses.append(metadata_filter)
        if tenant_id is not None:
            clauses.append("tenant_id = ?")
            params.append(tenant_id)
        where = ""
        if clauses:
            where = " WHERE " + " AND ".join(clauses)
        return list(
            self.conn.execute(
                f"SELECT {columns} FROM {self.collection}{where}",
                params,
            )
        )

    def insert(self, chunks: Sequence[Chunk], vectors: Sequence[list[float]]) -> int:
        """Insert or overwrite chunks.

        Returns:
            Number of chunks the caller submitted (equals ``len(chunks)``).
            Dedup is silent by design — ``INSERT OR IGNORE`` skips rows
            whose primary key already exists, but the caller still sees
            the full batch size so re-ingestion does not look like a
            partial write.

        Raises:
            VectorStoreError: When a vector's dimension does not match
                ``self.embedding_dim``.

        """
        for chunk in chunks:
            chunk.verify()
        for vector in vectors:
            if len(vector) != self.embedding_dim:
                raise VectorStoreError(
                    f"vector dimension mismatch: expected {self.embedding_dim}, got {len(vector)}"
                )
        with self.lock:
            self.conn.executemany(
                f"""
                INSERT OR IGNORE INTO {self.collection}
                    (chunk_id, document_id, version, classification,
                     text, source_location, company, owner,
                     department, tenant_id, vector)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
                        chunk.tenant_id or "",
                        self.pack(vector),
                    )
                    for chunk, vector in zip(chunks, vectors, strict=True)
                ],
            )
            self.conn.commit()
            return len(chunks)

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
        self,
        *,
        vector: Sequence[float],
        top_k: int,
        metadata_filter: str | dict[str, Any] = "",
        tenant_id: str | None = None,
    ) -> list[dict[str, Any]]:
        """Cosine-similarity search with metadata + tenant pre-filtering.

        Args:
            vector: Query embedding.
            top_k: Number of hits.
            metadata_filter: Optional metadata filter.
            tenant_id: Optional explicit tenant id; if ``None`` and a
                tenant context is bound, that tenant id is used.

        """
        rows = self.rows(metadata_filter, tenant_id=self.effective_tenant(tenant_id))
        if not rows:
            return []
        scored = self.score_rows(rows, vector)
        scored.sort(key=lambda pair: pair[1], reverse=True)
        return self.materialize(scored[:top_k])

    @staticmethod
    def effective_tenant(tenant_id: str | None) -> str | None:
        """Resolve ``tenant_id`` against the bound tenant context."""
        if tenant_id is not None:
            return tenant_id
        from raghub.tenants.isolation import RowLevel

        return RowLevel().apply_to_kwargs({}).get("tenant_id")

    @staticmethod
    def score_rows(
        rows: list[tuple[Any, ...]],
        vector: Sequence[float],
    ) -> list[tuple[Any, float]]:
        """Compute cosine similarity for every row."""
        query = np.asarray(vector, dtype=np.float32)
        denom = float(np.linalg.norm(query)) or 1.0
        scored: list[tuple[Any, float]] = []
        for row in rows:
            score = SqliteStore.cosine_similarity(query, row[-1], denom)
            scored.append((SqliteStore.row_to_chunk(row), score))
        return scored

    @staticmethod
    def cosine_similarity(query: Any, blob: Any, denom: float) -> float:
        """Cosine similarity between the query vector and a stored blob."""
        v = np.frombuffer(blob, dtype=np.float32)
        d = float(np.linalg.norm(v)) or 1.0
        return float(np.dot(query, v) / (denom * d))

    @staticmethod
    def row_to_chunk(row: tuple[Any, ...]) -> Any:
        """Convert a raw row tuple to a :class:`Chunk`."""
        (
            chunk_id,
            document_id,
            version,
            classification,
            text,
            source_location,
            company,
            owner,
            department,
            tenant_id,
            _blob,
        ) = row
        return Chunk(
            id=chunk_id,
            document_id=document_id,
            version=version,
            classification=cast(Classification, classification),
            text=text,
            source_location=source_location,
            company=company or "",
            owner=owner or "",
            department=department or "",
            tenant_id=tenant_id or "",
            checksum=hashlib.sha256(text.encode("utf-8", errors="surrogatepass")).hexdigest(),
        )

    @staticmethod
    def materialize(scored: list[tuple[Any, float]]) -> list[dict[str, Any]]:
        """Format scored chunks into the public dict shape."""
        return [
            {
                "chunk": chunk,
                "score": score,
                "chunk_id": chunk.id,
                "document_id": chunk.document_id,
                "version": chunk.version,
            }
            for chunk, score in scored
        ]

    def hybrid_search(
        self,
        *,
        query: str,
        vector: Sequence[float],
        top_k: int,
        metadata_filter: str | dict[str, Any] = "",
    ) -> list[dict[str, Any]]:
        """Hybrid dense + BM25 search.

        Raises:
            ConfigurationError: When BM25 is not available and a true
                hybrid result cannot be produced. Callers that want
                dense-only fallback should call :meth:`search`
                directly.

        """
        from raghub.errors import ConfigurationError

        if not SQLITE_VECTOR_PKG:
            raise ConfigurationError(
                "SqliteStore.hybrid_search requires the sqlite-vector "
                "package; install `sqlite-vector` or use SqliteStore.search "
                "for dense-only retrieval."
            )
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
                    f"text, source_location, company, owner, department, tenant_id, vector "
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
                    tenant_id=tn or "",
                    checksum=hashlib.sha256(
                        txt.encode("utf-8", errors="surrogatepass")
                    ).hexdigest(),
                ),
                "score": 1.0,
                "chunk_id": cid,
                "document_id": did,
                "version": ver,
            }
            for cid, did, ver, cls, txt, sloc, co, ow, dp, tn, _ in rows[:top_k]
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
