"""pgvector-backed :class:`Store` adapter.

The first first-class vector-store adapter. Opt-in via
``pip install raghub``. Other backends (Qdrant, FAISS,
Chroma, Milvus) remain pluggable via :class:`raghub.plugins.Plugins`
and the entry-point ``group="raghub.vector_stores"``; no
first-class adapters ship in this release.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, cast

import asyncpg

from raghub.models import Chunk, Hit

__all__ = ["PgVectorStore"]


SCHEMA_SQL = """
CREATE EXTENSION IF NOT EXISTS vector;
CREATE TABLE IF NOT EXISTS raghub_chunks (
    id TEXT PRIMARY KEY,
    document_id TEXT NOT NULL,
    ordinal INTEGER NOT NULL,
    text TEXT NOT NULL,
    metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
    embedding VECTOR({dim}) NOT NULL,
    tenant_id TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS raghub_chunks_tenant_id
    ON raghub_chunks (tenant_id);
CREATE INDEX IF NOT EXISTS raghub_chunks_document_id
    ON raghub_chunks (document_id);
CREATE INDEX IF NOT EXISTS raghub_chunks_embedding_hnsw
    ON raghub_chunks USING hnsw (embedding vector_cosine_ops);
CREATE INDEX IF NOT EXISTS raghub_chunks_text_search
    ON raghub_chunks USING GIN (to_tsvector('english', text));
"""


class PgVectorStore:
    """Postgres + pgvector adapter implementing the ``Store`` contract."""

    DISTANCE_METRIC = "cosine"

    def __init__(
        self,
        dsn: str,
        embedding_dim: int,
        *,
        distance_metric: str = "cosine",
        index_type: str = "hnsw",
    ) -> None:
        """Initialise the store.

        Args:
            dsn: Postgres connection string (``postgresql://…``).
            embedding_dim: Vector dimensionality; matches the embedder.
            distance_metric: ``"cosine"`` (default), ``"l2"``, or
                ``"inner_product"``.
            index_type: ``"hnsw"`` (default) or ``"ivfflat"``.

        """
        self.dsn = dsn
        self.embedding_dim = embedding_dim
        self.distance_metric = distance_metric
        self.index_type = index_type

    async def initialize(self) -> None:
        """Create the schema, indexes, and pgvector extension."""
        conn = await asyncpg.connect(self.dsn)
        try:
            schema = SCHEMA_SQL.format(dim=self.embedding_dim)
            await conn.execute(schema)
            await self.set_session(conn)
        finally:
            await conn.close()

    @staticmethod
    async def set_session(conn: Any, *, user_id: str = "", tenant_id: str = "") -> None:
        """Propagate the per-request identity into the session."""
        try:
            await conn.execute(
                "SET LOCAL app.current_user_id = $1",
                user_id or "",
            )
            await conn.execute(
                "SET LOCAL app.current_tenant_id = $1",
                tenant_id or "",
            )
        except Exception:
            # The ``app.*`` settings are RLS hooks; if a deployment
            # does not configure them, this is a no-op.
            pass

    async def create_collection(self) -> None:
        """Idempotently create the collection and indexes."""
        await self.initialize()

    async def insert(
        self,
        chunks: Sequence[Chunk],
        vectors: Sequence[Sequence[float]],
    ) -> int:
        """Insert ``chunks`` with their ``vectors``."""
        if len(chunks) != len(vectors):
            raise ValueError("chunks and vectors must be parallel")
        conn = await asyncpg.connect(self.dsn)
        try:
            inserted = 0
            for chunk, vector in zip(chunks, vectors, strict=True):
                if len(vector) != self.embedding_dim:
                    raise ConfigurationError(
                        f"vector dimension mismatch: expected {self.embedding_dim}, "
                        f"got {len(vector)}"
                    )
                await conn.execute(
                    "INSERT INTO raghub_chunks "
                    "(id, document_id, ordinal, text, metadata, embedding, tenant_id) "
                    "VALUES ($1, $2, $3, $4, $5, $6::vector, $7)",
                    chunk.id,
                    chunk.document_id,
                    getattr(chunk, "ordinal", 0),
                    chunk.text,
                    jsonb_dumps(chunk.metadata),
                    format_vector(vector),
                    tenant_id(chunk),
                )
                inserted += 1
            return inserted
        finally:
            await conn.close()

    async def upsert(
        self,
        chunks: Sequence[Chunk],
        vectors: Sequence[Sequence[float]],
    ) -> int:
        """Upsert ``chunks``; returns the count."""
        if len(chunks) != len(vectors):
            raise ValueError("chunks and vectors must be parallel")
        conn = await asyncpg.connect(self.dsn)
        try:
            updated = 0
            for chunk, vector in zip(chunks, vectors, strict=True):
                await conn.execute(
                    "INSERT INTO raghub_chunks "
                    "(id, document_id, ordinal, text, metadata, embedding, tenant_id) "
                    "VALUES ($1, $2, $3, $4, $5, $6::vector, $7) "
                    "ON CONFLICT (id) DO UPDATE SET "
                    "embedding = EXCLUDED.embedding, "
                    "metadata = EXCLUDED.metadata, "
                    "updated_at = now()",
                    chunk.id,
                    chunk.document_id,
                    getattr(chunk, "ordinal", 0),
                    chunk.text,
                    jsonb_dumps(chunk.metadata),
                    format_vector(vector),
                    tenant_id(chunk),
                )
                updated += 1
            return updated
        finally:
            await conn.close()

    async def delete(self, chunk_id: str) -> None:
        """Delete one chunk."""
        conn = await asyncpg.connect(self.dsn)
        try:
            await conn.execute("DELETE FROM raghub_chunks WHERE id = $1", chunk_id)
        finally:
            await conn.close()

    async def delete_document(self, document_id: str) -> None:
        """Delete every chunk tied to ``document_id``."""
        conn = await asyncpg.connect(self.dsn)
        try:
            await conn.execute(
                "DELETE FROM raghub_chunks WHERE document_id = $1",
                document_id,
            )
        finally:
            await conn.close()

    async def search(
        self,
        query_vector: Sequence[float],
        top_k: int = 5,
        *,
        tenant_id: str | None = None,
        metadata_filter: dict[str, Any] | None = None,
    ) -> list[Hit]:
        """Dense-similarity search."""
        if len(query_vector) != self.embedding_dim:
            raise ConfigurationError(
                f"vector dimension mismatch: expected {self.embedding_dim}, got {len(query_vector)}"
            )
        conn = await asyncpg.connect(self.dsn)
        try:
            await self.set_session(conn, tenant_id=tenant_id or "")
            where = ["tenant_id IS NULL"]
            params: list[Any] = [format_vector(query_vector), int(top_k)]
            if tenant_id is not None:
                where = ["tenant_id = $2"]
                params.append(tenant_id)
            where_clause = " AND ".join(where)
            sql = (
                "SELECT id, 1 - (embedding <=> $1) AS score "
                "FROM raghub_chunks "
                f"WHERE {where_clause} "  # nosec B608 - where clause is a literal column ref; values use asyncpg $N placeholders
                "ORDER BY embedding <=> $1 "
                f"LIMIT ${len(params)}"
            )
            rows = await conn.fetch(sql, *params)
            # pgvector only persists the embedding; we don't have the full Chunk
            # available here, so emit a Hit-shaped dict and let callers rehydrate.
            # The framework's vector_store.search contract returns
            # ``dict[str, Any]`` entries; this adapter honours that.
            return cast(
                list[Hit],
                [Hit(score=float(row["score"]), chunk=row["id"]) for row in rows],
            )
        finally:
            await conn.close()

    async def search_hybrid(
        self,
        *,
        query: str,
        query_vector: Sequence[float],
        top_k: int = 5,
        tenant_id: str | None = None,
    ) -> list[Hit]:
        """Hybrid dense + Postgres FTS search, fused by RRF."""
        conn = await asyncpg.connect(self.dsn)
        try:
            await self.set_session(conn, tenant_id=tenant_id or "")
            dense_rows = await conn.fetch(
                "SELECT id, 1 - (embedding <=> $1) AS score "
                "FROM raghub_chunks "
                "ORDER BY embedding <=> $1 LIMIT $2",
                format_vector(query_vector),
                int(top_k),
            )
            fts_rows = await conn.fetch(
                "SELECT id, ts_rank_cd(text_search, plainto_tsquery('english', $1)) AS score "
                "FROM raghub_chunks "
                "ORDER BY score DESC LIMIT $2",
                query,
                int(top_k),
            )
        finally:
            await conn.close()

        k = 60
        fused: dict[str, float] = {}
        for rank, row in enumerate(dense_rows, start=1):
            fused[row["id"]] = fused.get(row["id"], 0.0) + 1.0 / (k + rank)
        for rank, row in enumerate(fts_rows, start=1):
            fused[row["id"]] = fused.get(row["id"], 0.0) + 1.0 / (k + rank)
        sorted_hits = sorted(fused.items(), key=lambda kv: kv[1], reverse=True)
        return [
            Hit(score=score, chunk=chunk_id)  # type: ignore[arg-type]
            for chunk_id, score in sorted_hits[: int(top_k)]
        ]

    async def keyword_search(
        self,
        query: str,
        top_k: int = 5,
    ) -> list[Hit]:
        """Postgres FTS-only search."""
        conn = await asyncpg.connect(self.dsn)
        try:
            rows = await conn.fetch(
                "SELECT id, ts_rank_cd(text_search, plainto_tsquery('english', $1)) AS score "
                "FROM raghub_chunks "
                "ORDER BY score DESC LIMIT $2",
                query,
                int(top_k),
            )
        finally:
            await conn.close()
        # See note above; pgvector emits Hit-shaped dicts, not Hit records.
        return cast(
            list[Hit],
            [Hit(score=float(row["score"]), chunk=row["id"]) for row in rows],
        )

    async def optimize(self) -> None:
        """VACUUM ANALYZE; rebuilds HNSW if drift is high."""
        conn = await asyncpg.connect(self.dsn)
        try:
            await conn.execute("VACUUM ANALYZE raghub_chunks")
            await conn.execute("REINDEX INDEX CONCURRENTLY raghub_chunks_embedding_hnsw")
        finally:
            await conn.close()

    async def health(self) -> dict[str, Any]:
        """Return a status dict."""
        conn = await asyncpg.connect(self.dsn)
        try:
            row = await conn.fetchrow("SELECT COUNT(*) AS n FROM raghub_chunks")
        finally:
            await conn.close()
        count = int(row["n"]) if row else 0
        return {"status": "ok", "backend": "pgvector", "chunks": count}


def format_vector(vector: Sequence[float]) -> str:
    """Format ``vector`` as the pgvector literal ``'[v1,v2,...]'``."""
    return "[" + ",".join(f"{v:.7f}" for v in vector) + "]"


def jsonb_dumps(metadata: dict[str, Any] | None) -> str:
    """Serialise metadata to a JSONB-safe JSON string."""
    import json

    return json.dumps(metadata or {}, default=str)


def tenant_id(chunk: Chunk) -> str | None:
    """Extract the tenant id from a chunk's metadata, if any."""
    return getattr(chunk, "tenant_id", None)
