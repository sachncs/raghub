"""SQLite document repository.

Implements :class:`raghub.domain.repositories.DocumentRepository`
against a SQLite database. Part of the legacy persistence layer.

Schema:
    The ``documents`` table uses a composite primary key
    ``(document_id, version)`` so the same document can have many
    historical versions stored side-by-side. A ``UNIQUE(checksum)``
    index makes checksum-based dedup race-detectable: two concurrent
    writers racing to insert the same content collide on the index
    and one of them gets :class:`aiosqlite.IntegrityError`.

Migration:
    Pre-existing databases created with the legacy single-column PK
    ``(document_id)`` are rebuilt transparently on first
    :meth:`initialize`. Rows are copied 1:1 (the ``version`` column
    already existed; if it was 0 on legacy data it is normalised to 1).

Concurrency:
    :meth:`save` is an upsert (replace by primary key). :meth:`try_insert`
    is a plain ``INSERT`` that surfaces :class:`aiosqlite.IntegrityError`
    so callers can detect concurrent duplicate writes by checksum or
    primary key.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import aiosqlite

from raghub.domain import DocumentRepository
from raghub.models import DocumentLifecycleStatus, DocumentRecord
from raghub.storage.database import DatabaseManager

MAX_INSERT_RETRIES = 3
RETRY_BASE_DELAY = 0.05

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS documents (
    document_id TEXT NOT NULL,
    version INTEGER NOT NULL,
    checksum TEXT NOT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    owner TEXT NOT NULL,
    organization TEXT NOT NULL,
    department TEXT DEFAULT '',
    tags TEXT DEFAULT '[]',
    classification TEXT DEFAULT 'internal',
    visibility TEXT DEFAULT 'organization',
    status TEXT DEFAULT 'NEW',
    filename TEXT DEFAULT '',
    file_type TEXT DEFAULT '',
    mime_type TEXT DEFAULT '',
    chunk_count INTEGER DEFAULT 0,
    chunk_ids TEXT DEFAULT '[]',
    error TEXT,
    PRIMARY KEY (document_id, version)
);
"""

UNIQUE_CHECKSUM_INDEX = (
    "CREATE UNIQUE INDEX IF NOT EXISTS ux_documents_checksum ON documents(checksum)"
)

INSERT_SQL = """
INSERT {mode} INTO documents (
    document_id, version, checksum, created_at, updated_at,
    owner, organization, department, tags, classification,
    visibility, status, filename, file_type, mime_type,
    chunk_count, chunk_ids, error
) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
"""


class SqliteDocumentRepository(DocumentRepository):
    def __init__(self, db_path: str | Path, db_manager: DatabaseManager | None = None) -> None:
        self.db_path = str(db_path)
        self.db_manager = db_manager

    async def conn(self) -> aiosqlite.Connection:
        if self.db_manager is not None:
            return self.db_manager.connection
        # ponytail: ad-hoc connections get WAL + a 5s busy timeout so
        # back-to-back calls on the same file don't surface "database is
        # locked" when the prior call's connection hasn't fully released
        # its exclusive rollback-journal.
        conn = await aiosqlite.connect(self.db_path)
        conn.row_factory = aiosqlite.Row
        await conn.execute("PRAGMA journal_mode=WAL")
        await conn.execute("PRAGMA busy_timeout=5000")
        return conn

    async def maybe_commit_close(self, conn: aiosqlite.Connection) -> None:
        # Only commit+close when we own the connection. With the shared
        # DatabaseManager the connection runs in autocommit mode so each
        # statement is already durable; the UnitOfWork still wraps
        # multi-statement work in BEGIN/COMMIT explicitly.
        if self.db_manager is None:
            await conn.commit()
            await conn.close()

    async def initialize(self) -> None:
        conn = await self.conn()
        await conn.executescript(SCHEMA_SQL)
        await conn.execute(UNIQUE_CHECKSUM_INDEX)
        await self.migrate_legacy_schema(conn)
        await self.maybe_commit_close(conn)

    async def migrate_legacy_schema(self, conn: aiosqlite.Connection) -> None:
        # ponytail: legacy DBs created before this migration had a single-column
        # PRIMARY KEY (document_id). Rebuild transparently so existing data
        # survives without manual SQL.
        cursor = await conn.execute(
            "SELECT sql FROM sqlite_master WHERE type='table' AND name='documents'"
        )
        row = await cursor.fetchone()
        if row is None:
            return
        ddl = row[0] or ""
        if "PRIMARY KEY (document_id, version)" in ddl:
            return
        # Copy every column except the old PK constraint, normalise
        # NULL/0 versions to 1 to satisfy the new NOT NULL semantics.
        await conn.executescript("""
            CREATE TABLE IF NOT EXISTS documents_new (
                document_id TEXT NOT NULL,
                version INTEGER NOT NULL,
                checksum TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                owner TEXT NOT NULL,
                organization TEXT NOT NULL,
                department TEXT DEFAULT '',
                tags TEXT DEFAULT '[]',
                classification TEXT DEFAULT 'internal',
                visibility TEXT DEFAULT 'organization',
                status TEXT DEFAULT 'NEW',
                filename TEXT DEFAULT '',
                file_type TEXT DEFAULT '',
                mime_type TEXT DEFAULT '',
                chunk_count INTEGER DEFAULT 0,
                chunk_ids TEXT DEFAULT '[]',
                error TEXT,
                PRIMARY KEY (document_id, version)
            );
            INSERT INTO documents_new (
                document_id, version, checksum, created_at, updated_at,
                owner, organization, department, tags, classification,
                visibility, status, filename, file_type, mime_type,
                chunk_count, chunk_ids, error
            )
            SELECT
                document_id,
                CASE WHEN version IS NULL OR version = 0 THEN 1 ELSE version END,
                checksum, created_at, updated_at,
                owner, organization, department, tags, classification,
                visibility, status, filename, file_type, mime_type,
                chunk_count, chunk_ids, error
            FROM documents;
            DROP TABLE documents;
            ALTER TABLE documents_new RENAME TO documents;
            CREATE UNIQUE INDEX IF NOT EXISTS ux_documents_checksum ON documents(checksum);
        """)

    def record_params(self, record: DocumentRecord) -> tuple[Any, ...]:
        return (
            record.document_id,
            record.version,
            record.checksum,
            record.created_at.isoformat()
            if hasattr(record.created_at, "isoformat")
            else record.created_at,
            record.updated_at.isoformat()
            if hasattr(record.updated_at, "isoformat")
            else record.updated_at,
            record.owner,
            record.organization,
            getattr(record, "department", ""),
            json.dumps(getattr(record, "tags", [])),
            record.classification.value,
            record.visibility.value,
            record.status.value,
            getattr(record, "filename", ""),
            getattr(record, "file_type", ""),
            getattr(record, "mime_type", ""),
            getattr(record, "chunk_count", 0),
            json.dumps(getattr(record, "chunk_ids", [])),
            getattr(record, "error", None),
        )

    async def save(self, record: DocumentRecord) -> None:
        conn = await self.conn()
        await conn.execute(INSERT_SQL.format(mode="OR REPLACE"), self.record_params(record))
        await self.maybe_commit_close(conn)

    async def try_insert(
        self,
        record: DocumentRecord,
        max_retries: int = MAX_INSERT_RETRIES,
    ) -> bool:
        last_exc: Exception | None = None
        for attempt in range(max_retries):
            conn = await self.conn()
            try:
                await conn.execute(INSERT_SQL.format(mode=""), self.record_params(record))
                await self.maybe_commit_close(conn)
                return True
            except aiosqlite.IntegrityError as exc:
                last_exc = exc
                # ponytail: rollback before close so the leaked implicit
                # transaction's write lock is released for the next retry.
                with contextlib.suppress(Exception):
                    await conn.rollback()
                if self.db_manager is None:
                    with contextlib.suppress(Exception):
                        await conn.close()
                if attempt < max_retries - 1:
                    await asyncio.sleep(RETRY_BASE_DELAY * (2**attempt))
                continue
        raise last_exc  # type: ignore[misc]

    async def get(self, document_id: str) -> DocumentRecord | None:
        return await self.get_version(document_id)

    async def get_version(
        self, document_id: str, version: int | None = None
    ) -> DocumentRecord | None:
        """Return a specific version, or the latest when ``version`` is None."""
        conn = await self.conn()
        if version is None:
            cursor = await conn.execute(
                "SELECT * FROM documents WHERE document_id = ? ORDER BY version DESC LIMIT 1",
                (document_id,),
            )
        else:
            cursor = await conn.execute(
                "SELECT * FROM documents WHERE document_id = ? AND version = ?",
                (document_id, version),
            )
        row = await cursor.fetchone()
        await self.maybe_commit_close(conn)
        if row is None:
            return None
        return self.row_to_record(row)

    async def list_versions(self, document_id: str) -> list[DocumentRecord]:
        conn = await self.conn()
        cursor = await conn.execute(
            "SELECT * FROM documents WHERE document_id = ? ORDER BY version ASC",
            (document_id,),
        )
        rows = await cursor.fetchall()
        await self.maybe_commit_close(conn)
        return [self.row_to_record(row) for row in rows]

    async def get_by_checksum(self, checksum: str) -> DocumentRecord | None:
        conn = await self.conn()
        cursor = await conn.execute(
            "SELECT * FROM documents WHERE checksum = ? ORDER BY version DESC LIMIT 1",
            (checksum,),
        )
        row = await cursor.fetchone()
        await self.maybe_commit_close(conn)
        if row is None:
            return None
        return self.row_to_record(row)

    async def delete(self, document_id: str) -> None:
        # Delete every version: the public API is "remove this document".
        # Callers that need version-scoped deletes use ``delete_version``.
        conn = await self.conn()
        await conn.execute("DELETE FROM documents WHERE document_id = ?", (document_id,))
        await self.maybe_commit_close(conn)

    async def delete_version(self, document_id: str, version: int) -> None:
        conn = await self.conn()
        await conn.execute(
            "DELETE FROM documents WHERE document_id = ? AND version = ?",
            (document_id, version),
        )
        await self.maybe_commit_close(conn)

    async def list_by_organization(self, organization: str) -> list[DocumentRecord]:
        # Latest version per (document_id) only — matches the legacy
        # single-row-per-document listing semantics. The inner GROUP BY
        # picks the max version; the outer ORDER BY applies to it.
        conn = await self.conn()
        cursor = await conn.execute(
            """
            SELECT d.* FROM documents d
            JOIN (
                SELECT document_id, MAX(version) AS max_version
                FROM documents
                WHERE organization = ?
                GROUP BY document_id
            ) latest
              ON latest.document_id = d.document_id
             AND latest.max_version = d.version
            ORDER BY d.updated_at DESC
            """,
            (organization,),
        )
        rows = await cursor.fetchall()
        await self.maybe_commit_close(conn)
        return [self.row_to_record(row) for row in rows]

    async def list_all(self) -> list[DocumentRecord]:
        conn = await self.conn()
        cursor = await conn.execute(
            """
            SELECT d.* FROM documents d
            JOIN (
                SELECT document_id, MAX(version) AS max_version
                FROM documents
                GROUP BY document_id
            ) latest
              ON latest.document_id = d.document_id
             AND latest.max_version = d.version
            ORDER BY d.updated_at DESC
            """
        )
        rows = await cursor.fetchall()
        await self.maybe_commit_close(conn)
        return [self.row_to_record(row) for row in rows]

    async def update_status(self, document_id: str, status: DocumentLifecycleStatus) -> None:
        # Mutate only the latest version — historical records stay
        # frozen at the status they had when they were superseded.
        conn = await self.conn()
        now = datetime.now(UTC).isoformat()
        await conn.execute(
            """
            UPDATE documents SET status = ?, updated_at = ?
            WHERE document_id = ?
              AND version = (SELECT MAX(version) FROM documents WHERE document_id = ?)
            """,
            (status.value, now, document_id, document_id),
        )
        await self.maybe_commit_close(conn)

    def row_to_record(self, row: aiosqlite.Row) -> DocumentRecord:
        data: dict[str, Any] = dict(row)
        data["created_at"] = datetime.fromisoformat(data["created_at"])
        data["updated_at"] = datetime.fromisoformat(data["updated_at"])
        data["tags"] = json.loads(data.get("tags", "[]"))
        data["chunk_ids"] = json.loads(data.get("chunk_ids", "[]"))
        return DocumentRecord(**data)
