"""SQLite document repository.

Implements :class:`raghub.domain.repositories.DocumentRepository`
against a SQLite database. Part of the legacy persistence layer.

Concurrency:
    The repository uses ``INSERT OR REPLACE`` for upsert semantics,
    which means duplicate primary-key writes never raise
    :class:`IntegrityError`. However, callers that want to detect
    concurrent duplicate inserts (same checksum) should use
    :meth:`try_insert` which performs a plain ``INSERT`` and
    surfaces the exception.
"""

from __future__ import annotations

import asyncio
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import aiosqlite

from raghub.domain import DocumentRepository
from raghub.models import DocumentLifecycleStatus, DocumentRecord
from raghub.storage.database import DatabaseManager

#: Maximum retries for concurrent insert conflicts.
_MAX_INSERT_RETRIES = 3
#: Base delay (seconds) for exponential backoff between retries.
_RETRY_BASE_DELAY = 0.05


class SqliteDocumentRepository(DocumentRepository):
    def __init__(self, db_path: str | Path, db_manager: DatabaseManager | None = None) -> None:
        self.db_path = str(db_path)
        self.db_manager = db_manager

    async def conn(self) -> aiosqlite.Connection:
        if self.db_manager is not None:
            return self.db_manager.connection
        conn = await aiosqlite.connect(self.db_path)
        conn.row_factory = aiosqlite.Row
        return conn

    async def maybe_commit_close(self, conn: aiosqlite.Connection) -> None:
        if self.db_manager is None:
            await conn.commit()
            await conn.close()

    async def initialize(self) -> None:
        conn = await self.conn()
        await conn.executescript("""
            CREATE TABLE IF NOT EXISTS documents (
                document_id TEXT PRIMARY KEY,
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
                error TEXT
            );
        """)
        if self.db_manager is None:
            await conn.commit()
            await conn.close()

    async def save(self, record: DocumentRecord) -> None:
        conn = await self.conn()
        await conn.execute(
            """
            INSERT OR REPLACE INTO documents (
                document_id, version, checksum, created_at, updated_at,
                owner, organization, department, tags, classification,
                visibility, status, filename, file_type, mime_type,
                chunk_count, chunk_ids, error
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                record.document_id,
                record.version,
                record.checksum,
                record.created_at.isoformat() if hasattr(record.created_at, 'isoformat') else record.created_at,
                record.updated_at.isoformat() if hasattr(record.updated_at, 'isoformat') else record.updated_at,
                record.owner,
                record.organization,
                getattr(record, 'department', ''),
                json.dumps(getattr(record, 'tags', [])),
                record.classification.value,
                record.visibility.value,
                record.status.value,
                getattr(record, 'filename', ''),
                getattr(record, 'file_type', ''),
                getattr(record, 'mime_type', ''),
                getattr(record, 'chunk_count', 0),
                json.dumps(getattr(record, 'chunk_ids', [])),
                getattr(record, 'error', None),
            ),
        )
        await self.maybe_commit_close(conn)

    async def try_insert(
        self,
        record: DocumentRecord,
        max_retries: int = _MAX_INSERT_RETRIES,
    ) -> bool:
        """Insert ``record`` with a plain ``INSERT`` (no replacement).

        Unlike :meth:`save` (which uses ``INSERT OR REPLACE``), this
        method raises :class:`aiosqlite.IntegrityError` when a row with
        the same primary key already exists. Callers use this to detect
        concurrent duplicate-write races.

        The method retries up to ``max_retries`` times with exponential
        backoff when the conflict is transient (WAL busy, etc.).

        Args:
            record: The document record to insert.
            max_retries: Number of retry attempts before giving up.

        Returns:
            ``True`` when the insert succeeded.

        Raises:
            aiosqlite.IntegrityError: When the primary key already
                exists after all retries.
        """
        last_exc: Exception | None = None
        for attempt in range(max_retries):
            try:
                conn = await self.conn()
                await conn.execute(
                    """
                    INSERT INTO documents (
                        document_id, version, checksum, created_at, updated_at,
                        owner, organization, department, tags, classification,
                        visibility, status, filename, file_type, mime_type,
                        chunk_count, chunk_ids, error
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        record.document_id,
                        record.version,
                        record.checksum,
                        record.created_at.isoformat() if hasattr(record.created_at, 'isoformat') else record.created_at,
                        record.updated_at.isoformat() if hasattr(record.updated_at, 'isoformat') else record.updated_at,
                        record.owner,
                        record.organization,
                        getattr(record, 'department', ''),
                        json.dumps(getattr(record, 'tags', [])),
                        record.classification.value,
                        record.visibility.value,
                        record.status.value,
                        getattr(record, 'filename', ''),
                        getattr(record, 'file_type', ''),
                        getattr(record, 'mime_type', ''),
                        getattr(record, 'chunk_count', 0),
                        json.dumps(getattr(record, 'chunk_ids', [])),
                        getattr(record, 'error', None),
                    ),
                )
                await self.maybe_commit_close(conn)
                return True
            except aiosqlite.IntegrityError as exc:
                last_exc = exc
                if attempt < max_retries - 1:
                    await asyncio.sleep(_RETRY_BASE_DELAY * (2 ** attempt))
                continue
        raise last_exc  # type: ignore[misc]

    async def get(self, document_id: str) -> DocumentRecord | None:
        conn = await self.conn()
        cursor = await conn.execute(
            "SELECT * FROM documents WHERE document_id = ? ORDER BY version DESC LIMIT 1",
            (document_id,),
        )
        row = await cursor.fetchone()
        await self.maybe_commit_close(conn)
        if row is None:
            return None
        return self.row_to_record(row)

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
        conn = await self.conn()
        await conn.execute("DELETE FROM documents WHERE document_id = ?", (document_id,))
        await self.maybe_commit_close(conn)

    async def list_by_organization(self, organization: str) -> list[DocumentRecord]:
        conn = await self.conn()
        cursor = await conn.execute(
            "SELECT * FROM documents WHERE organization = ? ORDER BY updated_at DESC",
            (organization,),
        )
        rows = await cursor.fetchall()
        await self.maybe_commit_close(conn)
        return [self.row_to_record(row) for row in rows]

    async def list_all(self) -> list[DocumentRecord]:
        conn = await self.conn()
        cursor = await conn.execute("SELECT * FROM documents ORDER BY updated_at DESC")
        rows = await cursor.fetchall()
        await self.maybe_commit_close(conn)
        return [self.row_to_record(row) for row in rows]

    async def update_status(self, document_id: str, status: DocumentLifecycleStatus) -> None:
        conn = await self.conn()
        now = datetime.now(timezone.utc).isoformat()
        await conn.execute(
            "UPDATE documents SET status = ?, updated_at = ? WHERE document_id = ?",
            (status.value, now, document_id),
        )
        await self.maybe_commit_close(conn)

    def row_to_record(self, row: aiosqlite.Row) -> DocumentRecord:
        data: dict[str, Any] = dict(row)
        data["created_at"] = datetime.fromisoformat(data["created_at"])
        data["updated_at"] = datetime.fromisoformat(data["updated_at"])
        data["tags"] = json.loads(data.get("tags", "[]"))
        data["chunk_ids"] = json.loads(data.get("chunk_ids", "[]"))
        return DocumentRecord(**data)
