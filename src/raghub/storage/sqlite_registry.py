from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import aiosqlite

from raghub.models import DocumentLifecycleStatus, DocumentVersion


class SqliteDocumentRegistry:
    def __init__(self, db_path: str | Path) -> None:
        self.db_path = str(db_path)

    async def initialize(self) -> None:
        async with aiosqlite.connect(self.db_path) as db:
            await db.executescript("""
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
            await db.commit()

    async def get(self, document_id: str) -> DocumentVersion | None:
        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            cursor = await db.execute(
                "SELECT * FROM documents WHERE document_id = ? ORDER BY version DESC LIMIT 1",
                (document_id,),
            )
            row = await cursor.fetchone()
            if row is None:
                return None
            return self._row_to_doc(row)

    async def get_by_checksum(self, checksum: str) -> DocumentVersion | None:
        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            cursor = await db.execute(
                "SELECT * FROM documents WHERE checksum = ? ORDER BY version DESC LIMIT 1",
                (checksum,),
            )
            row = await cursor.fetchone()
            if row is None:
                return None
            return self._row_to_doc(row)

    async def save(self, doc: DocumentVersion) -> None:
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute(
                """
                INSERT OR REPLACE INTO documents (
                    document_id, version, checksum, created_at, updated_at,
                    owner, organization, department, tags, classification,
                    visibility, status, filename, file_type, mime_type,
                    chunk_count, chunk_ids, error
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    doc.document_id,
                    doc.version,
                    doc.checksum,
                    doc.created_at.isoformat(),
                    doc.updated_at.isoformat(),
                    doc.owner,
                    doc.organization,
                    doc.department,
                    json.dumps(doc.tags),
                    doc.classification.value,
                    doc.visibility.value,
                    doc.status.value,
                    doc.filename,
                    doc.file_type,
                    doc.mime_type,
                    doc.chunk_count,
                    json.dumps(doc.chunk_ids),
                    doc.error,
                ),
            )
            await db.commit()

    async def update_status(self, document_id: str, status: DocumentLifecycleStatus) -> None:
        async with aiosqlite.connect(self.db_path) as db:
            now = datetime.now(timezone.utc).isoformat()
            await db.execute(
                "UPDATE documents SET status = ?, updated_at = ? WHERE document_id = ?",
                (status.value, now, document_id),
            )
            await db.commit()

    async def delete(self, document_id: str) -> None:
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute("DELETE FROM documents WHERE document_id = ?", (document_id,))
            await db.commit()

    async def list_by_company(self, company: str) -> list[DocumentVersion]:
        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            cursor = await db.execute(
                "SELECT * FROM documents WHERE organization = ? ORDER BY updated_at DESC",
                (company,),
            )
            rows = await cursor.fetchall()
            return [self._row_to_doc(row) for row in rows]

    async def list_all(self) -> list[DocumentVersion]:
        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            cursor = await db.execute("SELECT * FROM documents ORDER BY updated_at DESC")
            rows = await cursor.fetchall()
            return [self._row_to_doc(row) for row in rows]

    def _row_to_doc(self, row: aiosqlite.Row) -> DocumentVersion:
        data: dict[str, Any] = dict(row)
        data["created_at"] = datetime.fromisoformat(data["created_at"])
        data["updated_at"] = datetime.fromisoformat(data["updated_at"])
        data["tags"] = json.loads(data.get("tags", "[]"))
        data["chunk_ids"] = json.loads(data.get("chunk_ids", "[]"))
        return DocumentVersion.model_validate(data)
