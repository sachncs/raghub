"""SQLite-backed repository implementations.

Concrete implementations of the storage repository protocols
defined in :mod:`raghub.domain`. The four classes ship in a single
file because they share the SQLite persistence concern and are
always wired together by :class:`UnitOfWork`:

* :class:`ChunkStore` — chunk + embedding persistence
  through a vector store.
* :class:`DocStore` — versioned document rows.
* :class:`SessionStore` — session rows.
* :class:`UnitOfWork` — the transaction coordinator that ties them
  to a single :class:`DatabaseManager`.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import aiosqlite

from raghub.domain import (
    ChunkRepository,
    DatabaseManager,
    DocumentRepository,
    SessionRepository,
)
from raghub.domain import (
    UnitOfWork as BaseUnitOfWork,
)
from raghub.models import (
    ChunkRecord,
    DocumentLifecycleStatus,
    DocumentRecord,
    SessionRecord,
)
from raghub.store import Store
from raghub.stores import Database, Sessions

__all__ = [
    "ChunkStore",
    "DocStore",
    "UnitOfWork",
]

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


class ChunkStore(ChunkRepository):
    """Store chunk records and embeddings in a vector store."""

    def __init__(self, vector_store: Store) -> None:
        """Store ``vector_store`` for chunk persistence."""
        self.store = vector_store

    async def initialize(self) -> None:
        """Bring the underlying vector collection online."""
        self.store.create_collection()

    async def insert(self, record: ChunkRecord, embedding: list[float]) -> None:
        """Insert ``record`` with ``embedding`` into the vector store."""
        self.store.insert([record], [embedding])

    async def upsert(
        self, records: list[ChunkRecord], embeddings: list[list[float]] | None = None
    ) -> None:
        """Insert or update each ``record`` with its matching embedding."""
        if embeddings is None:
            raise ValueError("embeddings required for upsert")
        self.store.upsert(records, embeddings)

    async def delete_by_id(self, chunk_id: str) -> None:
        """Delete the chunk with ``chunk_id``."""
        self.store.delete([chunk_id])

    async def delete_by_document(self, document_id: str) -> None:
        """Delete every chunk associated with ``document_id``."""
        self.store.delete_document(document_id)

    async def search(
        self, vector: list[float], top_k: int, metadata_filter: str = ""
    ) -> list[dict[str, Any]]:
        """Return the top-k hits most similar to ``vector``."""
        return self.store.search(vector=vector, top_k=top_k, metadata_filter=metadata_filter)

    async def optimize(self) -> None:
        """Trigger an optimisation pass on the underlying vector store."""
        self.store.optimize()

    async def health(self) -> dict[str, Any]:
        """Return the health snapshot of the underlying vector store."""
        return self.store.health()


class DocStore(DocumentRepository):
    """Persist versioned documents in SQLite.

    Schema:
        The ``documents`` table uses a composite primary key
        ``(document_id, version)`` so the same document can have many
        historical versions stored side-by-side. A ``UNIQUE(checksum)``
        index makes checksum-based dedup race-detectable.

    Migration:
        Pre-existing databases created with a single-column primary key
        ``(document_id)`` are rebuilt transparently on first
        :meth:`initialize`.
    """

    def __init__(self, db_path: str | Path, db_manager: DatabaseManager | None = None) -> None:
        """Store ``db_path`` for the SQLite database."""
        self.db_path = str(db_path)
        self.db_manager = db_manager

    async def conn(self) -> aiosqlite.Connection:
        """Return a configured aiosqlite connection."""
        if self.db_manager is not None:
            return self.db_manager.connection
        conn = await aiosqlite.connect(self.db_path)
        conn.row_factory = aiosqlite.Row
        await conn.execute("PRAGMA journal_mode=WAL")
        await conn.execute("PRAGMA busy_timeout=5000")
        return conn

    async def maybe_commit_close(self, conn: aiosqlite.Connection) -> None:
        """Commit and close ``conn`` when not managed by ``db_manager``."""
        if self.db_manager is None:
            await conn.commit()
            await conn.close()

    async def initialize(self) -> None:
        """Create the schema, indexes, and apply any pending migrations."""
        conn = await self.conn()
        await conn.executescript(SCHEMA_SQL)
        await conn.execute(UNIQUE_CHECKSUM_INDEX)
        await self.migrate_schema(conn)
        await self.maybe_commit_close(conn)

    async def migrate_schema(self, conn: aiosqlite.Connection) -> None:
        """Rebuild the single-column documents table in place."""
        cursor = await conn.execute(
            "SELECT sql FROM sqlite_master WHERE type='table' AND name='documents'"
        )
        row = await cursor.fetchone()
        if row is None:
            return
        ddl = row[0] or ""
        if "PRIMARY KEY (document_id, version)" in ddl:
            return
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
        """Serialise ``record`` to the SQL bind-tuple shape."""
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
        """Insert or update ``record`` in the documents table."""
        conn = await self.conn()
        await conn.execute(INSERT_SQL.format(mode="OR REPLACE"), self.record_params(record))
        await self.maybe_commit_close(conn)

    async def try_insert(
        self,
        record: DocumentRecord,
        max_retries: int = MAX_INSERT_RETRIES,
    ) -> bool:
        """Insert ``record`` without raising on conflicts."""
        conn = await self.conn()
        await conn.execute(INSERT_SQL.format(mode=""), self.record_params(record))
        await self.maybe_commit_close(conn)
        return True

    async def get(self, document_id: str) -> DocumentRecord | None:
        """Return the latest version record for ``document_id``."""
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
        """Return every historical version of ``document_id``."""
        conn = await self.conn()
        cursor = await conn.execute(
            "SELECT * FROM documents WHERE document_id = ? ORDER BY version ASC",
            (document_id,),
        )
        rows = await cursor.fetchall()
        await self.maybe_commit_close(conn)
        return [self.row_to_record(row) for row in rows]

    async def get_by_checksum(self, checksum: str) -> DocumentRecord | None:
        """Return the latest record matching ``checksum``."""
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
        """Delete every version of ``document_id``."""
        conn = await self.conn()
        await conn.execute("DELETE FROM documents WHERE document_id = ?", (document_id,))
        await self.maybe_commit_close(conn)

    async def delete_version(self, document_id: str, version: int) -> None:
        """Delete the specific version of ``document_id``."""
        conn = await self.conn()
        await conn.execute(
            "DELETE FROM documents WHERE document_id = ? AND version = ?",
            (document_id, version),
        )
        await self.maybe_commit_close(conn)

    async def list_by_organization(self, organization: str) -> list[DocumentRecord]:
        """Return the latest version of every document in ``organization``."""
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
        """Return the latest version of every document."""
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
        """Update the lifecycle status of the latest version."""
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
        """Convert an aiosqlite row into a :class:`DocumentRecord`."""
        data: dict[str, Any] = dict(row)
        data["created_at"] = datetime.fromisoformat(data["created_at"])
        data["updated_at"] = datetime.fromisoformat(data["updated_at"])
        data["tags"] = json.loads(data.get("tags", "[]"))
        data["chunk_ids"] = json.loads(data.get("chunk_ids", "[]"))
        return DocumentRecord(**data)


class SessionStore(SessionRepository):
    """Adapt the SQLite session store to the session repository interface."""

    def __init__(
        self,
        db_path: str | Path,
        timeout_seconds: int = 3600,
        db_manager: DatabaseManager | None = None,
    ) -> None:
        self.inner = Sessions(db_path, timeout_seconds, db=db_manager)
        self.db_manager = db_manager

    async def initialize(self) -> None:
        """Initialise the underlying session store."""
        await self.inner.initialize()

    async def create(self, record: SessionRecord) -> None:
        """Insert a new session record."""
        conn = await self.conn()
        await conn.execute(
            """
            INSERT INTO sessions (session_id, user_id, token, created_at, expires_at, last_seen_at, history)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                record.session_id,
                record.user_id,
                record.token,
                record.created_at.isoformat(),
                record.expires_at.isoformat(),
                record.last_seen_at.isoformat(),
                "[]",
            ),
        )

    async def create_from_record(self, record: SessionRecord) -> None:
        """Insert a full :class:`SessionRecord` including its history.

        The default :meth:`create` writes an empty history column
        (fresh sessions have empty history). The migration path
        and any other consumer that already has history to persist
        uses this method to insert a complete row.
        """
        conn = await self.conn()
        await conn.execute(
            """
            INSERT INTO sessions (
                session_id, user_id, token,
                created_at, expires_at, last_seen_at,
                history, overrides
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                record.session_id,
                record.user_id,
                record.token,
                record.created_at.isoformat(),
                record.expires_at.isoformat(),
                record.last_seen_at.isoformat(),
                json.dumps([t.model_dump(mode="json") for t in record.history]),
                json.dumps(record.overrides or {}),
            ),
        )
        if self.db_manager is None:
            await conn.commit()
            await conn.close()

    async def save(self, record: SessionRecord) -> None:
        """Persist updates for an existing session."""
        await self.inner.update_session(record)

    async def get(self, session_id: str) -> SessionRecord | None:
        """Return the session with ``session_id`` or ``None``."""
        return await self.inner.get_session(session_id)

    async def get_by_token(self, token: str) -> SessionRecord | None:
        """Return the session holding ``token`` or ``None``."""
        return await self.inner.get_by_token(token)

    async def delete(self, session_id: str) -> None:
        """Delete the session with ``session_id``."""
        await self.inner.delete_session(session_id)

    async def conn(self) -> aiosqlite.Connection:
        """Return a configured aiosqlite connection for sessions."""
        if self.db_manager is not None:
            return self.db_manager.connection
        conn = await aiosqlite.connect(self.inner.db_path)
        conn.row_factory = aiosqlite.Row
        return conn


class UnitOfWork(BaseUnitOfWork):
    """Coordinate repositories over a shared SQLite transaction."""

    def __init__(self, db_path: str, vector_store: Store, session_timeout: int = 3600) -> None:
        """Bind the three collaborators behind one transaction boundary."""
        self.db_path = db_path
        self.vector_store = vector_store
        self.session_timeout = session_timeout
        self.initialized = False
        self.db_manager = Database(db_path)

        doc_repo = DocStore(db_path, db_manager=self.db_manager)
        chunk_repo = ChunkStore(vector_store)
        sess_repo = SessionStore(db_path, session_timeout, db_manager=self.db_manager)
        super().__init__(
            document_repo=doc_repo,
            chunk_repo=chunk_repo,
            session_repo=sess_repo,
            db_manager=self.db_manager,
        )

    async def initialize(self) -> None:
        """Open the database connection and initialise all repositories."""
        if not self.initialized:
            assert self.db_manager is not None
            await self.db_manager.connect()
            await self.document_repo.initialize()
            await self.chunk_repo.initialize()
            await self.session_repo.initialize()
            self.initialized = True

    async def close(self) -> None:
        """Close the underlying :class:`DatabaseManager`.

        Idempotent: a second call is a no-op.
        """
        if not self.initialized:
            return
        db_manager = self.db_manager
        if db_manager is not None:
            await db_manager.close()
        self.initialized = False

    async def __aenter__(self) -> UnitOfWork:
        """Enter the unit-of-work as an async context manager."""
        await self.initialize()
        await super().__aenter__()
        return self

    async def __aexit__(self, *args: object) -> None:
        """Exit the unit-of-work as an async context manager."""
        await super().__aexit__(*args)
        await self.close()
