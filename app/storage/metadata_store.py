"""SQLite metadata store.

This module owns all SQLite access for documents, chunks, and conversations.
"""

from __future__ import annotations

from collections.abc import Iterable
import logging
import sqlite3
from pathlib import Path

from app.models.schemas import ChunkRecord, ConversationEntry, DocumentRecord


LOGGER = logging.getLogger(__name__)


class MetadataStore:
    """SQLite-backed document, chunk, and conversation store."""

    def __init__(self, sqlite_path: Path) -> None:
        self._sqlite_path = sqlite_path
        self._sqlite_path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self._sqlite_path)
        connection.row_factory = sqlite3.Row
        return connection

    def _initialize(self) -> None:
        with self._connect() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS documents (
                    id TEXT PRIMARY KEY,
                    company TEXT NOT NULL,
                    title TEXT NOT NULL,
                    path TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS chunks (
                    id TEXT PRIMARY KEY,
                    document_id TEXT NOT NULL,
                    company TEXT NOT NULL,
                    page INTEGER NOT NULL,
                    text TEXT NOT NULL,
                    FOREIGN KEY(document_id) REFERENCES documents(id)
                );
                CREATE TABLE IF NOT EXISTS conversations (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user TEXT NOT NULL,
                    session TEXT NOT NULL,
                    role TEXT NOT NULL,
                    message TEXT NOT NULL,
                    timestamp TEXT NOT NULL
                );
                """
            )

    def add_document(self, document: DocumentRecord) -> None:
        """Insert or replace a document row."""

        with self._connect() as connection:
            connection.execute(
                "INSERT OR REPLACE INTO documents (id, company, title, path) VALUES (?, ?, ?, ?)",
                (document.id, document.company, document.title, document.path),
            )

    def add_chunks(self, chunks: Iterable[ChunkRecord]) -> None:
        """Insert chunk rows."""

        with self._connect() as connection:
            connection.executemany(
                "INSERT OR REPLACE INTO chunks (id, document_id, company, page, text) VALUES (?, ?, ?, ?, ?)",
                [(chunk.id, chunk.document_id, chunk.company, chunk.page, chunk.text) for chunk in chunks],
            )

    def add_conversation(self, entry: ConversationEntry) -> None:
        """Insert a conversation turn."""

        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO conversations (user, session, role, message, timestamp)
                VALUES (?, ?, ?, ?, ?)
                """,
                (entry.user, entry.session, entry.role, entry.message, entry.timestamp.isoformat()),
            )

    def get_conversation(self, user: str, session: str) -> list[ConversationEntry]:
        """Return conversation history for a user session."""

        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT user, session, role, message, timestamp
                FROM conversations
                WHERE user = ? AND session = ?
                ORDER BY id ASC
                """,
                (user, session),
            ).fetchall()
        return [
            ConversationEntry.model_validate(
                {
                    "user": row["user"],
                    "session": row["session"],
                    "role": row["role"],
                    "message": row["message"],
                    "timestamp": row["timestamp"],
                }
            )
            for row in rows
        ]

    def clear_session(self, user: str, session: str) -> None:
        """Delete all conversation turns for a session."""

        with self._connect() as connection:
            connection.execute("DELETE FROM conversations WHERE user = ? AND session = ?", (user, session))

    def get_chunks_for_companies(self, companies: list[str]) -> list[ChunkRecord]:
        """Return all chunks that belong to the supplied companies."""

        if not companies:
            return []
        placeholders = ", ".join("?" for _ in companies)
        query = f"SELECT id, document_id, company, page, text FROM chunks WHERE company IN ({placeholders})"
        with self._connect() as connection:
            rows = connection.execute(query, companies).fetchall()
        return [
            ChunkRecord(
                id=row["id"],
                document_id=row["document_id"],
                company=row["company"],
                page=int(row["page"]),
                text=row["text"],
            )
            for row in rows
        ]

    def get_document(self, document_id: str) -> DocumentRecord | None:
        """Fetch a document by identifier."""

        with self._connect() as connection:
            row = connection.execute(
                "SELECT id, company, title, path FROM documents WHERE id = ?",
                (document_id,),
            ).fetchone()
        if row is None:
            return None
        return DocumentRecord(
            id=row["id"],
            company=row["company"],
            title=row["title"],
            path=row["path"],
        )

    def list_documents(self, companies: list[str]) -> list[DocumentRecord]:
        """List documents accessible to a user."""

        if not companies:
            return []
        placeholders = ", ".join("?" for _ in companies)
        query = f"SELECT id, company, title, path FROM documents WHERE company IN ({placeholders})"
        with self._connect() as connection:
            rows = connection.execute(query, companies).fetchall()
        return [
            DocumentRecord(id=row["id"], company=row["company"], title=row["title"], path=row["path"])
            for row in rows
        ]
