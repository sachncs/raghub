"""Legacy domain models and repository contracts.

The classes here predate the canonical Pydantic models in
:mod:`raghub.models`. New code should use the canonical models; the
legacy wrappers are kept so existing call sites that pass
:class:`Chunk` / :class:`Document` / :class:`Session` keep working.

The :class:`UnitOfWork` and :class:`DocumentRepository` /
:class:`ChunkRepository` / :class:`SessionRepository` ABCs are the
contracts that :mod:`raghub.repositories` implements against SQLite.
New call sites should prefer the higher-level
:class:`raghub.knowledge.repository.KnowledgeRepository` interface.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import UTC, datetime
from typing import Any, Protocol

import aiosqlite

from raghub.models import (
    ChunkRecord,
    ConversationTurn,
    DocumentLifecycleStatus,
    DocumentRecord,
    SessionRecord,
)


class Chunk:
    """Active-record wrapper around a :class:`ChunkRecord`.

    Attribute reads/writes forward to the wrapped record so callers
    can use the chunk as if it were the underlying Pydantic model.
    """

    def __init__(self, record: ChunkRecord) -> None:
        """Wrap ``record``."""
        self.record = record

    @property
    def chunk_id(self) -> str:
        """Return the chunk id from the wrapped record."""
        return self.record.chunk_id

    def __getattr__(self, name: str) -> Any:
        """Forward unknown attribute reads to the wrapped record."""
        return getattr(self.record, name)

    def __setattr__(self, name: str, value: Any) -> None:
        """Forward attribute writes to the wrapped record.

        Only ``record`` itself is stored on the wrapper; everything
        else is set on the underlying Pydantic model.
        """
        if name == "record":
            super().__setattr__(name, value)
        else:
            setattr(self.record, name, value)

    def update(self, **kwargs: Any) -> Chunk:
        """Bulk-set fields on the wrapped record.

        Args:
            **kwargs: Field name/value pairs to assign.

        Returns:
            ``self`` for chaining.
        """
        for key, value in kwargs.items():
            setattr(self.record, key, value)
        return self


class Document:
    """Active-record wrapper around a :class:`DocumentRecord`.

    Attribute reads/writes forward to the wrapped record so callers
    can use the document as if it were the underlying Pydantic model.
    """

    def __init__(self, record: DocumentRecord) -> None:
        """Wrap ``record``."""
        self.record = record

    @property
    def document_id(self) -> str:
        """Return the document id from the wrapped record."""
        return self.record.document_id

    @property
    def status(self) -> DocumentLifecycleStatus:
        """Return the current lifecycle status."""
        return self.record.status

    @status.setter
    def status(self, value: DocumentLifecycleStatus) -> None:
        """Update the lifecycle status on the wrapped record."""
        self.record.status = value

    def __getattr__(self, name: str) -> Any:
        """Forward unknown attribute reads to the wrapped record."""
        return getattr(self.record, name)

    def __setattr__(self, name: str, value: Any) -> None:
        """Forward attribute writes to the wrapped record.

        Only ``record`` itself is stored on the wrapper; everything
        else is set on the underlying Pydantic model.
        """
        if name == "record":
            super().__setattr__(name, value)
        else:
            setattr(self.record, name, value)

    def update(self, **kwargs: Any) -> Document:
        """Bulk-set fields and bump ``updated_at``.

        Args:
            **kwargs: Field name/value pairs to assign.

        Returns:
            ``self`` for chaining.
        """
        for key, value in kwargs.items():
            setattr(self.record, key, value)
        self.record.updated_at = datetime.now(UTC)
        return self

    def mark_failed(self, error: str) -> Document:
        """Mark the document as failed and record ``error``.

        Args:
            error: Human-readable failure description.

        Returns:
            ``self`` for chaining.
        """
        self.record.status = self.record.status.__class__.FAILED
        self.record.error = error
        self.record.updated_at = datetime.now(UTC)
        return self


class Session:
    """Active-record wrapper around a :class:`SessionRecord`.

    Attribute reads/writes forward to the wrapped record so callers
    can use the session as if it were the underlying Pydantic model.
    """

    def __init__(self, record: SessionRecord) -> None:
        """Wrap ``record``."""
        self.record = record

    @property
    def session_id(self) -> str:
        """Return the session id from the wrapped record."""
        return self.record.session_id

    @property
    def history(self) -> list[ConversationTurn]:
        """Return a shallow copy of the conversation history."""
        return list(self.record.history)

    def __getattr__(self, name: str) -> Any:
        """Forward unknown attribute reads to the wrapped record."""
        return getattr(self.record, name)

    def __setattr__(self, name: str, value: Any) -> None:
        """Forward attribute writes to the wrapped record.

        Only ``record`` itself is stored on the wrapper; everything
        else is set on the underlying Pydantic model.
        """
        if name == "record":
            super().__setattr__(name, value)
        else:
            setattr(self.record, name, value)

    def add_turn(self, question: str, answer: str, **kwargs: Any) -> Session:
        """Append a new conversation turn and bump ``last_seen_at``.

        Args:
            question: The user's question text.
            answer: The assistant's answer text.
            **kwargs: Extra fields forwarded to
                :class:`ConversationTurn` (e.g. ``metadata``).

        Returns:
            ``self`` for chaining.
        """
        turn = ConversationTurn(question=question, answer=answer, **kwargs)
        self.record.history.append(turn)
        self.record.last_seen_at = datetime.now(UTC)
        return self

    def clear(self) -> Session:
        """Empty the conversation history and bump ``last_seen_at``.

        Returns:
            ``self`` for chaining.
        """
        self.record.history.clear()
        self.record.last_seen_at = datetime.now(UTC)
        return self


class DocumentRepository(ABC):
    """Persistence contract for :class:`DocumentRecord`."""

    @abstractmethod
    async def initialize(self) -> None:
        """Bring the underlying schema/connection online."""

    @abstractmethod
    async def save(self, record: DocumentRecord) -> None:
        """Persist ``record`` (insert or update)."""

    @abstractmethod
    async def get(self, document_id: str) -> DocumentRecord | None:
        """Return the document with ``document_id`` or ``None``."""

    @abstractmethod
    async def get_by_checksum(self, checksum: str) -> DocumentRecord | None:
        """Return the document matching ``checksum`` or ``None``."""

    @abstractmethod
    async def delete(self, document_id: str) -> None:
        """Remove the document and any dependent rows."""

    @abstractmethod
    async def list_by_organization(self, organization: str) -> list[DocumentRecord]:
        """Return every document in ``organization``."""

    @abstractmethod
    async def list_all(self) -> list[DocumentRecord]:
        """Return every persisted document."""

    @abstractmethod
    async def try_insert(self, record: DocumentRecord, max_retries: int = 1) -> bool:
        """Best-effort insert that survives transient conflicts.

        Args:
            record: The document to insert.
            max_retries: Number of times to retry on contention.

        Returns:
            ``True`` when the row was inserted, ``False`` otherwise.
        """


class ChunkRepository(ABC):
    """Persistence contract for :class:`ChunkRecord` and embeddings."""

    @abstractmethod
    async def initialize(self) -> None:
        """Bring the underlying schema/connection online."""

    @abstractmethod
    async def insert(self, record: ChunkRecord, embedding: list[float]) -> None:
        """Persist ``record`` with its ``embedding`` vector."""

    @abstractmethod
    async def upsert(
        self, records: list[ChunkRecord], embeddings: list[list[float]] | None = None
    ) -> None:
        """Insert or update ``records``; ``embeddings[i]`` aligns with ``records[i]``."""

    @abstractmethod
    async def delete_by_id(self, chunk_id: str) -> None:
        """Remove the chunk with ``chunk_id``."""

    @abstractmethod
    async def delete_by_document(self, document_id: str) -> None:
        """Remove every chunk belonging to ``document_id``."""

    @abstractmethod
    async def search(
        self, vector: list[float], top_k: int, metadata_filter: str = ""
    ) -> list[dict]:
        """Vector search with an optional RBAC metadata filter."""

    @abstractmethod
    async def optimize(self) -> None:
        """Compact/optimize the underlying indexes."""

    @abstractmethod
    async def health(self) -> dict:
        """Return a backend-native health snapshot."""


class SessionRepository(ABC):
    """Persistence contract for :class:`SessionRecord`."""

    @abstractmethod
    async def initialize(self) -> None:
        """Bring the underlying schema/connection online."""

    @abstractmethod
    async def create(self, record: SessionRecord) -> None:
        """Persist a freshly-created session record."""

    @abstractmethod
    async def save(self, record: SessionRecord) -> None:
        """Persist an updated session record."""

    @abstractmethod
    async def get(self, session_id: str) -> SessionRecord | None:
        """Return the session with ``session_id`` or ``None``."""

    @abstractmethod
    async def get_by_token(self, token: str) -> SessionRecord | None:
        """Return the session whose token matches ``token`` or ``None``."""

    @abstractmethod
    async def delete(self, session_id: str) -> None:
        """Remove the session and any dependent rows."""


class UnitOfWork:
    """Bundle the three repositories behind a single transaction handle.

    Wraps :class:`DocumentRepository`, :class:`ChunkRepository`, and
    :class:`SessionRepository` so callers can perform cross-aggregate
    writes atomically when a shared :class:`DatabaseManager` is in
    use. The unit also doubles as an ``async with`` context manager.
    """

    def __init__(
        self,
        document_repo: DocumentRepository,
        chunk_repo: ChunkRepository,
        session_repo: SessionRepository,
        db_manager: DatabaseManager | None = None,
    ) -> None:
        """Store the per-repository handles and optional DB manager."""
        self.document_repo = document_repo
        self.chunk_repo = chunk_repo
        self.session_repo = session_repo
        self.db_manager = db_manager
        self.in_transaction = False

    async def initialize(self) -> None:
        """Connect the DB manager (when present) and initialise every repo."""
        if self.db_manager is not None:
            await self.db_manager.connect()
        await self.document_repo.initialize()
        await self.chunk_repo.initialize()
        await self.session_repo.initialize()

    async def commit(self) -> None:
        """Commit the active transaction when one is in flight."""
        if self.in_transaction and self.db_manager is not None:
            conn = self.db_manager.connection
            await conn.commit()
            self.in_transaction = False

    async def rollback(self) -> None:
        """Rollback the active transaction when one is in flight."""
        if self.in_transaction and self.db_manager is not None:
            conn = self.db_manager.connection
            await conn.rollback()
            self.in_transaction = False

    async def __aenter__(self) -> UnitOfWork:
        """Open a transaction when a DB manager is wired in."""
        if self.db_manager is not None:
            conn = self.db_manager.connection
            await conn.execute("BEGIN")
            self.in_transaction = True
        return self

    async def __aexit__(self, *args: object) -> None:
        """Commit on clean exit, rollback on exception."""
        exc_type = args[0]
        if exc_type is not None:
            await self.rollback()
        else:
            await self.commit()

    async def close(self) -> None:
        """Close the shared DB manager (if any)."""
        if self.db_manager is not None:
            await self.db_manager.close()


class DatabaseManager(Protocol):
    async def connect(self) -> aiosqlite.Connection: ...

    @property
    def connection(self) -> aiosqlite.Connection: ...

    async def close(self) -> None: ...
