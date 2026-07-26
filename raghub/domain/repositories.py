"""Legacy repository protocols for domain persistence.

Defines the abstract interfaces (:class:`DocumentRepository`,
:class:`ChunkRepository`, :class:`SessionRepository`,
:class:`UnitOfWork`) that the ``raghub.repositories`` package
implements against SQLite. New call sites should prefer the
higher-level :class:`raghub.knowledge.repository.KnowledgeRepository`
interface.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from raghub.models import ChunkRecord, DocumentRecord, SessionRecord


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


__all__ = [
    "ChunkRepository",
    "DocumentRepository",
    "SessionRepository",
    "UnitOfWork",
]