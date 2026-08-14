"""Domain repository contracts.

These classes are polymorphic base classes (via the
:class:`Registry` mixin) for the concrete SQLite-backed repositories
in :mod:`raghub.repos`. New code should prefer the higher-level
:class:`raghub.knowledge.repository.KnowledgeRepository` interface.
"""

from __future__ import annotations

from typing import Any

from raghub.models import (
    Chunk,
    Document,
    Session,
)
from raghub.registry import Registry

__all__ = [
    "ChunkRepository",
    "Database",
    "DocumentRepository",
    "SessionRepository",
    "UnitOfWork",
]


class DocumentRepository(Registry):
    """Polymorphic base for :class:`Document` persistence."""

    async def get(self, document_id: str) -> Document | None:
        """Return the document with ``document_id`` or ``None``."""
        raise NotImplementedError

    async def list_by_organization(self, organization: str) -> list[Document]:
        """Return every document belonging to ``organization``."""
        raise NotImplementedError

    async def list_all(self) -> list[Document]:
        """Return every document in the repository."""
        raise NotImplementedError

    async def upsert(self, document: Document) -> None:
        """Insert or update ``document``."""
        raise NotImplementedError

    async def delete(self, document_id: str) -> None:
        """Remove the document with ``document_id``."""
        raise NotImplementedError

    async def list_versions(self, document_id: str) -> list[Document]:
        """Return every version of ``document_id`` in version order."""
        raise NotImplementedError


class ChunkRepository(Registry):
    """Polymorphic base for :class:`Chunk` persistence."""

    async def get(self, chunk_id: str) -> Chunk | None:
        """Return the chunk with ``chunk_id`` or ``None``."""
        raise NotImplementedError

    async def list_by_document(
        self, document_id: str, version: int | None = None
    ) -> list[Chunk]:
        """Return every chunk for ``document_id`` (optionally at ``version``)."""
        raise NotImplementedError

    async def upsert(self, chunk: Chunk) -> None:
        """Insert or update ``chunk``."""
        raise NotImplementedError

    async def delete(self, chunk_id: str) -> None:
        """Remove the chunk with ``chunk_id``."""
        raise NotImplementedError

    async def delete_by_document(self, document_id: str) -> None:
        """Remove every chunk for ``document_id``."""
        raise NotImplementedError

    async def search_by_metadata(
        self, filters: dict[str, Any], *, limit: int = 100
    ) -> list[Chunk]:
        """Return chunks whose metadata matches ``filters``."""
        raise NotImplementedError


class SessionRepository(Registry):
    """Polymorphic base for :class:`Session` persistence."""

    async def get(self, session_id: str) -> Session | None:
        """Return the session with ``session_id`` or ``None``."""
        raise NotImplementedError

    async def get_by_token(self, token: str) -> Session | None:
        """Return the session whose ``token`` matches or ``None``."""
        raise NotImplementedError

    async def upsert(self, session: Session) -> None:
        """Insert or update ``session``."""
        raise NotImplementedError

    async def delete(self, session_id: str) -> None:
        """Remove the session with ``session_id``."""
        raise NotImplementedError


class Database(Registry):
    """Polymorphic base for the SQLite-backed database wrapper."""

    name: str = "database"
    conn: Any | None = None

    def connect(self) -> None:
        """Open the underlying database connection."""
        raise NotImplementedError

    def close(self) -> None:
        """Close the underlying database connection."""
        raise NotImplementedError

    def connection(self) -> Any:
        """Return the live connection or raise if not connected."""
        raise NotImplementedError


class UnitOfWork(Registry):
    """Polymorphic base for the unit-of-work contract."""

    document_repo: DocumentRepository
    chunk_repo: ChunkRepository
    session_repo: SessionRepository
    image_store: Any

    async def initialize(self) -> None:
        """Open connections and create tables."""
        raise NotImplementedError

    async def close(self) -> None:
        """Close every held resource."""
        raise NotImplementedError

    async def commit(self) -> None:
        """Commit the active transaction."""
        raise NotImplementedError

    async def rollback(self) -> None:
        """Roll back the active transaction."""
        raise NotImplementedError
