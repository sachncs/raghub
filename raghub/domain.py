"""Legacy domain repository contracts.

The classes here predate the canonical Pydantic models in
:mod:`raghub.models`. New code should use the canonical models
directly; the active-record wrappers (ChunkRef, DocumentRef,
SessionWrap) have been removed in favour of Pydantic's
``model_copy(update=...)`` for mutations.

The :class:`UnitOfWork` and :class:`DocumentRepository` /
:class:`ChunkRepository` / :class:`SessionRepository` ABCs are the
contracts that :mod:`raghub.repositories` implements against SQLite.
New call sites should prefer the higher-level
:class:`raghub.knowledge.repository.KnowledgeRepository` interface.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Protocol

from raghub.models import (
    Chunk,
    Document,
    Session,
)

__all__ = [
    "ChunkRepository",
    "Database",
    "DocumentRepository",
    "SessionRepository",
    "UnitOfWork",
]


class DocumentRepository(ABC):
    """Persistence contract for :class:`Document`."""

    @abstractmethod
    async def get(self, document_id: str) -> Document | None:
        """Return the document with ``document_id`` or ``None``."""

    @abstractmethod
    async def list_by_organization(self, organization: str) -> list[Document]:
        """Return every document belonging to ``organization``."""

    @abstractmethod
    async def list_all(self) -> list[Document]:
        """Return every document in the repository."""

    @abstractmethod
    async def upsert(self, document: Document) -> None:
        """Insert or update ``document``."""

    @abstractmethod
    async def delete(self, document_id: str) -> None:
        """Remove the document with ``document_id``."""

    @abstractmethod
    async def list_versions(self, document_id: str) -> list[Document]:
        """Return every version of ``document_id`` in version order."""


class ChunkRepository(ABC):
    """Persistence contract for :class:`Chunk`."""

    @abstractmethod
    async def get(self, chunk_id: str) -> Chunk | None:
        """Return the chunk with ``chunk_id`` or ``None``."""

    @abstractmethod
    async def list_by_document(self, document_id: str, version: int | None = None) -> list[Chunk]:
        """Return every chunk for ``document_id`` (optionally at ``version``)."""

    @abstractmethod
    async def upsert(self, chunk: Chunk) -> None:
        """Insert or update ``chunk``."""

    @abstractmethod
    async def delete(self, chunk_id: str) -> None:
        """Remove the chunk with ``chunk_id``."""

    @abstractmethod
    async def delete_by_document(self, document_id: str) -> None:
        """Remove every chunk for ``document_id``."""

    @abstractmethod
    async def search_by_metadata(self, filters: dict[str, Any], *, limit: int = 100) -> list[Chunk]:
        """Return chunks whose metadata matches ``filters``."""


class SessionRepository(ABC):
    """Persistence contract for :class:`Session`."""

    @abstractmethod
    async def get(self, session_id: str) -> Session | None:
        """Return the session with ``session_id`` or ``None``."""

    @abstractmethod
    async def get_by_token(self, token: str) -> Session | None:
        """Return the session whose ``token`` matches or ``None``."""

    @abstractmethod
    async def upsert(self, session: Session) -> None:
        """Insert or update ``session``."""

    @abstractmethod
    async def delete(self, session_id: str) -> None:
        """Remove the session with ``session_id``."""


class Database(Protocol):
    """Structural protocol for the SQLite-backed database wrapper."""

    conn: Any | None

    def connect(self) -> None:
        """Open the underlying database connection."""

    def close(self) -> None:
        """Close the underlying database connection."""

    def connection(self) -> Any:
        """Return the live connection or raise if not connected."""


class UnitOfWork(Protocol):
    """Unit-of-Work contract that bundles the repositories."""

    document_repo: DocumentRepository
    chunk_repo: ChunkRepository
    session_repo: SessionRepository
    image_store: Any

    async def initialize(self) -> None:
        """Open connections and create tables."""

    async def close(self) -> None:
        """Close every held resource."""

    async def commit(self) -> None:
        """Commit the active transaction."""

    async def rollback(self) -> None:
        """Roll back the active transaction."""
