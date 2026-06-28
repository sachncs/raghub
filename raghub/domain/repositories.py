from __future__ import annotations

from abc import ABC, abstractmethod


from raghub.models import ChunkRecord, DocumentRecord, SessionRecord
from raghub.storage.database import DatabaseManager


class DocumentRepository(ABC):
    @abstractmethod
    async def initialize(self) -> None:
        ...

    @abstractmethod
    async def save(self, record: DocumentRecord) -> None:
        ...

    @abstractmethod
    async def get(self, document_id: str) -> DocumentRecord | None:
        ...

    @abstractmethod
    async def get_by_checksum(self, checksum: str) -> DocumentRecord | None:
        ...

    @abstractmethod
    async def delete(self, document_id: str) -> None:
        ...

    @abstractmethod
    async def list_by_organization(self, organization: str) -> list[DocumentRecord]:
        ...

    @abstractmethod
    async def list_all(self) -> list[DocumentRecord]:
        ...


class ChunkRepository(ABC):
    @abstractmethod
    async def initialize(self) -> None:
        ...

    @abstractmethod
    async def insert(self, record: ChunkRecord, embedding: list[float]) -> None:
        ...

    @abstractmethod
    async def upsert(self, records: list[ChunkRecord],
                     embeddings: list[list[float]] | None = None) -> None:
        ...

    @abstractmethod
    async def delete_by_id(self, chunk_id: str) -> None:
        ...

    @abstractmethod
    async def delete_by_document(self, document_id: str) -> None:
        ...

    @abstractmethod
    async def search(self, vector: list[float], top_k: int,
                     metadata_filter: str = "") -> list[dict]:
        ...

    @abstractmethod
    async def optimize(self) -> None:
        ...

    @abstractmethod
    async def health(self) -> dict:
        ...


class SessionRepository(ABC):
    @abstractmethod
    async def initialize(self) -> None:
        ...

    @abstractmethod
    async def create(self, record: SessionRecord) -> None:
        ...

    @abstractmethod
    async def save(self, record: SessionRecord) -> None:
        ...

    @abstractmethod
    async def get(self, session_id: str) -> SessionRecord | None:
        ...

    @abstractmethod
    async def get_by_token(self, token: str) -> SessionRecord | None:
        ...

    @abstractmethod
    async def delete(self, session_id: str) -> None:
        ...


class UnitOfWork:
    def __init__(self, document_repo: DocumentRepository,
                 chunk_repo: ChunkRepository,
                 session_repo: SessionRepository,
                 db_manager: DatabaseManager | None = None) -> None:
        self.document_repo = document_repo
        self.chunk_repo = chunk_repo
        self.session_repo = session_repo
        self.db_manager = db_manager
        self.in_transaction = False

    async def initialize(self) -> None:
        if self.db_manager is not None:
            await self.db_manager.connect()
        await self.document_repo.initialize()
        await self.chunk_repo.initialize()
        await self.session_repo.initialize()

    async def commit(self) -> None:
        if self.in_transaction and self.db_manager is not None:
            conn = self.db_manager.connection
            await conn.commit()
            self.in_transaction = False

    async def rollback(self) -> None:
        if self.in_transaction and self.db_manager is not None:
            conn = self.db_manager.connection
            await conn.rollback()
            self.in_transaction = False

    async def __aenter__(self) -> UnitOfWork:
        if self.db_manager is not None:
            conn = self.db_manager.connection
            await conn.execute("BEGIN")
            self.in_transaction = True
        return self

    async def __aexit__(self, *args: object) -> None:
        exc_type = args[0]
        if exc_type is not None:
            await self.rollback()
        else:
            await self.commit()
