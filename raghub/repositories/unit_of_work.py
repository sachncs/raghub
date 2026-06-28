from __future__ import annotations

from raghub.domain import UnitOfWork as BaseUnitOfWork
from raghub.repositories.sqlite_document_repo import SqliteDocumentRepository
from raghub.repositories.sqlite_chunk_repo import SqliteChunkRepository
from raghub.repositories.sqlite_session_repo import SqliteSessionRepository
from raghub.storage.database import DatabaseManager
from raghub.vectorstore.base import BaseVectorStore


class UnitOfWork(BaseUnitOfWork):
    def __init__(self, db_path: str, vector_store: BaseVectorStore,
                 session_timeout: int = 3600) -> None:
        self.db_path = db_path
        self.vector_store = vector_store
        self.session_timeout = session_timeout
        self.initialized = False
        self.db_manager = DatabaseManager(db_path)

        doc_repo = SqliteDocumentRepository(db_path, db_manager=self.db_manager)
        chunk_repo = SqliteChunkRepository(vector_store)
        sess_repo = SqliteSessionRepository(db_path, session_timeout, db_manager=self.db_manager)
        super().__init__(
            document_repo=doc_repo,
            chunk_repo=chunk_repo,
            session_repo=sess_repo,
            db_manager=self.db_manager,
        )

    async def initialize(self) -> None:
        if not self.initialized:
            assert self.db_manager is not None
            await self.db_manager.connect()
            await self.document_repo.initialize()
            await self.chunk_repo.initialize()
            await self.session_repo.initialize()
            self.initialized = True
