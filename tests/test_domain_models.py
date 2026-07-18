from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock

import pytest

from raghub.models import (
    ChunkRecord,
    ConversationTurn,
    DocumentLifecycleStatus,
    DocumentRecord,
    SessionRecord,
)
from raghub.storage.database import DatabaseManager

from raghub.domain.chunk import Chunk
from raghub.domain.document import Document
from raghub.domain.session import Session
from raghub.domain.repositories import (
    ChunkRepository,
    DocumentRepository,
    SessionRepository,
    UnitOfWork,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def chunk_record() -> ChunkRecord:
    return ChunkRecord(
        document_id="doc-1",
        version=1,
        company="acme",
        owner="alice",
        text="Hello world",
    )


@pytest.fixture
def chunk(chunk_record: ChunkRecord) -> Chunk:
    return Chunk(chunk_record)


@pytest.fixture
def doc_record() -> DocumentRecord:
    return DocumentRecord(
        checksum="abc123",
        owner="alice",
        organization="acme",
        filename="test.pdf",
    )


@pytest.fixture
def document(doc_record: DocumentRecord) -> Document:
    return Document(doc_record)


@pytest.fixture
def session_record() -> SessionRecord:
    now = datetime.now(timezone.utc)
    return SessionRecord(
        user_id="user-1",
        expires_at=now,
        last_seen_at=now,
    )


@pytest.fixture
def session(session_record: SessionRecord) -> Session:
    return Session(session_record)


# ===================================================================
# Chunk tests
# ===================================================================


class TestChunk:
    def test_init_wraps_record(self, chunk: Chunk, chunk_record: ChunkRecord) -> None:
        assert chunk.record is chunk_record

    def test_chunk_id_property(self, chunk: Chunk, chunk_record: ChunkRecord) -> None:
        assert chunk.chunk_id == chunk_record.chunk_id

    def test_getattr_delegates_to_record(self, chunk: Chunk) -> None:
        assert chunk.text == "Hello world"
        assert chunk.document_id == "doc-1"
        assert chunk.company == "acme"

    def test_getattr_raises_on_missing(self, chunk: Chunk) -> None:
        with pytest.raises(AttributeError):
            _ = chunk.nonexistent_attr

    def test_setattr_sets_on_record(self, chunk: Chunk) -> None:
        chunk.text = "Updated text"
        assert chunk.record.text == "Updated text"
        assert chunk.text == "Updated text"

    def test_setattr_record_overriden_via_super(self, chunk: Chunk) -> None:
        new_record = ChunkRecord(document_id="d2", version=2, company="b", owner="o", text="t")
        chunk.record = new_record
        assert chunk.record is new_record
        assert chunk.chunk_id == new_record.chunk_id

    def test_update_sets_multiple_fields(self, chunk: Chunk, chunk_record: ChunkRecord) -> None:
        result = chunk.update(text="New text", page=42)
        assert chunk_record.text == "New text"
        assert chunk_record.page == 42
        assert result is chunk

    def test_update_empty_kwargs_is_noop(self, chunk: Chunk) -> None:
        original_text = chunk.text
        chunk.update()
        assert chunk.text == original_text


# ===================================================================
# Document tests
# ===================================================================


class TestDocument:
    def test_init_wraps_record(self, document: Document, doc_record: DocumentRecord) -> None:
        assert document.record is doc_record

    def test_document_id_property(self, document: Document, doc_record: DocumentRecord) -> None:
        assert document.document_id == doc_record.document_id

    def test_status_property(self, document: Document) -> None:
        assert document.status == DocumentLifecycleStatus.NEW

    def test_status_setter(self, document: Document) -> None:
        document.status = DocumentLifecycleStatus.PROCESSING
        assert document.record.status == DocumentLifecycleStatus.PROCESSING
        assert document.status == DocumentLifecycleStatus.PROCESSING

    def test_getattr_delegates_to_record(self, document: Document) -> None:
        assert document.owner == "alice"
        assert document.organization == "acme"

    def test_getattr_raises_on_missing(self, document: Document) -> None:
        with pytest.raises(AttributeError):
            _ = document.nonexistent

    def test_setattr_sets_on_record(self, document: Document) -> None:
        document.filename = "updated.pdf"
        assert document.record.filename == "updated.pdf"
        assert document.filename == "updated.pdf"

    def test_setattr_record_overriden_via_super(self, document: Document) -> None:
        new_record = DocumentRecord(checksum="x", owner="o", organization="a")
        document.record = new_record
        assert document.record is new_record
        assert document.document_id == new_record.document_id

    def test_update_sets_fields_and_updated_at(
        self, document: Document, doc_record: DocumentRecord
    ) -> None:
        before = doc_record.updated_at
        result = document.update(owner="bob", filename="new.pdf")
        assert doc_record.owner == "bob"
        assert doc_record.filename == "new.pdf"
        assert doc_record.updated_at > before
        assert result is document

    def test_update_empty_kwargs_still_touches_updated_at(
        self, document: Document, doc_record: DocumentRecord
    ) -> None:
        before = doc_record.updated_at
        document.update()
        assert doc_record.updated_at > before

    def test_mark_failed_sets_status_error_and_updated_at(
        self, document: Document, doc_record: DocumentRecord
    ) -> None:
        before = doc_record.updated_at
        result = document.mark_failed("Something went wrong")
        assert doc_record.status == DocumentLifecycleStatus.FAILED
        assert doc_record.error == "Something went wrong"
        assert doc_record.updated_at > before
        assert result is document

    def test_mark_failed_overwrites_previous_status(self, document: Document) -> None:
        document.status = DocumentLifecycleStatus.PROCESSING
        document.mark_failed("crash")
        assert document.status == DocumentLifecycleStatus.FAILED


# ===================================================================
# Session tests
# ===================================================================


class TestSession:
    def test_init_wraps_record(self, session: Session, session_record: SessionRecord) -> None:
        assert session.record is session_record

    def test_session_id_property(self, session: Session, session_record: SessionRecord) -> None:
        assert session.session_id == session_record.session_id

    def test_history_returns_copy(self, session: Session) -> None:
        assert session.history == []
        session.record.history.append(ConversationTurn(question="q", answer="a"))
        # the property returns a new list
        assert len(session.history) == 1
        hist = session.history
        session.record.history.append(ConversationTurn(question="q2", answer="a2"))
        assert len(hist) == 1  # original copy unchanged

    def test_getattr_delegates_to_record(self, session: Session) -> None:
        assert session.user_id == "user-1"

    def test_getattr_raises_on_missing(self, session: Session) -> None:
        with pytest.raises(AttributeError):
            _ = session.nonexistent

    def test_setattr_sets_on_record(self, session: Session) -> None:
        session.user_id = "user-2"
        assert session.record.user_id == "user-2"

    def test_setattr_record_overriden_via_super(self, session: Session) -> None:
        new_record = SessionRecord(
            user_id="u2",
            expires_at=datetime.now(timezone.utc),
            last_seen_at=datetime.now(timezone.utc),
        )
        session.record = new_record
        assert session.record is new_record
        assert session.session_id == new_record.session_id

    def test_add_turn_appends_and_updates_last_seen(
        self, session: Session, session_record: SessionRecord
    ) -> None:
        before = session_record.last_seen_at
        result = session.add_turn("Hello?", "World!")
        assert len(session_record.history) == 1
        turn = session_record.history[0]
        assert turn.question == "Hello?"
        assert turn.answer == "World!"
        assert session_record.last_seen_at > before
        assert result is session

    def test_add_turn_with_extra_kwargs(self, session: Session) -> None:
        session.add_turn("q", "a", metadata={"source": "test"})
        turn = session.record.history[0]
        assert turn.metadata == {"source": "test"}

    def test_clear_removes_history_and_updates_last_seen(
        self, session: Session, session_record: SessionRecord
    ) -> None:
        session.add_turn("q1", "a1")
        session.add_turn("q2", "a2")
        assert len(session_record.history) == 2
        before = session_record.last_seen_at
        result = session.clear()
        assert len(session_record.history) == 0
        assert session_record.last_seen_at >= before
        assert result is session

    def test_clear_empty_history(self, session: Session) -> None:
        result = session.clear()
        assert len(session.record.history) == 0
        assert result is session


# ===================================================================
# Repository protocol tests
# ===================================================================


class TestDocumentRepository:
    def test_cannot_instantiate(self) -> None:
        with pytest.raises(TypeError):
            DocumentRepository()  # type: ignore[abstract]

    def test_has_abstract_methods(self) -> None:
        expected = {
            "initialize",
            "save",
            "get",
            "get_by_checksum",
            "delete",
            "list_by_organization",
            "list_all",
        }
        for name in expected:
            assert hasattr(DocumentRepository, name)


class TestChunkRepository:
    def test_cannot_instantiate(self) -> None:
        with pytest.raises(TypeError):
            ChunkRepository()  # type: ignore[abstract]

    def test_has_abstract_methods(self) -> None:
        expected = {
            "initialize",
            "insert",
            "upsert",
            "delete_by_id",
            "delete_by_document",
            "search",
            "optimize",
            "health",
        }
        for name in expected:
            assert hasattr(ChunkRepository, name)


class TestSessionRepository:
    def test_cannot_instantiate(self) -> None:
        with pytest.raises(TypeError):
            SessionRepository()  # type: ignore[abstract]

    def test_has_abstract_methods(self) -> None:
        expected = {
            "initialize",
            "create",
            "save",
            "get",
            "get_by_token",
            "delete",
        }
        for name in expected:
            assert hasattr(SessionRepository, name)


# ===================================================================
# UnitOfWork tests
# ===================================================================


@pytest.fixture
def mock_doc_repo() -> MagicMock:
    return MagicMock(spec=DocumentRepository)


@pytest.fixture
def mock_chunk_repo() -> MagicMock:
    return MagicMock(spec=ChunkRepository)


@pytest.fixture
def mock_session_repo() -> MagicMock:
    return MagicMock(spec=SessionRepository)


@pytest.fixture
def mock_db_manager() -> MagicMock:
    manager = MagicMock(spec=DatabaseManager)
    conn = AsyncMock()
    manager.connection = conn
    return manager


@pytest.fixture
def uow(
    mock_doc_repo: MagicMock,
    mock_chunk_repo: MagicMock,
    mock_session_repo: MagicMock,
) -> UnitOfWork:
    return UnitOfWork(mock_doc_repo, mock_chunk_repo, mock_session_repo)


@pytest.fixture
def uow_with_db(
    mock_doc_repo: MagicMock,
    mock_chunk_repo: MagicMock,
    mock_session_repo: MagicMock,
    mock_db_manager: MagicMock,
) -> UnitOfWork:
    return UnitOfWork(mock_doc_repo, mock_chunk_repo, mock_session_repo, mock_db_manager)


class TestUnitOfWork:
    def test_init_stores_repos(
        self,
        mock_doc_repo: MagicMock,
        mock_chunk_repo: MagicMock,
        mock_session_repo: MagicMock,
    ) -> None:
        uow = UnitOfWork(mock_doc_repo, mock_chunk_repo, mock_session_repo)
        assert uow.document_repo is mock_doc_repo
        assert uow.chunk_repo is mock_chunk_repo
        assert uow.session_repo is mock_session_repo
        assert uow.db_manager is None
        assert uow.in_transaction is False

    def test_init_with_db_manager(
        self,
        mock_doc_repo: MagicMock,
        mock_chunk_repo: MagicMock,
        mock_session_repo: MagicMock,
        mock_db_manager: MagicMock,
    ) -> None:
        uow = UnitOfWork(mock_doc_repo, mock_chunk_repo, mock_session_repo, mock_db_manager)
        assert uow.db_manager is mock_db_manager

    @pytest.mark.asyncio
    async def test_initialize_calls_db_connect_and_all_repos(
        self,
        uow_with_db: UnitOfWork,
        mock_db_manager: MagicMock,
        mock_doc_repo: MagicMock,
        mock_chunk_repo: MagicMock,
        mock_session_repo: MagicMock,
    ) -> None:
        await uow_with_db.initialize()
        mock_db_manager.connect.assert_awaited_once()
        mock_doc_repo.initialize.assert_awaited_once()
        mock_chunk_repo.initialize.assert_awaited_once()
        mock_session_repo.initialize.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_initialize_without_db_skips_connect(
        self,
        uow: UnitOfWork,
        mock_doc_repo: MagicMock,
        mock_chunk_repo: MagicMock,
        mock_session_repo: MagicMock,
    ) -> None:
        await uow.initialize()
        mock_doc_repo.initialize.assert_awaited_once()
        mock_chunk_repo.initialize.assert_awaited_once()
        mock_session_repo.initialize.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_commit_when_in_transaction_and_db(
        self, uow_with_db: UnitOfWork, mock_db_manager: MagicMock
    ) -> None:
        uow_with_db.in_transaction = True
        await uow_with_db.commit()
        mock_db_manager.connection.commit.assert_awaited_once()
        assert uow_with_db.in_transaction is False

    @pytest.mark.asyncio
    async def test_commit_when_not_in_transaction(
        self, uow_with_db: UnitOfWork, mock_db_manager: MagicMock
    ) -> None:
        uow_with_db.in_transaction = False
        await uow_with_db.commit()
        mock_db_manager.connection.commit.assert_not_called()

    @pytest.mark.asyncio
    async def test_commit_without_db_manager(self, uow: UnitOfWork) -> None:
        uow.in_transaction = True
        await uow.commit()  # should not raise

    @pytest.mark.asyncio
    async def test_rollback_when_in_transaction_and_db(
        self, uow_with_db: UnitOfWork, mock_db_manager: MagicMock
    ) -> None:
        uow_with_db.in_transaction = True
        await uow_with_db.rollback()
        mock_db_manager.connection.rollback.assert_awaited_once()
        assert uow_with_db.in_transaction is False

    @pytest.mark.asyncio
    async def test_rollback_when_not_in_transaction(
        self, uow_with_db: UnitOfWork, mock_db_manager: MagicMock
    ) -> None:
        uow_with_db.in_transaction = False
        await uow_with_db.rollback()
        mock_db_manager.connection.rollback.assert_not_called()

    @pytest.mark.asyncio
    async def test_rollback_without_db_manager(self, uow: UnitOfWork) -> None:
        uow.in_transaction = True
        await uow.rollback()  # should not raise

    @pytest.mark.asyncio
    async def test_aenter_starts_transaction(
        self, uow_with_db: UnitOfWork, mock_db_manager: MagicMock
    ) -> None:
        result = await uow_with_db.__aenter__()
        mock_db_manager.connection.execute.assert_awaited_with("BEGIN")
        assert uow_with_db.in_transaction is True
        assert result is uow_with_db

    @pytest.mark.asyncio
    async def test_aenter_without_db_does_not_start_transaction(self, uow: UnitOfWork) -> None:
        result = await uow.__aenter__()
        assert uow.in_transaction is False
        assert result is uow

    @pytest.mark.asyncio
    async def test_aexit_no_exception_commits(
        self, uow_with_db: UnitOfWork, mock_db_manager: MagicMock
    ) -> None:
        uow_with_db.in_transaction = True
        await uow_with_db.__aexit__(None, None, None)
        mock_db_manager.connection.commit.assert_awaited_once()
        assert uow_with_db.in_transaction is False

    @pytest.mark.asyncio
    async def test_aexit_with_exception_rolls_back(
        self, uow_with_db: UnitOfWork, mock_db_manager: MagicMock
    ) -> None:
        uow_with_db.in_transaction = True
        await uow_with_db.__aexit__(ValueError, ValueError("bad"), None)
        mock_db_manager.connection.rollback.assert_awaited_once()
        assert uow_with_db.in_transaction is False

    @pytest.mark.asyncio
    async def test_aexit_without_db_does_nothing(self, uow: UnitOfWork) -> None:
        uow.in_transaction = True
        await uow.__aexit__(None, None, None)  # should not raise
