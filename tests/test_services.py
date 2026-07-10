from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from raghub.exceptions import AuthenticationError, AuthorizationError, DocumentError
from raghub.models import (
    ChunkRecord,
    DocumentLifecycleStatus,
    DocumentRecord,
    UserPrincipal,
)


class TestDocumentService:
    """Tests for :class:`raghub.services.document_service.DocumentService`."""

    @pytest.fixture
    def container(self) -> MagicMock:
        c = MagicMock()
        c.auth = AsyncMock()
        c.auth.resolve_user.return_value = (
            UserPrincipal(
                user_id="u1",
                email="admin@acme.com",
                allowed_companies=["Acme"],
                is_admin=True,
            ),
            [],
        )
        c.uow = MagicMock()
        c.uow.document_repo = AsyncMock()
        c.ingestion = AsyncMock()
        c.vector_store = MagicMock()
        return c

    @pytest.fixture
    def service(self, container: MagicMock) -> Any:
        from raghub.services.document_service import DocumentService

        return DocumentService(container)

    @pytest.mark.asyncio
    async def test_upload_document_success(self, service: Any, container: MagicMock) -> None:
        container.ingestion.ingest.return_value = MagicMock(
            document=DocumentRecord(
                document_id="d1",
                checksum="abc123",
                owner="admin@acme.com",
                organization="Acme",
                status=DocumentLifecycleStatus.READY,
            )
        )
        with patch("raghub.services.document_service.detect_mime_type", return_value="application/pdf"):
            result = await service.upload_document(
                token="tok1", filename="Acme_report.pdf", content=b"pdf data"
            )
        assert result.document_id == "d1"

    @pytest.mark.asyncio
    async def test_upload_document_authorization_error(
        self, service: Any, container: MagicMock
    ) -> None:
        container.auth.resolve_user.return_value = (
            UserPrincipal(
                user_id="u1",
                email="a@b.com",
                allowed_companies=["OtherCorp"],
                is_admin=False,
            ),
            [],
        )
        with (
            patch("raghub.services.document_service.detect_mime_type", return_value="application/pdf"),
            pytest.raises(AuthorizationError, match="cannot upload documents"),
        ):
            await service.upload_document(
                token="tok1", filename="Acme_report.pdf", content=b"data"
            )

    @pytest.mark.asyncio
    async def test_list_documents_admin_sees_all(
        self, service: Any, container: MagicMock
    ) -> None:
        container.uow.document_repo.list_all.return_value = [
            DocumentRecord(
                document_id="d1",
                checksum="abc",
                owner="admin@acme.com",
                organization="Acme",
                status=DocumentLifecycleStatus.READY,
            )
        ]
        docs = await service.list_documents("tok1")
        assert len(docs) == 1

    @pytest.mark.asyncio
    async def test_list_documents_nonadmin_scoped(
        self, service: Any, container: MagicMock
    ) -> None:
        container.auth.resolve_user.return_value = (
            UserPrincipal(
                user_id="u1",
                email="a@b.com",
                allowed_companies=["Acme"],
                is_admin=False,
            ),
            [],
        )
        container.uow.document_repo.list_by_organization.return_value = [
            DocumentRecord(
                document_id="d1",
                checksum="abc",
                owner="a@b.com",
                organization="Acme",
                status=DocumentLifecycleStatus.READY,
            )
        ]
        docs = await service.list_documents("tok1")
        assert len(docs) == 1
        container.uow.document_repo.list_by_organization.assert_awaited_once_with("Acme")

    @pytest.mark.asyncio
    async def test_document_status_success(
        self, service: Any, container: MagicMock
    ) -> None:
        container.uow.document_repo.get.return_value = DocumentRecord(
            document_id="d1",
            checksum="abc",
            owner="admin@acme.com",
            organization="Acme",
            status=DocumentLifecycleStatus.READY,
        )
        doc = await service.document_status("tok1", "d1")
        assert doc.document_id == "d1"

    @pytest.mark.asyncio
    async def test_document_status_not_found(
        self, service: Any, container: MagicMock
    ) -> None:
        container.uow.document_repo.get.return_value = None
        with pytest.raises(DocumentError):
            await service.document_status("tok1", "d1")

    @pytest.mark.asyncio
    async def test_delete_document_admin_only(
        self, service: Any, container: MagicMock
    ) -> None:
        await service.delete_document("tok1", "d1")
        container.uow.document_repo.delete.assert_awaited_once_with("d1")

    @pytest.mark.asyncio
    async def test_delete_document_forbidden_for_nonadmin(
        self, service: Any, container: MagicMock
    ) -> None:
        container.auth.resolve_user.return_value = (
            UserPrincipal(
                user_id="u1",
                email="a@b.com",
                allowed_companies=[],
                is_admin=False,
            ),
            [],
        )
        with pytest.raises(AuthorizationError):
            await service.delete_document("tok1", "d1")


class TestHealthService:
    """Tests for :class:`raghub.services.health_service.HealthService`."""

    @pytest.fixture
    def container(self) -> MagicMock:
        c = MagicMock()
        c.vector_store = MagicMock()
        return c

    @pytest.fixture
    def service(self, container: MagicMock) -> Any:
        from raghub.services.health_service import HealthService

        return HealthService(container)

    def test_health_returns_ok(self, service: Any, container: MagicMock) -> None:
        container.vector_store.health.return_value = {"status": "ok", "chunks": 42}
        result = service.health()
        assert result["status"] == "ok"
        assert result["components"]["vectorstore"]["chunks"] == 42


class TestQueryService:
    """Tests for :class:`raghub.services.query_service.QueryService`."""

    @pytest.fixture
    def container(self) -> MagicMock:
        c = MagicMock()
        c.auth = AsyncMock()
        c.auth.resolve_user.return_value = (
            UserPrincipal(
                user_id="u1",
                email="a@b.com",
                allowed_companies=["Acme"],
                is_admin=True,
            ),
            [],
        )
        c.retrieval = MagicMock()
        c.llm = MagicMock()
        c.prompt_builder = MagicMock()
        c.prompt_builder.config = MagicMock()
        c.prompt_builder.config.system_prompt = "You are a helpful assistant."
        c.conversation = AsyncMock()
        c.settings = MagicMock()
        c.settings.top_k = 5
        return c

    @pytest.fixture
    def service(self, container: MagicMock) -> Any:
        from raghub.services.query_service import QueryService

        return QueryService(container)

    @pytest.mark.asyncio
    async def test_query_returns_response(self, service: Any, container: MagicMock) -> None:
        chunk = ChunkRecord(
            chunk_id="c1",
            document_id="d1",
            version=1,
            text="Revenue grew 12%.",
            company="Acme",
            owner="a@b.com",
        )
        container.retrieval.retrieve.return_value = [
            MagicMock(chunk=chunk)
        ]
        container.llm.generate.return_value = "Revenue grew 12%."
        result = await service.query(token="tok1", question="What is revenue?")
        assert result.answer == "Revenue grew 12%."
        assert len(result.citations) == 1


class TestAuthService:
    """Tests for :class:`raghub.services.auth_service.AuthService`."""

    @pytest.fixture
    def container(self) -> MagicMock:
        c = MagicMock()
        c.authenticator = AsyncMock()
        c.user_store = AsyncMock()
        c.store = AsyncMock()
        return c

    @pytest.fixture
    def service(self, container: MagicMock) -> Any:
        from raghub.services.auth_service import AuthService

        return AuthService(container)

    @pytest.mark.asyncio
    async def test_login_success(self, service: Any, container: MagicMock) -> None:
        container.user_store.get_by_email.return_value = MagicMock(
            user_id="u1",
            email="a@b.com",
            allowed_companies=["Acme"],
            is_admin=False,
        )
        container.store.create_session.return_value = MagicMock(token="tok1")
        result = await service.login("a@b.com", "pwd")
        assert result.session_token == "tok1"
        assert result.user_email == "a@b.com"

    @pytest.mark.asyncio
    async def test_login_user_not_found(self, service: Any, container: MagicMock) -> None:
        container.authenticator.authenticate.return_value = None
        container.user_store.get_by_email.return_value = None
        with pytest.raises(AuthenticationError):
            await service.login("a@b.com", "pwd")

    @pytest.mark.asyncio
    async def test_logout_invalidates_session(self, service: Any, container: MagicMock) -> None:
        container.store.get_by_token.return_value = MagicMock(session_id="s1")
        await service.logout("tok1")
        container.store.delete_session.assert_awaited_once_with("s1")

    @pytest.mark.asyncio
    async def test_logout_noop_for_missing_session(self, service: Any, container: MagicMock) -> None:
        container.store.get_by_token.return_value = None
        await service.logout("tok1")
        container.store.delete_session.assert_not_called()

    @pytest.mark.asyncio
    async def test_resolve_user_success(self, service: Any, container: MagicMock) -> None:
        container.store.get_by_token.return_value = MagicMock(
            user_id="u1", history=[MagicMock()]
        )
        container.user_store.get_by_id.return_value = MagicMock(
            user_id="u1",
            email="a@b.com",
            allowed_companies=["Acme"],
            is_admin=False,
        )
        user, history = await service.resolve_user("tok1")
        assert isinstance(user, UserPrincipal)
        assert user.email == "a@b.com"

    @pytest.mark.asyncio
    async def test_resolve_user_session_expired(self, service: Any, container: MagicMock) -> None:
        container.store.get_by_token.return_value = None
        with pytest.raises(AuthenticationError):
            await service.resolve_user("tok1")

    @pytest.mark.asyncio
    async def test_resolve_user_user_deleted(self, service: Any, container: MagicMock) -> None:
        container.store.get_by_token.return_value = MagicMock(user_id="u1", history=[])
        container.user_store.get_by_id.return_value = None
        with pytest.raises(AuthenticationError):
            await service.resolve_user("tok1")
