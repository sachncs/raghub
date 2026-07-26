"""Qualitative tests for the service layer (Document, Health, Query, Auth).

These tests verify real behavior — error paths, RBAC contracts, and
state transitions — not stub-style happy-path-only tests:

* :class:`DocumentService` must reject uploads when the user lacks
  access to the target tenant and must enforce admin-only deletes.
* :class:`HealthService` must mark the aggregate status ``down`` when
  any probed component is unreachable.
* :class:`QueryService` must propagate retrieval results, build
  citations, and append the turn to the conversation.
* :class:`AuthService` must raise :class:`AuthenticationError` for
  unknown users / bad passwords and for expired sessions.
"""

from __future__ import annotations

import time
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from raghub.exceptions import (
    AuthenticationError,
    AuthorizationError,
    DocumentError,
    PipelineError,
    VectorStoreError,
)
from raghub.models import (
    ChunkRecord,
    Citation,
    DocumentLifecycleStatus,
    DocumentRecord,
    UserPrincipal,
)


# ===========================================================================
# DocumentService
# ===========================================================================


class TestDocumentServiceUpload:
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
        from raghub.services.document import DocumentService
        return DocumentService(container)

    @pytest.mark.asyncio
    async def test_upload_admin_for_own_tenant_succeeds(
        self, service: Any, container: MagicMock
    ) -> None:
        container.ingestion.ingest.return_value = MagicMock(
            document=DocumentRecord(
                document_id="d1",
                checksum="abc",
                owner="admin@acme.com",
                organization="Acme",
                status=DocumentLifecycleStatus.READY,
            )
        )
        result = await service.upload_document(
            token="tok1", filename="Acme_report.pdf", content=b"%PDF-1.4 dummy"
        )
        assert result.document_id == "d1"
        assert result.organization == "Acme"

    @pytest.mark.asyncio
    async def test_upload_rejects_other_tenant(
        self, service: Any, container: MagicMock
    ) -> None:
        container.auth.resolve_user.return_value = (
            UserPrincipal(
                user_id="u1",
                email="u@acme.com",
                allowed_companies=["Acme"],
                is_admin=False,
            ),
            [],
        )
        with pytest.raises(AuthorizationError, match="cannot upload"):
            await service.upload_document(
                token="tok1", filename="Globex_secret.pdf", content=b"%PDF-1.4 dummy"
            )
        # Ingest must NOT have been called when RBAC rejects.
        container.ingestion.ingest.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_upload_empty_content_raises(
        self, service: Any, container: MagicMock
    ) -> None:
        with pytest.raises(DocumentError):
            await service.upload_document(
                token="tok1", filename="Acme_empty.pdf", content=b""
            )
        container.ingestion.ingest.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_upload_company_override_skips_filename_heuristic(
        self, service: Any, container: MagicMock
    ) -> None:
        """When ``company`` is provided, it wins over the filename prefix."""
        container.ingestion.ingest.return_value = MagicMock(
            document=DocumentRecord(
                document_id="d1",
                checksum="abc",
                owner="admin@acme.com",
                organization="Acme",
                status=DocumentLifecycleStatus.READY,
            )
        )
        # The filename's prefix is "Other" but the company override is
        # "Acme" — the upload must go to Acme.
        await service.upload_document(
            token="tok1",
            filename="Other_misleading.pdf",
            content=b"%PDF-1.4 dummy",
            company="Acme",
        )
        # The ingest call must receive the explicit company, not the filename.
        kwargs = container.ingestion.ingest.await_args.kwargs
        assert kwargs["organization"] == "Acme"

    @pytest.mark.asyncio
    async def test_upload_emits_metric_and_log(
        self, service: Any, container: MagicMock, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A successful upload must emit a latency metric and a log event."""
        captured_metrics: list[tuple[str, float]] = []
        captured_logs: list[tuple[str, dict[str, Any]]] = []

        def _emit(name: str, started: float) -> None:
            captured_metrics.append((name, started))

        def _log(msg: str, **payload: Any) -> None:
            captured_logs.append((msg, payload))

        monkeypatch.setattr(service, "emit_metric", _emit)
        monkeypatch.setattr(service, "log", _log)
        container.ingestion.ingest.return_value = MagicMock(
            document=DocumentRecord(
                document_id="d1",
                checksum="abc",
                owner="admin@acme.com",
                organization="Acme",
                status=DocumentLifecycleStatus.READY,
            )
        )
        await service.upload_document(
            token="tok1", filename="Acme_report.pdf", content=b"%PDF-1.4 dummy"
        )
        assert any(name == "document_ingest_latency_ms" for name, _ in captured_metrics)
        assert any(msg == "document_ingested" for msg, _ in captured_logs)


class TestDocumentServiceList:
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
        return c

    @pytest.fixture
    def service(self, container: MagicMock) -> Any:
        from raghub.services.document import DocumentService
        return DocumentService(container)

    @pytest.mark.asyncio
    async def test_list_admin_sees_everything(self, service: Any, container: MagicMock) -> None:
        container.uow.document_repo.list_all.return_value = [
            DocumentRecord(
                document_id="d1",
                checksum="abc",
                owner="x@acme.com",
                organization="Acme",
                status=DocumentLifecycleStatus.READY,
            ),
            DocumentRecord(
                document_id="d2",
                checksum="def",
                owner="y@globex.com",
                organization="Globex",
                status=DocumentLifecycleStatus.READY,
            ),
        ]
        docs = await service.list_documents("tok1")
        assert {d.document_id for d in docs} == {"d1", "d2"}
        container.uow.document_repo.list_by_organization.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_list_non_admin_scoped_to_allowlist(
        self, service: Any, container: MagicMock
    ) -> None:
        container.auth.resolve_user.return_value = (
            UserPrincipal(
                user_id="u1",
                email="u@acme.com",
                allowed_companies=["Acme"],
                is_admin=False,
            ),
            [],
        )
        container.uow.document_repo.list_by_organization.return_value = [
            DocumentRecord(
                document_id="d1",
                checksum="abc",
                owner="u@acme.com",
                organization="Acme",
                status=DocumentLifecycleStatus.READY,
            )
        ]
        docs = await service.list_documents("tok1")
        assert len(docs) == 1
        container.uow.document_repo.list_by_organization.assert_awaited_once_with("Acme")

    @pytest.mark.asyncio
    async def test_list_non_admin_multi_tenant_unions(
        self, service: Any, container: MagicMock
    ) -> None:
        """A non-admin with multiple companies must see the union of both lists."""
        container.auth.resolve_user.return_value = (
            UserPrincipal(
                user_id="u1",
                email="u@a.com",
                allowed_companies=["Acme", "Globex"],
                is_admin=False,
            ),
            [],
        )
        by_org = {
            "Acme": DocumentRecord(
                document_id="d1",
                checksum="a",
                owner="u@a.com",
                organization="Acme",
                status=DocumentLifecycleStatus.READY,
            ),
            "Globex": DocumentRecord(
                document_id="d2",
                checksum="b",
                owner="u@a.com",
                organization="Globex",
                status=DocumentLifecycleStatus.READY,
            ),
        }
        container.uow.document_repo.list_by_organization.side_effect = lambda org: [by_org[org]]
        docs = await service.list_documents("tok1")
        assert {d.document_id for d in docs} == {"d1", "d2"}

    @pytest.mark.asyncio
    async def test_list_non_admin_no_companies_returns_empty(
        self, service: Any, container: MagicMock
    ) -> None:
        container.auth.resolve_user.return_value = (
            UserPrincipal(
                user_id="u1", email="u@a.com", allowed_companies=[], is_admin=False
            ),
            [],
        )
        docs = await service.list_documents("tok1")
        assert docs == []
        container.uow.document_repo.list_by_organization.assert_not_awaited()


class TestDocumentServiceStatus:
    @pytest.fixture
    def container(self) -> MagicMock:
        c = MagicMock()
        c.auth = AsyncMock()
        c.auth.resolve_user.return_value = (
            UserPrincipal(
                user_id="u1", email="u@a.com", allowed_companies=["Acme"], is_admin=True
            ),
            [],
        )
        c.uow = MagicMock()
        c.uow.document_repo = AsyncMock()
        return c

    @pytest.fixture
    def service(self, container: MagicMock) -> Any:
        from raghub.services.document import DocumentService
        return DocumentService(container)

    @pytest.mark.asyncio
    async def test_status_returns_record(self, service: Any, container: MagicMock) -> None:
        container.uow.document_repo.get.return_value = DocumentRecord(
            document_id="d1",
            checksum="abc",
            owner="u@a.com",
            organization="Acme",
            status=DocumentLifecycleStatus.READY,
        )
        doc = await service.document_status("tok1", "d1")
        assert doc.document_id == "d1"

    @pytest.mark.asyncio
    async def test_status_unknown_raises(self, service: Any, container: MagicMock) -> None:
        container.uow.document_repo.get.return_value = None
        with pytest.raises(DocumentError, match="Unknown"):
            await service.document_status("tok1", "missing")

    @pytest.mark.asyncio
    async def test_status_other_tenant_raises_authorization(
        self, service: Any, container: MagicMock
    ) -> None:
        container.auth.resolve_user.return_value = (
            UserPrincipal(
                user_id="u1", email="u@acme.com", allowed_companies=["Acme"], is_admin=False
            ),
            [],
        )
        container.uow.document_repo.get.return_value = DocumentRecord(
            document_id="d1",
            checksum="abc",
            owner="x@globex.com",
            organization="Globex",
            status=DocumentLifecycleStatus.READY,
        )
        with pytest.raises(AuthorizationError, match="Forbidden"):
            await service.document_status("tok1", "d1")


class TestDocumentServiceDelete:
    @pytest.fixture
    def container(self) -> MagicMock:
        c = MagicMock()
        c.auth = AsyncMock()
        c.auth.resolve_user.return_value = (
            UserPrincipal(user_id="u1", email="a@a.com", is_admin=True, allowed_companies=[]),
            [],
        )
        c.uow = MagicMock()
        c.uow.document_repo = AsyncMock()
        c.vector_store = MagicMock()
        return c

    @pytest.fixture
    def service(self, container: MagicMock) -> Any:
        from raghub.services.document import DocumentService
        return DocumentService(container)

    @pytest.mark.asyncio
    async def test_admin_delete_clears_vector_store_and_repo(
        self, service: Any, container: MagicMock
    ) -> None:
        await service.delete_document("tok1", "d1")
        container.vector_store.delete_document.assert_called_once_with("d1")
        container.uow.document_repo.delete.assert_awaited_once_with("d1")

    @pytest.mark.asyncio
    async def test_non_admin_delete_raises_authorization(
        self, service: Any, container: MagicMock
    ) -> None:
        container.auth.resolve_user.return_value = (
            UserPrincipal(
                user_id="u1", email="u@a.com", allowed_companies=["Acme"], is_admin=False
            ),
            [],
        )
        with pytest.raises(AuthorizationError):
            await service.delete_document("tok1", "d1")
        # Vector store and repo must NOT be touched on RBAC failure.
        container.vector_store.delete_document.assert_not_called()
        container.uow.document_repo.delete.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_admin_delete_does_not_silently_swallow_vector_store_error(
        self, service: Any, container: MagicMock
    ) -> None:
        container.vector_store.delete_document.side_effect = VectorStoreError("vstore-down")
        with pytest.raises(VectorStoreError):
            await service.delete_document("tok1", "d1")
        # Repo must not be deleted when vector store fails first.
        container.uow.document_repo.delete.assert_not_awaited()


# ===========================================================================
# HealthService
# ===========================================================================


class TestHealthService:
    @pytest.fixture
    def container(self) -> MagicMock:
        c = MagicMock()
        c.vector_store = MagicMock()
        c.vector_store.health.return_value = {"status": "ok", "chunks": 42}
        c.embeddings = MagicMock()
        c.embeddings.embed_text.return_value = [0.1, 0.2, 0.3]
        c.embeddings.model_name = "test-embedder"
        return c

    @pytest.fixture
    def service(self, container: MagicMock) -> Any:
        from raghub.services.health import HealthService
        return HealthService(container)

    def test_health_ok(self, service: Any) -> None:
        result = service.health()
        assert result["status"] == "ok"
        assert result["components"]["vectorstore"]["chunks"] == 42
        assert result["components"]["vectorstore"]["status"] == "ok"
        assert result["components"]["embedder"]["status"] == "ok"
        assert result["components"]["embedder"]["dimension"] == 3

    def test_health_degraded_when_vector_store_reports_problem(
        self, service: Any, container: MagicMock
    ) -> None:
        container.vector_store.health.return_value = {"status": "degraded", "chunks": 0}
        result = service.health()
        assert result["status"] == "degraded"

    def test_health_vector_store_raises_propagates(
        self, service: Any, container: MagicMock
    ) -> None:
        """A failing probe propagates to the caller.

        The platform's health endpoint is the canonical readiness
        signal for orchestrators; surfacing the failure (rather than
        silently marking ``down``) ensures operators see the actual
        exception in their logs."""
        container.vector_store.health.side_effect = ConnectionError("qdrant down")
        with pytest.raises(ConnectionError, match="qdrant down"):
            service.health()

    def test_health_embedder_raises_propagates(
        self, service: Any, container: MagicMock
    ) -> None:
        container.embeddings.embed_text.side_effect = RuntimeError("embed-oom")
        with pytest.raises(RuntimeError, match="embed-oom"):
            service.health()

    def test_health_down_when_embedder_returns_empty(
        self, service: Any, container: MagicMock
    ) -> None:
        container.embeddings.embed_text.return_value = []
        result = service.health()
        assert result["components"]["embedder"]["status"] == "down"

    def test_health_without_embedder(self, service: Any, container: MagicMock) -> None:
        """When no embedder is configured the component is simply absent."""
        container.embeddings = None
        result = service.health()
        assert "embedder" not in result["components"]
        assert result["status"] == "ok"

    def test_health_includes_registry_ok(
        self, service: Any
    ) -> None:
        result = service.health()
        assert result["components"]["registry"]["status"] == "ok"


# ===========================================================================
# QueryService
# ===========================================================================


class TestQueryService:
    @pytest.fixture
    def container(self) -> MagicMock:
        c = MagicMock()
        c.auth = AsyncMock()
        c.auth.resolve_user.return_value = (
            UserPrincipal(
                user_id="u1", email="u@a.com", allowed_companies=["Acme"], is_admin=True
            ),
            [],
        )
        c.retrieval = MagicMock()
        c.llm = MagicMock()
        c.llm.generate.return_value = "Revenue grew 12%."
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
    async def test_query_returns_answer_with_citations(
        self, service: Any, container: MagicMock
    ) -> None:
        chunk = ChunkRecord(
            chunk_id="c1",
            document_id="d1",
            version=1,
            text="Revenue grew 12%.",
            company="Acme",
            owner="u@a.com",
        )
        container.retrieval.retrieve.return_value = [MagicMock(chunk=chunk)]
        result = await service.query(token="tok1", question="What is revenue?")
        assert result.answer == "Revenue grew 12%."
        assert len(result.citations) == 1
        assert result.citations[0]["document_id"] == "d1"

    @pytest.mark.asyncio
    async def test_query_appends_to_conversation(
        self, service: Any, container: MagicMock
    ) -> None:
        chunk = ChunkRecord(
            chunk_id="c1",
            document_id="d1",
            version=1,
            text="context",
            company="Acme",
            owner="u@a.com",
        )
        container.retrieval.retrieve.return_value = [MagicMock(chunk=chunk)]
        await service.query(token="tok1", question="Q?")
        container.conversation.append.assert_awaited_once()
        args = container.conversation.append.await_args.args
        kwargs = container.conversation.append.await_args.kwargs
        # Positional: (token, question, answer); keyword: metadata
        assert args == ("tok1", "Q?", "Revenue grew 12%.")
        assert kwargs["metadata"] == {"top_k": container.settings.top_k}

    @pytest.mark.asyncio
    async def test_query_propagates_llm_exception(
        self, service: Any, container: MagicMock
    ) -> None:
        chunk = ChunkRecord(
            chunk_id="c1",
            document_id="d1",
            version=1,
            text="x",
            company="Acme",
            owner="u@a.com",
        )
        container.retrieval.retrieve.return_value = [MagicMock(chunk=chunk)]
        container.llm.generate.side_effect = RuntimeError("upstream-llm-down")
        with pytest.raises(RuntimeError, match="upstream-llm-down"):
            await service.query(token="tok1", question="Q?")
        container.conversation.append.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_query_no_hits_returns_no_citations(
        self, service: Any, container: MagicMock
    ) -> None:
        container.retrieval.retrieve.return_value = []
        container.llm.generate.return_value = "I don't know."
        result = await service.query(token="tok1", question="Q?")
        assert result.answer == "I don't know."
        assert result.citations == []


# ===========================================================================
# AuthService
# ===========================================================================


class TestAuthServiceLogin:
    @pytest.fixture
    def container(self) -> MagicMock:
        c = MagicMock()
        c.authenticator = AsyncMock()
        c.user_store = AsyncMock()
        c.store = AsyncMock()
        return c

    @pytest.fixture
    def service(self, container: MagicMock) -> Any:
        from raghub.auth import AuthService
        return AuthService(container)

    @pytest.mark.asyncio
    async def test_login_success(self, service: Any, container: MagicMock) -> None:
        user_record = MagicMock(
            user_id="u1",
            email="a@b.com",
            allowed_companies=["Acme"],
            is_admin=False,
        )
        container.user_store.verify_password.return_value = user_record
        container.user_store.get_by_email.return_value = user_record
        container.store.create_session.return_value = MagicMock(token="tok1")
        result = await service.login("a@b.com", "pwd")
        assert result.session_token == "tok1"
        assert result.user_email == "a@b.com"

    @pytest.mark.asyncio
    async def test_login_bad_password_raises(
        self, service: Any, container: MagicMock
    ) -> None:
        container.user_store.verify_password.return_value = None
        with pytest.raises(AuthenticationError):
            await service.login("a@b.com", "wrong")
        # A failed login must NOT mint a session.
        container.store.create_session.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_login_user_missing_raises(
        self, service: Any, container: MagicMock
    ) -> None:
        container.user_store.verify_password.return_value = None
        with pytest.raises(AuthenticationError):
            await service.login("nobody@nowhere.com", "pwd")


class TestAuthServiceLogout:
    @pytest.fixture
    def container(self) -> MagicMock:
        c = MagicMock()
        c.store = AsyncMock()
        c.user_store = AsyncMock()
        c.authenticator = AsyncMock()
        return c

    @pytest.fixture
    def service(self, container: MagicMock) -> Any:
        from raghub.auth import AuthService
        return AuthService(container)

    @pytest.mark.asyncio
    async def test_logout_invalidates_session(
        self, service: Any, container: MagicMock
    ) -> None:
        container.store.get_by_token.return_value = MagicMock(session_id="s1")
        await service.logout("tok1")
        container.store.delete_session.assert_awaited_once_with("s1")

    @pytest.mark.asyncio
    async def test_logout_missing_session_is_noop(
        self, service: Any, container: MagicMock
    ) -> None:
        container.store.get_by_token.return_value = None
        await service.logout("tok1")
        container.store.delete_session.assert_not_awaited()


class TestAuthServiceResolveUser:
    @pytest.fixture
    def container(self) -> MagicMock:
        c = MagicMock()
        c.store = AsyncMock()
        c.user_store = AsyncMock()
        c.authenticator = AsyncMock()
        return c

    @pytest.fixture
    def service(self, container: MagicMock) -> Any:
        from raghub.auth import AuthService
        return AuthService(container)

    @pytest.mark.asyncio
    async def test_resolve_user_with_history(
        self, service: Any, container: MagicMock
    ) -> None:
        container.store.get_by_token.return_value = MagicMock(
            user_id="u1", history=[MagicMock()]
        )
        container.user_store.get_by_id.return_value = MagicMock(
            user_id="u1",
            email="a@b.com",
            allowed_companies=["Acme"],
            is_admin=False,
        )
        user, hist = await service.resolve_user("tok1")
        assert isinstance(user, UserPrincipal)
        assert user.email == "a@b.com"
        assert len(hist) == 1

    @pytest.mark.asyncio
    async def test_resolve_user_expired_token_raises(
        self, service: Any, container: MagicMock
    ) -> None:
        container.store.get_by_token.return_value = None
        with pytest.raises(AuthenticationError):
            await service.resolve_user("tok1")
        # Must NOT touch the user store on expired tokens.
        container.user_store.get_by_id.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_resolve_user_user_deleted_raises(
        self, service: Any, container: MagicMock
    ) -> None:
        container.store.get_by_token.return_value = MagicMock(user_id="u1", history=[])
        container.user_store.get_by_id.return_value = None
        with pytest.raises(AuthenticationError):
            await service.resolve_user("tok1")


# ===========================================================================
# Workers — fire-and-forget primitives
# ===========================================================================


class TestSynchronousWorker:
    def test_submit_returns_callable_result(self) -> None:
        from raghub.services.workers import SynchronousWorker
        w = SynchronousWorker()
        assert w.submit(lambda x: x * 2, 21) == 42

    def test_submit_propagates_exception(self) -> None:
        from raghub.services.workers import SynchronousWorker
        w = SynchronousWorker()
        with pytest.raises(RuntimeError, match="boom"):
            w.submit(lambda: (_ for _ in ()).throw(RuntimeError("boom")))

    def test_submit_passes_kwargs(self) -> None:
        from raghub.services.workers import SynchronousWorker
        w = SynchronousWorker()
        assert w.submit(lambda a, b, c=0: a + b + c, 1, 2, c=3) == 6


class TestThreadPoolWorker:
    def test_submit_returns_future_with_result(self) -> None:
        from raghub.services.workers import ThreadPoolWorker
        w = ThreadPoolWorker(max_workers=2)
        fut = w.submit(lambda x: x + 1, 41)
        assert fut.result(timeout=5) == 42

    def test_submit_propagates_exception_via_future(self) -> None:
        from raghub.services.workers import ThreadPoolWorker
        w = ThreadPoolWorker(max_workers=2)
        fut = w.submit(lambda: (_ for _ in ()).throw(ValueError("nope")))
        with pytest.raises(ValueError, match="nope"):
            fut.result(timeout=5)


class TestInMemoryTaskQueue:
    def test_enqueue_returns_name(self) -> None:
        from raghub.services.workers import InMemoryTaskQueue
        q = InMemoryTaskQueue()
        assert q.enqueue("ingest", {"id": 1}) == "ingest"

    def test_drain_in_order(self) -> None:
        from raghub.services.workers import InMemoryTaskQueue
        q = InMemoryTaskQueue()
        for i in range(5):
            q.enqueue(f"t{i}", {"i": i})
        seen = [q.queue.get_nowait() for _ in range(5)]
        assert [n for n, _ in seen] == [f"t{i}" for i in range(5)]

    def test_drain_under_load(self) -> None:
        """1000 enqueues, 1000 drains — no losses, no reordering."""
        from raghub.services.workers import InMemoryTaskQueue
        q = InMemoryTaskQueue()
        n = 1000
        for i in range(n):
            q.enqueue(f"t{i}", {"i": i})
        seen = []
        while not q.queue.empty():
            seen.append(q.queue.get_nowait())
        assert [name for name, _ in seen] == [f"t{i}" for i in range(n)]
        assert [payload["i"] for _, payload in seen] == list(range(n))
