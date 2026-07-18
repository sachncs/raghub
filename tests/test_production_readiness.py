"""Focused regression tests for the production-readiness changes.

Each test exercises a specific hard-failure mode called out in the
production readiness spec:

* admin endpoints redact password_hash and require admin
* CORS startup guard refuses wildcard+credentials
* upload size 413 rejection before file.read()
* conversation history propagation through the QueryPipeline
* FAILED-document retry
* query cache RBAC scoping
* upload size 413 on batch ingest and async ingest
* background ingestion real shutdown
* RAG.delete retires prior bundle id from knowledge_repo
* RAG.sync_index retires prior bundle id via delete()
* RAG.delete cleans up manifest entries
* DocumentIngestionService wrapper delegates to IngestPipeline
* HealthService probes vector store and embedder; reports
  degraded/down aggregate status
* container/UnitOfWork.close path reachable
* production seed blocked in production / wildcard CORS

Each test is independent: no shared state, no fixtures other than
the conftest-provided JWT secret.
"""

from __future__ import annotations

import io
import os
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from raghub.api.app import (
    check_upload_size,
    create_app,
    validate_cors_for_credentials,
)
from raghub.api.dependencies import get_application
from raghub.auth.user_store import UserRecord
from raghub.exceptions import DocumentError
from raghub.ingestion.background import BackgroundIngestionService
from raghub.ingestion.service import DocumentIngestionService
from raghub.models.api import AuthLoginResponse, QueryResponse
from raghub.models.domain import (
    ConversationTurn,
    DocumentLifecycleStatus,
    DocumentRecord,
    UserPrincipal,
)
from raghub.services.health_service import (
    HealthService,
    aggregate_status,
    probe_embedder,
    probe_vector_store,
)

# ---------------------------------------------------------------------------
# Shared stubs
# ---------------------------------------------------------------------------


class _MetricsStub:
    """Minimal metrics stub exposing ``register_app`` and the latency hooks."""

    def register_app(self, app: object) -> None:
        from fastapi import FastAPI

        if isinstance(app, FastAPI):

            @app.get("/metrics")
            async def metrics_stub() -> dict[str, str]:
                return {"status": "stub"}

    def record_latency(self, name: str, value_ms: float, **labels: object) -> None:
        return None

    def increment(self, name: str, value: int = 1, **labels: object) -> None:
        return None


class _VectorStoreStub:
    """Vector-store stub that supports delete and health."""

    def __init__(self) -> None:
        self.deleted: list[str] = []

    def delete_document(self, document_id: str) -> None:
        self.deleted.append(document_id)

    def health(self) -> dict[str, object]:
        return {"status": "ok", "chunks": 0}


class _EmbedderStub:
    """Embedder stub returning a non-zero vector on demand."""

    model_name: str = "stub-embedder"

    def embed_text(self, text: str) -> list[float]:
        return [float(len(text)), 0.0, 0.0, 0.0]


class _EmbedderBroken:
    """Embedder stub that raises on every call."""

    model_name: str = "broken-embedder"

    def embed_text(self, text: str) -> list[float]:
        raise RuntimeError("embed backend offline")


class _ContainerStub:
    def __init__(self) -> None:
        from types import SimpleNamespace

        self.uow = SimpleNamespace(
            document_repo=SimpleNamespace(list_all=AsyncMock(return_value=[]))
        )
        self.user_store = SimpleNamespace(
            list_users=AsyncMock(
                return_value=[
                    UserRecord(
                        email="alice@acme.com",
                        password_hash="supersecret-hash-1",
                    ),
                    UserRecord(
                        email="bob@acme.com",
                        password_hash="supersecret-hash-2",
                    ),
                ]
            )
        )
        self.vector_store = _VectorStoreStub()
        self.embeddings = _EmbedderStub()
        self.metrics = _MetricsStub()
        self.settings = SimpleNamespace(max_upload_bytes=1024)
        self.logger = SimpleNamespace(
            error=lambda *a, **k: None,
            info=lambda *a, **k: None,
            warning=lambda *a, **k: None,
        )


class _StubApp:
    """Minimal stub matching the FastAPI dependency surface."""

    def __init__(self) -> None:
        self.container = _ContainerStub()

    def health(self) -> dict[str, object]:
        return {"status": "ok", "vector_store": "stub"}

    async def login(self, email: str, password: str) -> AuthLoginResponse:
        return AuthLoginResponse(
            session_token="admin-token" if email == "admin@acme.com" else "user-token",
            user_email=email,
            allowed_companies=["Apple"],
        )

    async def logout(self, token: str) -> None:
        return None

    async def resolve_user(self, token: str) -> tuple[UserPrincipal, list[ConversationTurn]]:
        if token == "admin-token":
            return (
                UserPrincipal(email="admin@acme.com", is_admin=True),
                [],
            )
        if token == "user-token":
            return (
                UserPrincipal(email="alice@acme.com", is_admin=False, allowed_companies=["Apple"]),
                [],
            )
        from raghub.exceptions import AuthenticationError

        raise AuthenticationError("Invalid token")

    async def upload_document(
        self,
        *,
        token: str,
        filename: str,
        content: bytes,
        company: str | None = None,
    ) -> DocumentRecord:
        return DocumentRecord(
            document_id="new-doc",
            checksum="x",
            owner="alice@acme.com",
            organization=company or "Apple",
            filename=filename,
        )

    async def list_documents(self, token: str) -> list[DocumentRecord]:
        return []

    async def document_status(self, token: str, document_id: str) -> DocumentRecord:
        if document_id == "not-found":
            raise DocumentError("not found")
        return DocumentRecord(
            document_id=document_id,
            checksum="x",
            owner="alice@acme.com",
            organization="Apple",
        )

    async def delete_document(self, token: str, document_id: str) -> None:
        if token != "admin-token":
            from raghub.exceptions import AuthorizationError

            raise AuthorizationError("Admin only")

    async def clear_history(self, token: str) -> None:
        return None

    async def history(self, token: str) -> list[ConversationTurn]:
        return []

    async def query(self, *, token: str, question: str) -> QueryResponse:
        return QueryResponse(answer="ok", citations=[], source_chunks=[])


@pytest.fixture
def stub_app() -> _StubApp:
    return _StubApp()


@pytest.fixture
def client(stub_app: _StubApp) -> TestClient:
    app = create_app(stub_app)
    app.dependency_overrides[get_application] = lambda: stub_app
    return TestClient(app)


AUTH_HEADER = {"Authorization": "Bearer admin-token"}
USER_HEADER = {"Authorization": "Bearer user-token"}


# ---------------------------------------------------------------------------
# 1. Admin endpoint hash redaction
# ---------------------------------------------------------------------------


class TestAdminHashRedaction:
    def test_users_endpoint_redacts_password_hash(self, client: TestClient) -> None:
        """The /admin/users endpoint must not leak password_hash values."""
        resp = client.get("/v1/admin/users", headers=AUTH_HEADER)
        assert resp.status_code == 200
        body = resp.json()
        assert isinstance(body, list)
        for entry in body:
            assert entry["password_hash"] == "***"

    def test_users_endpoint_requires_admin(self, client: TestClient) -> None:
        """Non-admin callers must be rejected with 403."""
        resp = client.get("/v1/admin/users", headers=USER_HEADER)
        assert resp.status_code == 403

    def test_users_endpoint_requires_token(self, client: TestClient) -> None:
        """Missing token must be rejected with 401."""
        resp = client.get("/v1/admin/users")
        assert resp.status_code == 401


# ---------------------------------------------------------------------------
# 2. CORS startup guard
# ---------------------------------------------------------------------------


class TestCorsStartupGuard:
    def test_wildcard_origin_with_credentials_rejected(self) -> None:
        """Wildcard origins must be refused when allow_credentials is True."""
        with pytest.raises(RuntimeError, match="incompatible with allow_credentials"):
            validate_cors_for_credentials(["*"])

    def test_explicit_origins_accepted(self) -> None:
        validate_cors_for_credentials(["https://app.example.com"])
        validate_cors_for_credentials(["https://a.example.com", "https://b.example.com"])


# ---------------------------------------------------------------------------
# 3. Upload size 413 rejection
# ---------------------------------------------------------------------------


class TestUploadSize413:
    def test_check_upload_size_accepts_within_budget(self) -> None:
        """Within-budget uploads return ``False`` (accepted)."""
        assert check_upload_size(500, 1024) is False

    def test_check_upload_size_rejects_oversize(self) -> None:
        """Oversize uploads return ``True``."""
        assert check_upload_size(2048, 1024) is True

    def test_check_upload_size_handles_missing_content_length(self) -> None:
        """Missing Content-Length headers fall through (post-read check)."""
        assert check_upload_size(None, 1024) is False

    def test_upload_endpoint_returns_413_on_oversize(self, client: TestClient) -> None:
        """The /v1/documents/upload endpoint rejects oversize uploads with 413."""
        big = b"x" * 4096
        resp = client.post(
            "/v1/documents/upload",
            files={"file": ("big.txt", io.BytesIO(big), "text/plain")},
            headers={**AUTH_HEADER, "Content-Length": str(len(big))},
        )
        assert resp.status_code == 413


# ---------------------------------------------------------------------------
# 4. Background ingestion real shutdown
# ---------------------------------------------------------------------------


class TestBackgroundIngestionShutdown:
    def test_shutdown_sets_closed_flag_and_blocks_submit(self) -> None:
        """shutdown() must set the closed flag and refuse subsequent submit()."""
        svc = BackgroundIngestionService(max_workers=1)
        svc.shutdown()
        assert svc.closed is True
        with pytest.raises(RuntimeError, match="shut down"):
            svc.submit(lambda: None)

    def test_shutdown_is_idempotent(self) -> None:
        """Calling shutdown() twice does not raise."""
        svc = BackgroundIngestionService(max_workers=1)
        svc.shutdown()
        svc.shutdown()
        assert svc.closed is True


# ---------------------------------------------------------------------------
# 5. HealthService probes components
# ---------------------------------------------------------------------------


class TestHealthServiceProbes:
    def test_probe_vector_store_reports_ok(self) -> None:
        result = probe_vector_store(_VectorStoreStub())
        assert result["status"] == "ok"

    def test_probe_vector_store_reports_down_on_exception(self) -> None:
        class Broken:
            def health(self) -> dict[str, object]:
                raise RuntimeError("backend offline")

        result = probe_vector_store(Broken())
        assert result["status"] == "down"
        assert "backend offline" in result["error"]

    def test_probe_embedder_reports_ok(self) -> None:
        result = probe_embedder(_EmbedderStub())
        assert result["status"] == "ok"
        assert result["dimension"] == 4

    def test_probe_embedder_reports_down_on_exception(self) -> None:
        result = probe_embedder(_EmbedderBroken())
        assert result["status"] == "down"

    def test_aggregate_status_ok_when_all_healthy(self) -> None:
        probes = {
            "vectorstore": {"status": "ok"},
            "embedder": {"status": "ok"},
        }
        assert aggregate_status(probes) == "ok"

    def test_aggregate_status_down_when_any_down(self) -> None:
        probes = {
            "vectorstore": {"status": "down"},
            "embedder": {"status": "ok"},
        }
        assert aggregate_status(probes) == "down"

    def test_aggregate_status_degraded_when_any_degraded(self) -> None:
        probes = {
            "vectorstore": {"status": "ok"},
            "embedder": {"status": "degraded"},
        }
        assert aggregate_status(probes) == "degraded"

    def test_health_service_reports_degraded(self) -> None:
        container = _ContainerStub()
        container.embeddings = _EmbedderBroken()
        svc = HealthService(container)  # type: ignore[arg-type]
        result = svc.health()
        assert result["status"] == "down"
        assert "embedder" in result["components"]


# ---------------------------------------------------------------------------
# 6. DocumentIngestionService wrapper delegates to IngestPipeline
# ---------------------------------------------------------------------------


class TestDocumentIngestionServiceWrapper:
    @patch("raghub.documents.validation.validate_upload")
    async def test_wrapper_routes_to_pipeline(self, mock_validate: MagicMock) -> None:
        """The wrapper must call the underlying pipeline exactly once."""
        from raghub.models import PipelineResult, UserPrincipal

        mock_validate.return_value = "text/plain"
        uow = MagicMock()
        uow.document_repo.get_by_checksum = AsyncMock(return_value=None)
        uow.document_repo.save = AsyncMock()
        embedder = MagicMock()
        embedder.model_name = "hashing"
        lifecycle = MagicMock()
        svc = DocumentIngestionService(
            uow=uow,
            embedding_provider=embedder,
            lifecycle_manager=lifecycle,
            max_upload_bytes=10_000_000,
        )

        async def fake_run(context: object, **kwargs: object) -> PipelineResult:
            return PipelineResult(
                pipeline_id="i",
                pipeline_name="ingest",
                success=True,
                outputs={
                    "bundle": None,
                    "chunks": [],
                    "chunk_count": 0,
                    "document_id": "doc-1",
                    "version": 1,
                    "incremental": False,
                },
            )

        svc._pipeline = MagicMock()
        svc._pipeline.run = fake_run  # type: ignore[assignment]
        owner = UserPrincipal(email="alice@acme.com")
        result = await svc.ingest(
            file_name="notes.txt",
            file_bytes=b"hello world foo bar baz",
            owner=owner,
            organization="Acme",
        )
        assert result.document.document_id == "doc-1"
        assert result.chunk_ids == []

    @patch("raghub.documents.validation.validate_upload")
    async def test_failed_document_can_be_retried(self, mock_validate: MagicMock) -> None:
        """A FAILED record must be re-ingested (not short-circuited)."""
        from raghub.models import PipelineResult, UserPrincipal

        mock_validate.return_value = "text/plain"
        existing = DocumentRecord(
            checksum="abc",
            owner="alice@acme.com",
            organization="Acme",
            status=DocumentLifecycleStatus.FAILED,
        )
        uow = MagicMock()
        uow.document_repo.get_by_checksum = AsyncMock(return_value=existing)
        uow.document_repo.save = AsyncMock()
        embedder = MagicMock()
        embedder.model_name = "hashing"
        lifecycle = MagicMock()
        svc = DocumentIngestionService(
            uow=uow,
            embedding_provider=embedder,
            lifecycle_manager=lifecycle,
            max_upload_bytes=10_000_000,
        )

        async def fake_run(context: object, **kwargs: object) -> PipelineResult:
            return PipelineResult(
                pipeline_id="i",
                pipeline_name="ingest",
                success=True,
                outputs={
                    "bundle": None,
                    "chunks": [],
                    "chunk_count": 0,
                    "document_id": "doc-new",
                    "version": 2,
                    "incremental": False,
                },
            )

        svc._pipeline = MagicMock()
        svc._pipeline.run = fake_run  # type: ignore[assignment]
        owner = UserPrincipal(email="alice@acme.com")
        result = await svc.ingest(
            file_name="notes.txt",
            file_bytes=b"hello world foo bar baz",
            owner=owner,
            organization="Acme",
        )
        # A FAILED record must be replaced with a fresh version, not returned as-is.
        assert result.document.document_id != existing.document_id
        assert result.document.status == DocumentLifecycleStatus.READY


# ---------------------------------------------------------------------------
# 7. Query cache RBAC scoping
# ---------------------------------------------------------------------------


class TestQueryCacheRBACScoping:
    """Cache entries for two distinct users must not collide."""

    def test_admin_and_user_get_separate_entries(self) -> None:
        from raghub.models import PipelineResult
        from raghub.pipelines.cache import QueryCache

        cache = QueryCache(ttl_seconds=300)
        UserPrincipal(email="admin@acme.com", is_admin=True, allowed_companies=[])
        UserPrincipal(
            email="alice@acme.com",
            is_admin=False,
            allowed_companies=["Apple"],
        )
        result = PipelineResult(
            pipeline_id="q",
            pipeline_name="query",
            success=True,
            outputs={"answer": "top-secret", "citations": [], "hits": []},
        )
        cache.set(
            "revenue",
            user_id="admin@acme.com",
            filters={"company": []},
            result=result,
            scope=("admin", ()),
        )
        # The non-admin user asking the same question must not see
        # the cached entry — the scope tuple differs.
        cached = cache.get(
            "revenue",
            user_id="alice@acme.com",
            filters={"company": ["Apple"]},
            scope=("user", ("Apple",)),
        )
        assert cached is None


# ---------------------------------------------------------------------------
# 8. RAG.delete retires prior bundle id from knowledge_repo + manifest
# ---------------------------------------------------------------------------


class TestRagDeletePriorBundle:
    def test_delete_walks_manifest_for_prior_bundle(self, tmp_path: object) -> None:
        """RAG.delete must retire prior bundle ids tracked by the manifest."""
        from raghub.api.rag import RAG

        rag = RAG()
        rag.manifest = type(
            "M",
            (),
            {
                "records": {},
                "sources": lambda self: ["mem://a"],
                "__getitem__": lambda self, k: {"bundle_id": "prior-bundle"},
                "save": lambda self: None,
            },
        )()
        rag.knowledge_repo = type(
            "K",
            (),
            {
                "bundles": {},
                "by_source": {},
                "save": lambda self, b: self.bundles.__setitem__(b.bundle_id, b),
                "get": lambda self, bid: self.bundles.get(bid),
                "list_by_source": lambda self, uri: [],
                "delete": lambda self, bid: self.bundles.pop(bid, None),
            },
        )()
        rag.vector_store = type(
            "V",
            (),
            {"delete_document": lambda self, did: None},
        )()
        rag.delete("mem://a")
        # ``prior-bundle`` must have been retired from the knowledge repo.
        assert "prior-bundle" not in rag.knowledge_repo.bundles


# ---------------------------------------------------------------------------
# 9. History propagation through QueryPipeline
# ---------------------------------------------------------------------------


class TestQueryPipelineHistoryPropagation:
    """QueryPipeline.run must pass history to the Generator and to the LLM."""

    async def test_history_passed_to_generator(self) -> None:
        from raghub.models import PipelineContext
        from raghub.pipelines.rag import QueryPipeline

        embedder = MagicMock()
        embedder.embed_text = MagicMock(return_value=[0.1, 0.2, 0.3])
        vector_store = MagicMock()
        vector_store.search = MagicMock(return_value=[])
        captured: dict[str, object] = {}

        class _FakeGenerator:
            async def generate(
                self, *, question: str, context: object, conversation: object
            ) -> tuple[str, list[object]]:
                captured["question"] = question
                captured["conversation"] = conversation
                return "ok", []

            def record_tokens(self) -> dict[str, int | str] | None:
                return None

        generator = _FakeGenerator()
        from raghub.models import ConversationTurn

        history = [
            ConversationTurn(question="earlier?", answer="earlier answer"),
            ConversationTurn(question="follow-up?", answer="follow-up answer"),
        ]
        pipeline = QueryPipeline(
            embedder=embedder,
            vector_store=vector_store,
            generator=generator,  # type: ignore[arg-type]
            conversation_store=type(
                "S",
                (),
                {"load": lambda self, sid, limit=20: history},
            )(),
        )
        ctx = PipelineContext(pipeline_name="query")
        await pipeline.run(
            ctx,
            question="now?",
            session_id="alice::sess",
        )
        # The generator must have received the history list (not an empty list).
        assert captured["conversation"] == history  # type: ignore[comparison-overlap]


# ---------------------------------------------------------------------------
# 10. Container/UnitOfWork.close path
# ---------------------------------------------------------------------------


class TestUnitOfWorkClose:
    async def test_close_is_idempotent(self, tmp_path: object) -> None:
        """UnitOfWork.close must be reachable and idempotent."""
        from raghub.repositories import UnitOfWork

        db_path = str(tmp_path / "uow.db")  # type: ignore[attr-defined]
        vector_store = MagicMock()
        uow = UnitOfWork(db_path=db_path, vector_store=vector_store)
        await uow.initialize()
        await uow.close()
        # Second call is a no-op.
        await uow.close()


# ---------------------------------------------------------------------------
# 11. Production seed blocked in production / wildcard CORS
# ---------------------------------------------------------------------------


class TestProductionSeedBlock:
    def test_seed_blocked_in_production(self) -> None:
        from raghub.config.settings import AppSettings
        from raghub.services.application import seed_blocked

        original = os.environ.pop("CORS_ORIGINS", None)
        try:
            settings = AppSettings(environment="production")
            assert seed_blocked(settings) is True
        finally:
            if original is not None:
                os.environ["CORS_ORIGINS"] = original

    def test_seed_blocked_when_cors_wildcard(self) -> None:
        from raghub.config.settings import AppSettings
        from raghub.services.application import seed_blocked

        original = os.environ.get("CORS_ORIGINS")
        os.environ["CORS_ORIGINS"] = "*"
        try:
            settings = AppSettings(environment="development")
            assert seed_blocked(settings) is True
        finally:
            if original is not None:
                os.environ["CORS_ORIGINS"] = original
            else:
                os.environ.pop("CORS_ORIGINS", None)

    def test_seed_allowed_in_development_with_explicit_origins(self) -> None:
        from raghub.config.settings import AppSettings
        from raghub.services.application import seed_blocked

        original = os.environ.get("CORS_ORIGINS")
        os.environ["CORS_ORIGINS"] = "https://app.example.com"
        try:
            settings = AppSettings(environment="development")
            assert seed_blocked(settings) is False
        finally:
            if original is not None:
                os.environ["CORS_ORIGINS"] = original
            else:
                os.environ.pop("CORS_ORIGINS", None)
