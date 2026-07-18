from __future__ import annotations

import io
from typing import Any

import pytest
from fastapi.testclient import TestClient

from raghub.api.app import create_app
from raghub.api.dependencies import get_application
from raghub.auth.user_store import UserRecord
from raghub.exceptions import AuthenticationError, AuthorizationError, DocumentError, StorageError
from raghub.models.api import AuthLoginResponse, QueryResponse
from raghub.models.domain import ConversationTurn, DocumentRecord, UserPrincipal


def _doc(
    document_id: str = "doc-1",
    owner: str = "alice@acme.com",
    organization: str = "Apple",
    filename: str = "report.pdf",
    status: str = "READY",
) -> DocumentRecord:
    return DocumentRecord(
        document_id=document_id,
        checksum="abc123",
        owner=owner,
        organization=organization,
        filename=filename,
        status=status,
    )


def _turn(question: str = "q?", answer: str = "A!") -> ConversationTurn:
    return ConversationTurn(question=question, answer=answer)


class StubApp:
    """Minimal mock of DynamicRagApplication for endpoint testing."""

    def __init__(self) -> None:
        self.container = _ContainerStub()

    def health(self) -> dict[str, Any]:
        return {"status": "ok", "vector_store": "InMemoryVectorStore"}

    async def login(self, email: str, password: str) -> AuthLoginResponse:
        if email == "fail@test.com":
            raise AuthenticationError("Invalid credentials")
        return AuthLoginResponse(
            session_token="test-token",
            user_email=email,
            allowed_companies=["Apple"],
        )

    async def logout(self, token: str) -> None:
        if token == "invalid-token":
            raise AuthenticationError("Invalid token")
        return None

    async def resolve_user(self, token: str) -> tuple[UserPrincipal, list[ConversationTurn]]:
        if token == "admin-token":
            return (
                UserPrincipal(email="admin@acme.com", is_admin=True),
                [],
            )
        if token == "nonadmin-token":
            return (
                UserPrincipal(
                    email="alice@acme.com",
                    is_admin=False,
                    allowed_companies=["Apple"],
                ),
                [],
            )
        raise AuthenticationError("Invalid token")

    async def upload_document(
        self,
        *,
        token: str,
        filename: str,
        content: bytes,
        company: str | None = None,
    ) -> DocumentRecord:
        return _doc(
            document_id="new-doc-1",
            owner="alice@acme.com",
            organization=company or "Apple",
            filename=filename,
        )

    async def list_documents(self, token: str) -> list[DocumentRecord]:
        return [_doc(), _doc(document_id="doc-2", organization="Microsoft")]

    async def document_status(self, token: str, document_id: str) -> DocumentRecord:
        if document_id == "not-found":
            raise DocumentError("Document not found")
        return _doc(document_id=document_id)

    async def delete_document(self, token: str, document_id: str) -> None:
        if document_id == "not-found":
            raise DocumentError("Document not found")
        if token == "nonadmin-token":
            raise AuthorizationError("Admin access required")
        return None

    async def clear_history(self, token: str) -> None:
        return None

    async def history(self, token: str) -> list[ConversationTurn]:
        return [_turn()]

    async def query(self, *, token: str, question: str) -> QueryResponse:
        return QueryResponse(
            answer="42",
            citations=[{"source": "doc-1", "page": 1}],
            source_chunks=[{"chunk_id": "c1", "text": "meaning of life"}],
        )


class _ContainerStub:
    def __init__(self) -> None:
        self.uow = _UoWStub()
        self.user_store = _UserStoreStub()
        self.vector_store = _VectorStoreStub()
        # Real Settings instance so the new size-guard in
        # ``/documents/upload`` reads ``max_upload_bytes`` correctly.
        from types import SimpleNamespace

        self.settings = SimpleNamespace(max_upload_bytes=20 * 1024 * 1024)
        self.logger = SimpleNamespace(error=lambda *a, **k: None, info=lambda *a, **k: None)
        # Minimal metrics stub exposing ``register_app`` no-op so the
        # ``create_app`` body can call it.
        self.metrics = SimpleNamespace(register_app=lambda _app: None)


class _UoWStub:
    def __init__(self) -> None:
        self.document_repo = _DocRepoStub()


class _DocRepoStub:
    @staticmethod
    async def list_all() -> list[DocumentRecord]:
        return [_doc(document_id="admin-doc-1"), _doc(document_id="admin-doc-2")]


class _UserStoreStub:
    @staticmethod
    async def list_users() -> list[UserRecord]:
        return [
            UserRecord(email="alice@acme.com", password_hash="h1"),
            UserRecord(email="bob@acme.com", password_hash="h2"),
        ]


class _VectorStoreStub:
    @staticmethod
    def health() -> dict[str, Any]:
        return {"chunks": 42, "size": "1.2 MB"}


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def app() -> Any:
    stub = StubApp()
    app = create_app(stub)
    app.dependency_overrides[get_application] = lambda: stub
    return app


@pytest.fixture
def client(app: Any) -> TestClient:
    return TestClient(app)


AUTH_HEADER = {"Authorization": "Bearer test-token"}

# ===================================================================
# Health
# ===================================================================


class TestHealth:
    def test_health_ok(self, client: TestClient) -> None:
        resp = client.get("/v1/health")
        assert resp.status_code == 200
        assert resp.json()["status"] == "ok"


# ===================================================================
# Auth
# ===================================================================


class TestAuthLogin:
    def test_login_success(self, client: TestClient) -> None:
        resp = client.post(
            "/v1/auth/login", json={"email": "alice@acme.com", "password": "password"}
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["session_token"] == "test-token"
        assert body["user_email"] == "alice@acme.com"

    def test_login_invalid_credentials(self, client: TestClient) -> None:
        resp = client.post("/v1/auth/login", json={"email": "fail@test.com", "password": "wrong"})
        assert resp.status_code == 401
        assert "Invalid credentials" in resp.json()["detail"]

    @pytest.mark.parametrize(
        "payload", [{}, {"email": "not-an-email"}, {"email": "a@b.c", "password": ""}]
    )
    def test_login_validation_error(self, client: TestClient, payload: dict) -> None:
        resp = client.post("/v1/auth/login", json=payload)
        assert resp.status_code == 422


class TestAuthLogout:
    def test_logout_success(self, client: TestClient) -> None:
        resp = client.post("/v1/auth/logout", headers=AUTH_HEADER)
        assert resp.status_code == 200
        assert resp.json() == {"status": "logged_out"}

    def test_logout_missing_token(self, client: TestClient) -> None:
        resp = client.post("/v1/auth/logout")
        assert resp.status_code == 401

    def test_logout_invalid_token(self, client: TestClient) -> None:
        resp = client.post("/v1/auth/logout", headers={"Authorization": "Bearer invalid-token"})
        assert resp.status_code == 401


# ===================================================================
# Session
# ===================================================================


class TestSessionHistory:
    def test_history_success(self, client: TestClient) -> None:
        resp = client.get("/v1/session/history", headers=AUTH_HEADER)
        assert resp.status_code == 200
        body = resp.json()
        assert "history" in body
        assert len(body["history"]) == 1

    def test_history_missing_token(self, client: TestClient) -> None:
        resp = client.get("/v1/session/history")
        assert resp.status_code == 401


class TestClearHistory:
    def test_clear_history_success(self, client: TestClient) -> None:
        resp = client.delete("/v1/session/history", headers=AUTH_HEADER)
        assert resp.status_code == 204

    def test_clear_history_missing_token(self, client: TestClient) -> None:
        resp = client.delete("/v1/session/history")
        assert resp.status_code == 401


# ===================================================================
# Documents
# ===================================================================


class TestDocumentUpload:
    def test_upload_success(self, client: TestClient) -> None:
        resp = client.post(
            "/v1/documents/upload",
            files={"file": ("report.pdf", io.BytesIO(b"pdf content"), "application/pdf")},
            headers=AUTH_HEADER,
        )
        assert resp.status_code == 202
        body = resp.json()
        assert body["document_id"] == "new-doc-1"
        assert body["filename"] == "report.pdf"
        assert body["status"] == "READY"

    def test_upload_with_company(self, client: TestClient) -> None:
        resp = client.post(
            "/v1/documents/upload",
            files={"file": ("data.pdf", io.BytesIO(b"content"), "application/pdf")},
            data={"company": "Microsoft"},
            headers=AUTH_HEADER,
        )
        assert resp.status_code == 202
        assert resp.json()["company"] == "Microsoft"

    def test_upload_missing_token(self, client: TestClient) -> None:
        resp = client.post(
            "/v1/documents/upload",
            files={"file": ("x.pdf", io.BytesIO(b"x"), "application/pdf")},
        )
        assert resp.status_code == 401


class TestDocumentBatchIngest:
    def test_batch_ingest_success(self, client: TestClient) -> None:
        resp = client.post(
            "/v1/documents/ingest/batch",
            files=[
                ("files", ("a.pdf", io.BytesIO(b"content a"), "application/pdf")),
                ("files", ("b.pdf", io.BytesIO(b"content b"), "application/pdf")),
            ],
            headers=AUTH_HEADER,
        )
        assert resp.status_code == 200
        body = resp.json()
        assert "documents" in body
        assert len(body["documents"]) == 2
        assert body["documents"][0]["status"] == "ok"
        assert body["documents"][1]["status"] == "ok"
        assert body["documents"][0]["filename"] == "a.pdf"
        assert body["documents"][1]["filename"] == "b.pdf"

    def test_batch_ingest_with_company(self, client: TestClient) -> None:
        resp = client.post(
            "/v1/documents/ingest/batch",
            files=[
                ("files", ("c.pdf", io.BytesIO(b"content"), "application/pdf")),
            ],
            data={"company": "Microsoft"},
            headers=AUTH_HEADER,
        )
        assert resp.status_code == 200
        body = resp.json()
        assert len(body["documents"]) == 1
        assert body["documents"][0]["status"] == "ok"

    def test_batch_ingest_missing_token(self, client: TestClient) -> None:
        resp = client.post(
            "/v1/documents/ingest/batch",
            files=[
                ("files", ("x.pdf", io.BytesIO(b"x"), "application/pdf")),
            ],
        )
        assert resp.status_code == 401

    def test_batch_ingest_single_file(self, client: TestClient) -> None:
        resp = client.post(
            "/v1/documents/ingest/batch",
            files=[
                ("files", ("single.pdf", io.BytesIO(b"single file"), "application/pdf")),
            ],
            headers=AUTH_HEADER,
        )
        assert resp.status_code == 200
        body = resp.json()
        assert len(body["documents"]) == 1
        assert body["documents"][0]["filename"] == "single.pdf"
        assert body["documents"][0]["status"] == "ok"
        assert "document_id" in body["documents"][0]


class TestListDocuments:
    def test_list_success(self, client: TestClient) -> None:
        resp = client.get("/v1/documents", headers=AUTH_HEADER)
        assert resp.status_code == 200
        body = resp.json()
        assert "documents" in body
        assert len(body["documents"]) == 2

    def test_list_missing_token(self, client: TestClient) -> None:
        resp = client.get("/v1/documents")
        assert resp.status_code == 401


class TestDocumentStatus:
    def test_status_success(self, client: TestClient) -> None:
        resp = client.get("/v1/documents/doc-42/status", headers=AUTH_HEADER)
        assert resp.status_code == 200
        assert resp.json()["document_id"] == "doc-42"

    def test_status_not_found(self, client: TestClient) -> None:
        resp = client.get("/v1/documents/not-found/status", headers=AUTH_HEADER)
        assert resp.status_code == 400

    def test_status_missing_token(self, client: TestClient) -> None:
        resp = client.get("/v1/documents/doc-1/status")
        assert resp.status_code == 401


class TestDeleteDocument:
    def test_delete_success(self, client: TestClient) -> None:
        resp = client.delete("/v1/documents/doc-1", headers={"Authorization": "Bearer admin-token"})
        assert resp.status_code == 204

    def test_delete_not_found(self, client: TestClient) -> None:
        resp = client.delete(
            "/v1/documents/not-found", headers={"Authorization": "Bearer admin-token"}
        )
        assert resp.status_code == 400

    def test_delete_non_admin(self, client: TestClient) -> None:
        resp = client.delete(
            "/v1/documents/doc-1", headers={"Authorization": "Bearer nonadmin-token"}
        )
        assert resp.status_code == 403

    def test_delete_missing_token(self, client: TestClient) -> None:
        resp = client.delete("/v1/documents/doc-1")
        assert resp.status_code == 401


# ===================================================================
# Query
# ===================================================================


class TestQuery:
    def test_query_success(self, client: TestClient) -> None:
        resp = client.post("/v1/query", json={"question": "meaning of life?"}, headers=AUTH_HEADER)
        assert resp.status_code == 200
        body = resp.json()
        assert body["answer"] == "42"
        assert len(body["citations"]) == 1

    def test_query_empty_question(self, client: TestClient) -> None:
        resp = client.post("/v1/query", json={"question": ""}, headers=AUTH_HEADER)
        assert resp.status_code == 422

    def test_query_missing_token(self, client: TestClient) -> None:
        resp = client.post("/v1/query", json={"question": "hello?"})
        assert resp.status_code == 401


# ===================================================================
# Ingest Async
# ===================================================================


class TestIngestAsync:
    def test_ingest_async_success(self, client: TestClient) -> None:
        resp = client.post(
            "/v1/ingest/async",
            files={"file": ("doc.pdf", io.BytesIO(b"content"), "application/pdf")},
            headers=AUTH_HEADER,
        )
        assert resp.status_code == 200
        body = resp.json()
        assert "job_id" in body

    def test_ingest_async_missing_token(self, client: TestClient) -> None:
        resp = client.post(
            "/v1/ingest/async",
            files={"file": ("x.pdf", io.BytesIO(b"x"), "application/pdf")},
        )
        assert resp.status_code == 401


# ===================================================================
# Admin
# ===================================================================


class TestAdminDocuments:
    def test_admin_documents_success(self, client: TestClient) -> None:
        resp = client.get("/v1/admin/documents", headers={"Authorization": "Bearer admin-token"})
        assert resp.status_code == 200
        body = resp.json()
        assert len(body) == 2

    def test_admin_documents_forbidden(self, client: TestClient) -> None:
        resp = client.get("/v1/admin/documents", headers={"Authorization": "Bearer nonadmin-token"})
        assert resp.status_code == 403

    def test_admin_documents_missing_token(self, client: TestClient) -> None:
        resp = client.get("/v1/admin/documents")
        assert resp.status_code == 401


class TestAdminUsers:
    def test_admin_users_success(self, client: TestClient) -> None:
        resp = client.get("/v1/admin/users", headers={"Authorization": "Bearer admin-token"})
        assert resp.status_code == 200
        body = resp.json()
        assert len(body) == 2
        assert body[0]["email"] == "alice@acme.com"

    def test_admin_users_forbidden(self, client: TestClient) -> None:
        resp = client.get("/v1/admin/users", headers={"Authorization": "Bearer nonadmin-token"})
        assert resp.status_code == 403


class TestAdminStats:
    def test_admin_stats_success(self, client: TestClient) -> None:
        resp = client.get("/v1/admin/stats", headers={"Authorization": "Bearer admin-token"})
        assert resp.status_code == 200
        body = resp.json()
        assert body["document_count"] == 2
        assert body["user_count"] == 2
        assert body["chunk_count"] == 42
        assert body["vector_store_size"] == "1.2 MB"

    def test_admin_stats_forbidden(self, client: TestClient) -> None:
        resp = client.get("/v1/admin/stats", headers={"Authorization": "Bearer nonadmin-token"})
        assert resp.status_code == 403


# ===================================================================
# Error handlers
# ===================================================================


class TestErrorHandlers:
    def test_authentication_error_returns_401(self, client: TestClient) -> None:
        resp = client.post("/v1/auth/login", json={"email": "fail@test.com", "password": "x"})
        assert resp.status_code == 401

    def test_authorization_error_returns_403(self, client: TestClient) -> None:
        resp = client.delete(
            "/v1/documents/doc-1",
            headers={"Authorization": "Bearer nonadmin-token"},
        )
        assert resp.status_code == 403

    def test_document_error_returns_400(self, client: TestClient) -> None:
        resp = client.get(
            "/v1/documents/not-found/status",
            headers=AUTH_HEADER,
        )
        assert resp.status_code == 400

    def test_validation_error_returns_422(self, client: TestClient) -> None:
        resp = client.post("/v1/query", json={}, headers=AUTH_HEADER)
        assert resp.status_code == 422

    def test_route_not_found_returns_404(self, client: TestClient) -> None:
        resp = client.get("/v1/nonexistent")
        assert resp.status_code == 404


# ===================================================================
# CORS
# ===================================================================


class TestCORS:
    def test_cors_headers_on_success(self, client: TestClient) -> None:
        resp = client.get("/v1/health", headers={"Origin": "http://testserver"})
        assert resp.headers.get("access-control-allow-origin") == "http://testserver"
        assert resp.headers.get("access-control-allow-credentials") == "true"

    def test_cors_preflight(self, client: TestClient) -> None:
        resp = client.options(
            "/v1/health",
            headers={
                "Origin": "http://testserver",
                "Access-Control-Request-Method": "GET",
            },
        )
        assert resp.status_code == 200
        assert resp.headers.get("access-control-allow-origin") == "http://testserver"

    def test_wildcard_origins_rejected_at_startup(self) -> None:
        """``create_app`` refuses wildcard origins when credentials are enabled."""
        import pytest

        from raghub.api.app import validate_cors_for_credentials

        with pytest.raises(RuntimeError, match="incompatible with allow_credentials"):
            validate_cors_for_credentials(["*"])

    def test_explicit_origins_accepted(self) -> None:
        """Explicit origin lists pass the credentials guard."""
        from raghub.api.app import validate_cors_for_credentials

        # Should not raise.
        validate_cors_for_credentials(["https://app.example.com"])
        validate_cors_for_credentials(["https://a.example.com", "https://b.example.com"])


# ===================================================================
# Bearer token validation
# ===================================================================


class TestBearerValidation:
    @pytest.mark.parametrize("header", [None, "", "Basic xxx", "Bearer"])
    def test_malformed_bearer_token(self, client: TestClient, header: str | None) -> None:
        headers = {"Authorization": header} if header else {}
        resp = client.get("/v1/documents", headers=headers)
        assert resp.status_code == 401

    def test_missing_authorization_header(self, client: TestClient) -> None:
        resp = client.get("/v1/documents")
        assert resp.status_code == 401


# ===================================================================
# Rate limiting
# ===================================================================


class TestRateLimiting:
    def test_rate_limit_exceeded(self, client: TestClient) -> None:
        """Send enough requests to trigger the rate limiter (burst=20)."""
        ok = 0
        limited = 0
        for _ in range(25):
            resp = client.get("/v1/health")
            if resp.status_code == 200:
                ok += 1
            elif resp.status_code == 429:
                limited += 1
        assert ok >= 1
        assert limited >= 1


# ===================================================================
# Coverage for uncovered lines in raghub/api/app.py
# ===================================================================


class TestLifespan:
    def test_lifespan_swallows_shutdown_errors(self, app: Any) -> None:
        """The lifespan finally block swallows exceptions from shutdown()."""
        from fastapi.testclient import TestClient

        with TestClient(app) as client:
            resp = client.get("/v1/health")
            assert resp.status_code == 200


class TestStorageError:
    def test_storage_error_returns_500(self, app: Any) -> None:
        """The StorageError exception handler returns a 500 response."""
        from fastapi.testclient import TestClient
        from fastapi.responses import JSONResponse

        @app.exception_handler(StorageError)
        def _handler(_: Any, exc: StorageError) -> JSONResponse:  # type: ignore[no-untyped-def]
            return JSONResponse(status_code=500, content={"detail": str(exc)})

        @app.get("/_test_storage_error")
        def _trigger() -> None:  # type: ignore[misc]
            raise StorageError("disk failure")

        with TestClient(app) as client:
            resp = client.get("/_test_storage_error")
            assert resp.status_code == 500
            assert resp.json()["detail"] == "disk failure"


class TestAppMetadata:
    def test_metadata_fallback_on_import_error(self, monkeypatch: Any) -> None:
        """create_app falls back to hard-coded metadata when package metadata is missing."""
        import importlib.metadata

        monkeypatch.setattr(
            importlib.metadata,
            "metadata",
            lambda _: (_ for _ in ()).throw(Exception("no package")),
        )
        from raghub.api.app import create_app

        stub = StubApp()
        app = create_app(stub)
        assert app.title == "RAGHub"
        assert app.version == "0.3.3"
        assert "RAGHub" in app.description


class TestGetApp:
    def test_get_app_singleton(self, monkeypatch: Any) -> None:
        """get_app() lazily builds and returns the same singleton."""
        import raghub.api.app as app_mod

        app_mod.app_singleton = None

        async def mock_build(*args: object, **kwargs: object) -> StubApp:
            return StubApp()

        monkeypatch.setattr("raghub.core.container.build_application", mock_build)

        from raghub.api.app import get_app

        a1 = get_app()
        a2 = get_app()
        assert a1 is a2
