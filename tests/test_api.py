"""API surface coverage tests.

Exercises ``raghub.api`` helper functions and the lifespan / exception
handler wiring. The full route tree already has smoke coverage through
``tests/test_production_readiness.py``; this file targets the pieces
not reached there.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

import pytest
from fastapi import FastAPI, HTTPException, Request
from fastapi.testclient import TestClient

from raghub.api import (
    AppFactory,
    ExceptionHandlers,
    Lifespan,
    RouteGroup,
    check_upload_size,
    cors_origins_from_env,
    enforce_upload_limit,
    package_metadata,
    root_health_route,
    upload_content_length,
    user_store_or_503,
    validate_cors,
)

# ---------------------------------------------------------------------------
# CORS helpers
# ---------------------------------------------------------------------------


def test_cors_origins_default_is_wildcard(monkeypatch: pytest.MonkeyPatch) -> None:
    """When CORS_ORIGINS is unset, the default is ``['*']``."""

    monkeypatch.delenv("CORS_ORIGINS", raising=False)
    assert cors_origins_from_env() == ["*"]


def test_cors_origins_empty_string_is_wildcard(monkeypatch: pytest.MonkeyPatch) -> None:
    """An empty CORS_ORIGINS string falls back to ``['*']``."""

    monkeypatch.setenv("CORS_ORIGINS", "")
    assert cors_origins_from_env() == ["*"]


def test_cors_origins_splits_csv(monkeypatch: pytest.MonkeyPatch) -> None:
    """A comma-separated list is split into clean strings."""

    monkeypatch.setenv("CORS_ORIGINS", "https://a.example, https://b.example ,")
    assert cors_origins_from_env() == ["https://a.example", "https://b.example"]


def test_validate_cors_raises_on_wildcard() -> None:
    """validate_cors refuses ``'*'`` because allow_credentials=True forbids it."""

    from raghub.errors import ConfigurationError

    with pytest.raises(ConfigurationError, match="CORS_ORIGINS='\\*'"):
        validate_cors(["*"])


def test_validate_cors_raises_when_list_contains_wildcard() -> None:
    """validate_cors rejects ``'*'`` even when other origins are present."""

    from raghub.errors import ConfigurationError

    with pytest.raises(ConfigurationError, match="CORS_ORIGINS='\\*'"):
        validate_cors(["https://a.example", "*"])


def test_validate_cors_passes_explicit_list() -> None:
    """validate_cors is silent for an explicit list of origins."""

    validate_cors(["https://a.example", "https://b.example"])


# ---------------------------------------------------------------------------
# Upload guards
# ---------------------------------------------------------------------------


def test_check_upload_size_none_returns_false() -> None:
    """A missing Content-Length returns False (post-read check will run)."""

    assert check_upload_size(None, 1000) is False


def test_check_upload_size_under_limit_returns_false() -> None:
    """An upload under the limit returns False."""

    assert check_upload_size(500, 1000) is False


def test_check_upload_size_at_limit_returns_false() -> None:
    """An upload exactly at the limit returns False."""

    assert check_upload_size(1000, 1000) is False


def test_check_upload_size_over_limit_returns_true() -> None:
    """An upload over the limit returns True."""

    assert check_upload_size(2000, 1000) is True


def test_upload_content_length_returns_int() -> None:
    """upload_content_length parses an integer header value."""

    captured: dict[str, int | None] = {}

    def _route(request: Request) -> dict[str, int | None]:
        captured["value"] = upload_content_length(request)
        return {"value": captured["value"]}

    app = FastAPI()
    app.add_api_route("/probe", _route, methods=["GET"])
    client = TestClient(app)
    response = client.get("/probe", headers={"content-length": "123"})
    assert response.status_code == 200
    assert captured["value"] == 123


def test_upload_content_length_missing_returns_none() -> None:
    """upload_content_length returns None when the header is absent."""

    captured: dict[str, int | None] = {}

    def _route(request: Request) -> dict[str, int | None]:
        captured["value"] = upload_content_length(request)
        return {"value": captured["value"]}

    app = FastAPI()
    app.add_api_route("/probe", _route, methods=["GET"])
    client = TestClient(app)
    response = client.get("/probe")
    assert response.status_code == 200
    assert captured["value"] is None


def test_upload_content_length_invalid_returns_none() -> None:
    """upload_content_length returns None for a non-integer header value."""

    # The TestClient will not let us inject a non-numeric Content-Length,
    # but we can verify the branch by invoking the helper directly with a
    # request object whose header is bogus. Starlette will tolerate it
    # because we read the header lazily via request.headers.get.
    from starlette.datastructures import Headers

    headers = Headers(raw=[(b"content-length", b"not-a-number")])
    scope = {"type": "http", "headers": headers.raw}
    request = Request(scope)
    assert upload_content_length(request) is None


def test_enforce_upload_limit_no_setting_is_silent() -> None:
    """When max_upload_bytes is 0, enforce_upload_limit is a no-op."""

    container = MagicMock()
    container.settings.max_upload_bytes = 0

    captured: dict[str, bool] = {}

    def _route(request: Request) -> dict[str, bool]:
        enforce_upload_limit(request, container)
        captured["ok"] = True
        return {"ok": True}

    app = FastAPI()
    app.add_api_route("/probe", _route, methods=["POST"])
    client = TestClient(app)
    response = client.post("/probe", content=b"data")
    assert response.status_code == 200
    assert captured["ok"] is True


def test_enforce_upload_limit_oversize_raises() -> None:
    """An oversize upload raises HTTP 413 (exercised at the helper level)."""

    container = MagicMock()
    container.settings.max_upload_bytes = 100

    def _route(request: Request) -> dict[str, bool]:
        enforce_upload_limit(request, container)
        return {"ok": True}

    app = FastAPI()
    app.add_api_route("/probe", _route, methods=["POST"])
    client = TestClient(app, raise_server_exceptions=False)
    response = client.post("/probe", content=b"x" * 200)
    assert response.status_code == 413


def test_enforce_upload_limit_undersize_is_silent() -> None:
    """An undersize upload passes silently."""

    container = MagicMock()
    container.settings.max_upload_bytes = 10000

    captured: dict[str, bool] = {}

    def _route(request: Request) -> dict[str, bool]:
        enforce_upload_limit(request, container)
        captured["ok"] = True
        return {"ok": True}

    app = FastAPI()
    app.add_api_route("/probe", _route, methods=["POST"])
    client = TestClient(app)
    response = client.post("/probe", content=b"x" * 5)
    assert response.status_code == 200
    assert captured["ok"] is True


# ---------------------------------------------------------------------------
# user_store_or_503
# ---------------------------------------------------------------------------


def test_user_store_or_503_returns_user_store() -> None:
    """user_store_or_503 returns the configured user store with prefs API."""

    class _FakeStore:
        def get_pref(self, user_id: str, key: str) -> None:
            return None

        def set_pref(self, user_id: str, key: str, value: object) -> None:
            return None

    store = _FakeStore()
    app_service = MagicMock()
    app_service.container.user_store = store
    assert user_store_or_503(app_service) is store


def test_user_store_or_503_raises_503_when_missing() -> None:
    """user_store_or_503 raises HTTPException(503) when user_store is None."""

    app_service = MagicMock()
    app_service.container.user_store = None
    with pytest.raises(HTTPException) as exc_info:
        user_store_or_503(app_service)
    assert exc_info.value.status_code == 503


def test_user_store_or_503_raises_503_when_missing_prefs_api() -> None:
    """user_store_or_503 raises when store lacks get_pref/set_pref."""

    class _StubStore:
        pass

    app_service = MagicMock()
    app_service.container.user_store = _StubStore()
    with pytest.raises(HTTPException) as exc_info:
        user_store_or_503(app_service)
    assert exc_info.value.status_code == 503


# ---------------------------------------------------------------------------
# package_metadata
# ---------------------------------------------------------------------------


def test_package_metadata_returns_three_strings() -> None:
    """package_metadata returns (name, version, summary) for the raghub package."""

    name, version, summary = package_metadata()
    assert isinstance(name, str)
    assert isinstance(version, str)
    assert isinstance(summary, str)
    assert version != ""


# ---------------------------------------------------------------------------
# root_health_route
# ---------------------------------------------------------------------------


def test_root_health_route_adds_endpoint() -> None:
    """root_health_route wires GET /health onto the app."""

    app = FastAPI()
    root_health_route(app)
    client = TestClient(app)
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"


def test_root_health_route_adds_at_least_one() -> None:
    """root_health_route ensures at least one /health route exists."""

    app = FastAPI()
    root_health_route(app)
    paths = [r.path for r in app.router.routes]
    assert "/health" in paths


# ---------------------------------------------------------------------------
# ExceptionHandlers wiring
# ---------------------------------------------------------------------------


@pytest.fixture
def exception_handlers_app() -> FastAPI:
    """A tiny FastAPI app with exception handlers wired."""

    app = FastAPI()
    ExceptionHandlers.install(app)

    @app.get("/raise-auth")
    def _raise_auth() -> None:
        from raghub.errors import AuthenticationError

        raise AuthenticationError("nope")

    @app.get("/raise-authz")
    def _raise_authz() -> None:
        from raghub.errors import AuthorizationError

        raise AuthorizationError("forbidden")

    @app.get("/raise-ingest")
    def _raise_ingest() -> None:
        from raghub.errors import IngestionError

        raise IngestionError("bad input")

    @app.get("/raise-raghub")
    def _raise_raghub() -> None:
        from raghub.errors import RagHubError

        raise RagHubError("something")

    @app.get("/raise-generic")
    def _raise_generic() -> None:
        raise ValueError("crash")

    @app.get("/raise-not-impl")
    def _raise_not_impl() -> None:
        raise NotImplementedError("not implemented")

    return app


def test_exception_handler_authentication_401(exception_handlers_app: FastAPI) -> None:
    """AuthenticationError maps to HTTP 401."""

    client = TestClient(exception_handlers_app, raise_server_exceptions=False)
    response = client.get("/raise-auth")
    assert response.status_code == 401


def test_exception_handler_authorization_403(exception_handlers_app: FastAPI) -> None:
    """AuthorizationError maps to HTTP 403."""

    client = TestClient(exception_handlers_app, raise_server_exceptions=False)
    response = client.get("/raise-authz")
    assert response.status_code == 403


def test_exception_handler_ingestion_400(exception_handlers_app: FastAPI) -> None:
    """IngestionError maps to HTTP 400."""

    client = TestClient(exception_handlers_app, raise_server_exceptions=False)
    response = client.get("/raise-ingest")
    assert response.status_code == 400


def test_exception_handler_raghub_500(exception_handlers_app: FastAPI) -> None:
    """RagHubError (non-auth) maps to HTTP 500."""

    client = TestClient(exception_handlers_app, raise_server_exceptions=False)
    response = client.get("/raise-raghub")
    assert response.status_code == 500


def test_exception_handler_unexpected_500(exception_handlers_app: FastAPI) -> None:
    """Unexpected exceptions map to HTTP 500."""

    client = TestClient(exception_handlers_app, raise_server_exceptions=False)
    response = client.get("/raise-generic")
    assert response.status_code == 500


# ---------------------------------------------------------------------------
# Lifespan
# ---------------------------------------------------------------------------


def test_lifespan_drives_fastapi(monkeypatch: pytest.MonkeyPatch) -> None:
    """Lifespan __call__ shuts down the application facade on exit."""

    calls: list[str] = []
    app_service = MagicMock()

    async def _shutdown() -> None:
        calls.append("down")

    app_service.shutdown = _shutdown
    lifespan = Lifespan(app_service)

    async def _drive() -> None:
        async with lifespan(MagicMock()):
            calls.append("inside")

    asyncio.run(_drive())
    assert calls == ["inside", "down"]


# ---------------------------------------------------------------------------
# AppFactory wiring
# ---------------------------------------------------------------------------


def test_app_factory_class_attributes_exist() -> None:
    """AppFactory exposes a create_app method."""

    assert hasattr(AppFactory, "create_app")


# ---------------------------------------------------------------------------
# Module-level: create_app + helper exports
# ---------------------------------------------------------------------------


def test_module_all() -> None:
    """The api module exports the documented public names."""

    import raghub.api as api_module

    for name in (
        "AppFactory",
        "ExceptionHandlers",
        "Lifespan",
        "RouteGroup",
        "check_upload_size",
        "cors_origins_from_env",
        "create_app",
        "enforce_upload_limit",
        "package_metadata",
        "root_health_route",
        "upload_content_length",
        "user_store_or_503",
        "validate_cors",
    ):
        assert name in api_module.__all__


# ---------------------------------------------------------------------------
# RouteGroup interface
# ---------------------------------------------------------------------------


def test_route_group_class_exists() -> None:
    """The RouteGroup class is exported."""

    assert hasattr(RouteGroup, "__init__") or hasattr(RouteGroup, "router")
    assert "RouteGroup" in vars(__import__("raghub.api", fromlist=["RouteGroup"]))


# ---------------------------------------------------------------------------
# api_auth helpers
# ---------------------------------------------------------------------------


class _FakeApp:
    """Test client carrying an application facade on its state."""

    def __init__(self, application: Any) -> None:
        self.state = SimpleNamespace(application=application)


def test_app_get_returns_application() -> None:
    """App.get reads the application facade off the request app's state."""

    from raghub.api_auth import App

    application = MagicMock()
    request = MagicMock()
    request.app = _FakeApp(application)
    assert App.get(request) is application


def test_bearer_require_strips_token() -> None:
    """Bearer.require returns the trailing token, trimmed."""

    from raghub.api_auth import Bearer

    assert Bearer.require("Bearer my-token  ") == "my-token"


def test_bearer_require_lowercase() -> None:
    """Bearer.require is case-insensitive on the scheme."""

    from raghub.api_auth import Bearer

    assert Bearer.require("bearer t") == "t"


def test_bearer_require_missing_raises() -> None:
    """Bearer.require raises HTTPException(401) when header is missing."""

    from raghub.api_auth import Bearer

    with pytest.raises(HTTPException) as exc_info:
        Bearer.require(None)
    assert exc_info.value.status_code == 401


def test_bearer_require_wrong_scheme_raises() -> None:
    """Bearer.require rejects non-Bearer schemes."""

    from raghub.api_auth import Bearer

    with pytest.raises(HTTPException) as exc_info:
        Bearer.require("Basic xyz")
    assert exc_info.value.status_code == 401


def test_auth_admin_resolves_admin() -> None:
    """Auth.admin returns the user when is_admin is True."""

    from raghub.api_auth import Auth

    admin = MagicMock()
    admin.is_admin = True

    async def _resolve(_token: str) -> tuple[Any, list[Any]]:
        return admin, []

    app_service = MagicMock()
    app_service.resolve_user = _resolve

    async def _drive() -> Any:
        return await Auth.admin(authorization="Bearer t", app_service=app_service)

    assert asyncio.run(_drive()) is admin


def test_auth_admin_403_for_non_admin() -> None:
    """Auth.admin raises 403 for non-admin users."""

    from raghub.api_auth import Auth

    user = MagicMock()
    user.is_admin = False

    async def _resolve(_token: str) -> tuple[Any, list[Any]]:
        return user, []

    app_service = MagicMock()
    app_service.resolve_user = _resolve

    async def _drive() -> None:
        await Auth.admin(authorization="Bearer t", app_service=app_service)

    with pytest.raises(HTTPException) as exc_info:
        asyncio.run(_drive())
    assert exc_info.value.status_code == 403


def test_auth_user_id_helper() -> None:
    """Auth.user_id returns user.id from the inner auth service."""

    from raghub.api_auth import Auth

    user = MagicMock()
    user.id = "u-1"
    inner = MagicMock()

    async def _resolve(_token: str) -> tuple[Any, list[Any]]:
        return user, []

    inner.resolve_user = _resolve
    app_service = MagicMock()
    app_service.auth = inner

    async def _drive() -> str:
        return await Auth.user_id(app_service, "t")

    assert asyncio.run(_drive()) == "u-1"


# ---------------------------------------------------------------------------
# api_response helpers
# ---------------------------------------------------------------------------


def test_redaction_user_strips_sensitive_keys() -> None:
    """Redaction.user replaces sensitive keys with ``***``."""

    from raghub.api_response import Redaction

    payload = {"email": "a@x.com", "password": "x", "password_hash": "h", "token": "t"}
    redacted = Redaction.user(payload)
    assert redacted["email"] == "a@x.com"
    assert redacted["password"] == "***"
    assert redacted["password_hash"] == "***"
    assert redacted["token"] == "***"


def test_redaction_user_case_insensitive() -> None:
    """Redaction matches sensitive keys case-insensitively."""

    from raghub.api_response import Redaction

    payload = {"PASSWORD": "x", "Secret": "y", "Token": "z"}
    redacted = Redaction.user(payload)
    assert all(redacted[k] == "***" for k in payload)


def test_response_builder_from_pipeline() -> None:
    """ResponseBuilder.from_pipeline builds a Response from a Pipeline result."""

    from raghub.api_response import ResponseBuilder

    pipeline = MagicMock()
    pipeline.pipeline_id = "p1"
    pipeline.outputs = {
        "answer": "42",
        "citations": [],
        "hits": [],
        "transforms_applied": [],
        "tools_invoked": [],
        "planner_trace": None,
        "resolved_config": {"k": "v"},
    }
    response = ResponseBuilder.from_pipeline(pipeline)
    assert response.answer == "42"
    assert response.metadata["pipeline_id"] == "p1"
    assert response.metadata["resolved_config"] == {"k": "v"}


def test_response_builder_with_structured_output() -> None:
    """ResponseBuilder serialises the structured output as JSON."""

    from raghub.api_response import ResponseBuilder

    structured = MagicMock()
    structured.model_dump_json.return_value = '{"x":1}'
    structured.model_dump.return_value = {"x": 1}

    pipeline = MagicMock()
    pipeline.pipeline_id = "p2"
    pipeline.outputs = {
        "answer": "",
        "structured": structured,
        "citations": [],
        "hits": [],
        "transforms_applied": [],
        "tools_invoked": [],
        "planner_trace": None,
    }
    response = ResponseBuilder.from_pipeline(pipeline)
    assert response.answer == '{"x":1}'
    assert response.structured == {"x": 1}
    assert response.metadata["structured"] is True


def test_response_builder_with_hits() -> None:
    """ResponseBuilder converts hits into source_chunks (Hit objects)."""

    from raghub.api_response import ResponseBuilder
    from raghub.models import Chunk, Hit

    chunk = Chunk(
        id="c1",
        document_id="d1",
        version=1,
        company="acme",
        owner="alice@x.com",
        text="t",
        checksum="0" * 64,
    )
    hit = Hit(score=0.5, chunk=chunk, rank=1)

    pipeline = MagicMock()
    pipeline.pipeline_id = "p3"
    pipeline.outputs = {
        "answer": "a",
        "hits": [hit],
        "citations": [],
        "transforms_applied": [],
        "tools_invoked": [],
        "planner_trace": None,
    }
    response = ResponseBuilder.from_pipeline(pipeline)
    assert len(response.source_chunks) == 1
    assert response.source_chunks[0].chunk.id == "c1"


# ---------------------------------------------------------------------------
# api_sse helpers
# ---------------------------------------------------------------------------


def test_sse_format_serialises_dict() -> None:
    """Sse.format JSON-encodes dict payloads."""

    from raghub.api_sse import Sse

    encoded = Sse.format("event", {"x": 1})
    text = encoded.decode("utf-8")
    assert "event: event" in text
    assert '"x": 1' in text


def test_sse_format_passes_through_string() -> None:
    """Sse.format does not re-JSON a string payload."""

    from raghub.api_sse import Sse

    encoded = Sse.format("event", "hello")
    text = encoded.decode("utf-8")
    assert "data: hello" in text


def test_sse_comment() -> None:
    """Sse.comment emits an SSE comment frame."""

    from raghub.api_sse import Sse

    encoded = Sse.comment("keep-alive")
    text = encoded.decode("utf-8")
    assert text.startswith(": ")
    assert "keep-alive" in text


# ---------------------------------------------------------------------------
# api_ratelimit: TokenBucket + RateLimiterMiddleware
# ---------------------------------------------------------------------------


def test_token_bucket_admits_under_burst() -> None:
    """TokenBucket.allow admits up to ``burst`` requests immediately."""

    from raghub.api_ratelimit import TokenBucket

    bucket = TokenBucket(rate=1.0, burst=3)
    for _ in range(3):
        admitted, _ = bucket.allow("k")
        assert admitted is True
    # 4th hits the empty bucket (no time has elapsed for refill).
    admitted, _ = bucket.allow("k")
    assert admitted is False


def test_token_bucket_refills_over_time() -> None:
    """After waiting, the bucket refills and admits again."""

    import time

    from raghub.api_ratelimit import TokenBucket

    bucket = TokenBucket(rate=100.0, burst=2)
    admitted, _ = bucket.allow("k")
    assert admitted is True
    admitted, _ = bucket.allow("k")
    assert admitted is True
    admitted, _ = bucket.allow("k")
    assert admitted is False
    time.sleep(0.05)
    admitted, _ = bucket.allow("k")
    assert admitted is True


def test_token_bucket_keys_are_independent() -> None:
    """Different keys have independent buckets."""

    from raghub.api_ratelimit import TokenBucket

    bucket = TokenBucket(rate=0.1, burst=1)
    admitted, _ = bucket.allow("a")
    assert admitted is True
    admitted, _ = bucket.allow("a")
    assert admitted is False
    # Different key starts fresh.
    admitted, _ = bucket.allow("b")
    assert admitted is True


@pytest.mark.asyncio
async def test_rate_limiter_middleware_admits_http() -> None:
    """A rate-limited middleware lets through the first burst."""

    from raghub.api_ratelimit import RateLimiterMiddleware

    downstream_calls: list[str] = []

    async def _app(_scope: Any, _receive: Any, _send: Any) -> None:
        downstream_calls.append("hit")

    middleware = RateLimiterMiddleware(_app, rate=10.0, burst=2)
    scope: dict[str, Any] = {"type": "http", "client": ("1.2.3.4", 0)}
    sent: list[Any] = []

    async def _send(_msg: Any) -> None:
        sent.append(_msg)

    async def _receive() -> dict[str, Any]:
        return {"type": "http.request", "body": b"", "more_body": False}

    await middleware(scope, _receive, _send)
    await middleware(scope, _receive, _send)
    assert len(downstream_calls) == 2
    assert sent == []


@pytest.mark.asyncio
async def test_rate_limiter_middleware_short_circuits() -> None:
    """When the bucket is empty, the middleware emits 429 and skips the downstream."""

    from raghub.api_ratelimit import RateLimiterMiddleware

    downstream_calls: list[str] = []

    async def _app(_scope: Any, _receive: Any, _send: Any) -> None:
        downstream_calls.append("hit")

    middleware = RateLimiterMiddleware(_app, rate=10.0, burst=1)
    scope: dict[str, Any] = {"type": "http", "client": ("1.2.3.4", 0)}
    sent: list[Any] = []

    async def _send(msg: Any) -> None:
        sent.append(msg)

    async def _receive() -> dict[str, Any]:
        return {"type": "http.request", "body": b"", "more_body": False}

    await middleware(scope, _receive, _send)  # first request admitted
    await middleware(scope, _receive, _send)  # rate-limited
    assert len(downstream_calls) == 1
    assert any("http.response.start" in str(s) or s.get("status") == 429 for s in sent)


@pytest.mark.asyncio
async def test_rate_limiter_middleware_passes_lifespan() -> None:
    """Lifespan scopes are forwarded unchanged to the downstream app."""

    from raghub.api_ratelimit import RateLimiterMiddleware

    seen: list[str] = []

    async def _app(scope: Any, _receive: Any, _send: Any) -> None:
        seen.append(scope["type"])

    middleware = RateLimiterMiddleware(_app)
    await middleware({"type": "lifespan"}, lambda: None, lambda msg: None)
    assert seen == ["lifespan"]


@pytest.mark.asyncio
async def test_rate_limiter_middleware_passes_websocket() -> None:
    """Websocket scopes are forwarded unchanged."""

    from raghub.api_ratelimit import RateLimiterMiddleware

    seen: list[str] = []

    async def _app(scope: Any, _receive: Any, _send: Any) -> None:
        seen.append(scope["type"])

    middleware = RateLimiterMiddleware(_app)
    await middleware({"type": "websocket"}, lambda: None, lambda msg: None)
    assert seen == ["websocket"]


@pytest.mark.asyncio
async def test_rate_limiter_middleware_no_client() -> None:
    """When scope lacks ``client``, the middleware uses 'unknown' as the key."""

    from raghub.api_ratelimit import RateLimiterMiddleware

    downstream_calls: list[str] = []

    async def _app(_scope: Any, _receive: Any, _send: Any) -> None:
        downstream_calls.append("hit")

    middleware = RateLimiterMiddleware(_app, rate=10.0, burst=1)
    scope: dict[str, Any] = {"type": "http"}  # no client
    sent: list[Any] = []

    async def _send(msg: Any) -> None:
        sent.append(msg)

    async def _receive() -> dict[str, Any]:
        return {"type": "http.request", "body": b"", "more_body": False}

    await middleware(scope, _receive, _send)
    await middleware(scope, _receive, _send)
    assert len(downstream_calls) == 1


# ---------------------------------------------------------------------------
# v0.9.2 Tier 3 — Item 18: FeedbackRouter
# ---------------------------------------------------------------------------


def test_feedback_router_post_get_delete_round_trip(tmp_path) -> None:
    """POST /feedback, GET /feedback/{id}, DELETE /feedback/{id} all work."""
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from raghub.api import FeedbackRouter
    from raghub.feedback import SqliteFeedbackStore

    feedback_db = tmp_path / "feedback.db"
    store = SqliteFeedbackStore(db_path=str(feedback_db))
    store.initialize()

    class StubContainer:
        feedback_store = store

    class StubFacade:
        def __init__(self):
            self.container = StubContainer()

    app = FastAPI()
    app.include_router(FeedbackRouter().router)
    app.state.application = StubFacade()

    client = TestClient(app)

    # POST /feedback
    response = client.post(
        "/feedback",
        json={
            "session_id": "s1",
            "query_id": "q1",
            "chunk_id": "c1",
            "rating": 1,
            "comment": "great",
        },
    )
    assert response.status_code == 201, response.text
    feedback_id = response.json()["id"]

    # GET /feedback/{id}
    response = client.get(f"/feedback/{feedback_id}")
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["chunk_id"] == "c1"
    assert body["rating"] == 1

    # GET /feedback/aggregate
    response = client.get("/feedback/aggregate")
    assert response.status_code == 200, response.text
    aggregate = response.json()
    assert aggregate["positive"] == 1

    # DELETE /feedback/{id}
    response = client.delete(f"/feedback/{feedback_id}")
    assert response.status_code == 204, response.text


def test_feedback_router_aggregate_returns_counts(tmp_path) -> None:
    """GET /feedback/aggregate returns positive/negative/neutral counts."""
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from raghub.api import FeedbackRouter
    from raghub.feedback import (
        Feedback,
        Rating,
        SqliteFeedbackStore,
        new_feedback_id,
        now_utc,
    )

    store = SqliteFeedbackStore(db_path=str(tmp_path / "feedback.db"))
    store.initialize()
    for i, rating in enumerate(
        (Rating.POSITIVE, Rating.POSITIVE, Rating.NEGATIVE, Rating.NEUTRAL)
    ):
        asyncio.run(
            store.record(
                Feedback(
                    id=new_feedback_id(),
                    session_id=f"s{i}",
                    query_id="q1",
                    chunk_id="c1",
                    answer_id=None,
                    user_id=f"alice-{i}",
                    tenant_id="acme",
                    rating=rating,
                    comment=None,
                    created_at=now_utc(),
                )
            )
        )

    class StubContainer:
        feedback_store = store

    class StubFacade:
        def __init__(self):
            self.container = StubContainer()

    app = FastAPI()
    app.include_router(FeedbackRouter().router)
    app.state.application = StubFacade()

    client = TestClient(app)

    response = client.get("/feedback/aggregate")
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["positive"] == 2
    assert body["negative"] == 1
    assert body["neutral"] == 1


def test_feedback_router_503_when_store_absent() -> None:
    """503 returned when feedback_store is not configured."""
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from raghub.api import FeedbackRouter

    class StubContainer:
        feedback_store = None

    class StubFacade:
        def __init__(self):
            self.container = StubContainer()

    app = FastAPI()
    app.include_router(FeedbackRouter().router)
    app.state.application = StubFacade()

    client = TestClient(app)

    response = client.post(
        "/feedback",
        json={"session_id": "s1", "query_id": "q1", "rating": 1},
    )
    assert response.status_code == 503
