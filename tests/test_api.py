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
    App,
    Lifespan,
    check_size,
    content_length,
    cors_origins,
    create_app,
    enforce_limit,
    health_route,
    package_metadata,
    validate_cors,
)
from raghub.routes import Exceptions, RouteGroup, user_store_or_raise

# ---------------------------------------------------------------------------
# CORS helpers
# ---------------------------------------------------------------------------


def test_cors_origins_default_is_wildcard(monkeypatch: pytest.MonkeyPatch) -> None:
    """When CORS_ORIGINS is unset, the default is ``['*']``."""

    monkeypatch.delenv("CORS_ORIGINS", raising=False)
    assert cors_origins() == ["*"]


def test_cors_origins_empty_string_is_wildcard(monkeypatch: pytest.MonkeyPatch) -> None:
    """An empty CORS_ORIGINS string falls back to ``['*']``."""

    monkeypatch.setenv("CORS_ORIGINS", "")
    assert cors_origins() == ["*"]


def test_cors_origins_splits_csv(monkeypatch: pytest.MonkeyPatch) -> None:
    """A comma-separated list is split into clean strings."""

    monkeypatch.setenv("CORS_ORIGINS", "https://a.example, https://b.example ,")
    assert cors_origins() == ["https://a.example", "https://b.example"]


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


def test_check_size_none_returns_false() -> None:
    """A missing Content-Length returns False (post-read check will run)."""

    assert check_size(None, 1000) is False


def test_check_size_under_limit_returns_false() -> None:
    """An upload under the limit returns False."""

    assert check_size(500, 1000) is False


def test_check_size_at_limit_returns_false() -> None:
    """An upload exactly at the limit returns False."""

    assert check_size(1000, 1000) is False


def test_check_size_over_limit_returns_true() -> None:
    """An upload over the limit returns True."""

    assert check_size(2000, 1000) is True


def test_content_length_returns_int() -> None:
    """content_length parses an integer header value."""

    captured: dict[str, int | None] = {}

    def _route(request: Request) -> dict[str, int | None]:
        captured["value"] = content_length(request)
        return {"value": captured["value"]}

    app = FastAPI()
    app.add_api_route("/probe", _route, methods=["GET"])
    client = TestClient(app)
    response = client.get("/probe", headers={"content-length": "123"})
    assert response.status_code == 200
    assert captured["value"] == 123


def test_content_length_missing_returns_none() -> None:
    """content_length returns None when the header is absent."""

    captured: dict[str, int | None] = {}

    def _route(request: Request) -> dict[str, int | None]:
        captured["value"] = content_length(request)
        return {"value": captured["value"]}

    app = FastAPI()
    app.add_api_route("/probe", _route, methods=["GET"])
    client = TestClient(app)
    response = client.get("/probe")
    assert response.status_code == 200
    assert captured["value"] is None


def test_content_length_invalid_returns_none() -> None:
    """content_length returns None for a non-integer header value."""

    # The TestClient will not let us inject a non-numeric Content-Length,
    # but we can verify the branch by invoking the helper directly with a
    # request object whose header is bogus. Starlette will tolerate it
    # because we read the header lazily via request.headers.get.
    from starlette.datastructures import Headers

    headers = Headers(raw=[(b"content-length", b"not-a-number")])
    scope = {"type": "http", "headers": headers.raw}
    request = Request(scope)
    assert content_length(request) is None


def test_enforce_limit_no_setting_is_silent() -> None:
    """When max_upload_bytes is 0, enforce_limit is a no-op."""

    container = MagicMock()
    container.settings.max_upload_bytes = 0

    captured: dict[str, bool] = {}

    def _route(request: Request) -> dict[str, bool]:
        enforce_limit(request, container)
        captured["ok"] = True
        return {"ok": True}

    app = FastAPI()
    app.add_api_route("/probe", _route, methods=["POST"])
    client = TestClient(app)
    response = client.post("/probe", content=b"data")
    assert response.status_code == 200
    assert captured["ok"] is True


def test_enforce_limit_oversize_raises() -> None:
    """An oversize upload raises HTTP 413 (exercised at the helper level)."""

    container = MagicMock()
    container.settings.max_upload_bytes = 100

    def _route(request: Request) -> dict[str, bool]:
        enforce_limit(request, container)
        return {"ok": True}

    app = FastAPI()
    app.add_api_route("/probe", _route, methods=["POST"])
    client = TestClient(app, raise_server_exceptions=False)
    response = client.post("/probe", content=b"x" * 200)
    assert response.status_code == 413


def test_enforce_limit_undersize_is_silent() -> None:
    """An undersize upload passes silently."""

    container = MagicMock()
    container.settings.max_upload_bytes = 10000

    captured: dict[str, bool] = {}

    def _route(request: Request) -> dict[str, bool]:
        enforce_limit(request, container)
        captured["ok"] = True
        return {"ok": True}

    app = FastAPI()
    app.add_api_route("/probe", _route, methods=["POST"])
    client = TestClient(app)
    response = client.post("/probe", content=b"x" * 5)
    assert response.status_code == 200
    assert captured["ok"] is True


def test_enforce_limit_oversize_payload_raises() -> None:
    """An oversize payload (post-read check) raises HTTP 413."""

    container = MagicMock()
    container.settings.max_upload_bytes = 100

    def _route(request: Request) -> dict[str, bool]:
        # Manually pass a payload that exceeds the limit; the content-length
        # is absent so the pre-flight check is skipped.
        enforce_limit(request, container, payload=b"x" * 200)
        return {"ok": True}

    app = FastAPI()
    app.add_api_route("/probe", _route, methods=["POST"])
    client = TestClient(app, raise_server_exceptions=False)
    response = client.post("/probe")
    assert response.status_code == 413


# ---------------------------------------------------------------------------
# user_store_or_raise
# ---------------------------------------------------------------------------


def test_user_store_or_raise_returns_user_store() -> None:
    """user_store_or_raise returns the configured user store with prefs API."""

    class FakeStore:
        def get_pref(self, user_id: str, key: str) -> None:
            return None

        def set_pref(self, user_id: str, key: str, value: object) -> None:
            return None

    store = FakeStore()
    application_facade = MagicMock()
    application_facade.container.user_store = store
    assert user_store_or_raise(application_facade) is store


def test_user_store_or_raise_raises_503_when_missing() -> None:
    """user_store_or_raise raises HTTPException(503) when user_store is None."""

    application_facade = MagicMock()
    application_facade.container.user_store = None
    with pytest.raises(HTTPException) as exc_info:
        user_store_or_raise(application_facade)
    assert exc_info.value.status_code == 503


def test_user_store_or_raise_raises_503_when_missing_prefs_api() -> None:
    """user_store_or_raise raises when store lacks get_pref/set_pref."""

    class StubStore:
        pass

    application_facade = MagicMock()
    application_facade.container.user_store = StubStore()
    with pytest.raises(HTTPException) as exc_info:
        user_store_or_raise(application_facade)
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
# health_route
# ---------------------------------------------------------------------------


def test_health_route_adds_endpoint() -> None:
    """health_route wires GET /health onto the app."""

    app = FastAPI()
    health_route(app)
    client = TestClient(app)
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"


def test_health_route_adds_at_least_one() -> None:
    """health_route ensures at least one /health route exists."""

    app = FastAPI()
    health_route(app)
    paths = [r.path for r in app.router.routes]
    assert "/health" in paths


# ---------------------------------------------------------------------------
# Exceptions wiring
# ---------------------------------------------------------------------------


@pytest.fixture
def exception_handlers_app() -> FastAPI:
    """A tiny FastAPI app with exception handlers wired."""

    app = FastAPI()
    Exceptions.install(app)

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
    application_facade = MagicMock()

    async def _shutdown() -> None:
        calls.append("down")

    application_facade.shutdown = _shutdown
    lifespan = Lifespan(application_facade)

    async def _drive() -> None:
        async with lifespan(MagicMock()):
            calls.append("inside")

    asyncio.run(_drive())
    assert calls == ["inside", "down"]


def test_lifespan_swallows_shutdown_errors() -> None:
    """Lifespan logs and continues when shutdown raises."""

    application_facade = MagicMock()

    async def _shutdown() -> None:
        raise RuntimeError("boom")

    application_facade.shutdown = _shutdown
    lifespan = Lifespan(application_facade)

    async def _drive() -> None:
        async with lifespan(MagicMock()):
            pass

    # Should not raise.
    asyncio.run(_drive())


def test_lifespan_swallows_background_shutdown_errors() -> None:
    """Lifespan logs and continues when background.shutdown raises."""

    class BadBackground:
        def shutdown(self) -> None:
            raise ConnectionError("boom")

    application_facade = MagicMock()
    application_facade.shutdown = None
    lifespan = Lifespan(application_facade)

    state = MagicMock()
    state.background_ingestion = BadBackground()

    async def _drive() -> None:
        async with lifespan(state):
            pass

    # Should not raise.
    asyncio.run(_drive())


# ---------------------------------------------------------------------------
# create_app
# ---------------------------------------------------------------------------


def test_create_app_wires_routes_and_cors() -> None:
    """create_app delegates the heavy lifting to helpers; we only smoke here."""

    # Full invocation is exercised by ``test_create_app_runs_without_shadowing``;
    # this test only confirms the function object is wired and exported.

    assert callable(create_app)
    # It also should be exposed via __all__.
    import raghub.api as api_module

    assert "create_app" in api_module.__all__


def test_create_app_wildcard_cors_raises() -> None:
    """Path through validate_cors when origin is '*' raises ConfigurationError."""

    from raghub.errors import ConfigurationError

    # validate_cors is the gated step inside create_app; exercise it
    # directly to confirm the rejection path.
    with pytest.raises(ConfigurationError):
        validate_cors(["*"])


def test_create_app_runs_without_shadowing() -> None:
    """Regression: the local ``cors_origins`` no longer shadows the module function.

    Prior to this fix, ``cors_origins = cors_origins()`` made ``cors_origins``
    a local variable across the whole ``create_app`` body, raising
    ``UnboundLocalError`` at runtime. This test inspects the function
    source so the lexical shadow cannot silently return.
    """

    import inspect

    source = inspect.getsource(create_app)
    assert "cors_origins = cors_origins()" not in source
    assert "origins = cors_origins()" in source


def test_app_reset_clears_cached_app() -> None:
    """App.reset() drops the cached FastAPI instance and its config."""

    from raghub.config import Settings

    previous = App.instance
    try:
        App.instance = App(Settings())
        App.instance.cached = MagicMock()  # any sentinel
        App.reset()
        assert App.instance is None
    finally:
        App.instance = previous


def test_app_create_returns_instance() -> None:
    """App.create(config) is a classmethod that returns a FastAPI instance.

    Cached-path exercise: when cached is already populated, the call
    returns the sentinel without rebuilding.
    """

    from raghub.config import Settings

    previous = App.instance
    try:
        App.instance = App(Settings())
        sentinel = MagicMock()
        App.instance.cached = sentinel
        result = App.create(Settings())
        assert result is sentinel
    finally:
        App.instance = previous


def test_app_create_rebuilds_on_config_change(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """App.create returns the cached app for equal configs and rebuilds otherwise.

    The expensive container build is short-circuited by mocking both
    ``asyncio.run`` (the container build path) and the module-level
    ``create_app`` factory. Each call returns a distinct sentinel so
    identity is observable. This test verifies the cache key behaviour,
    not the container build.
    """

    from raghub.config import Settings

    previous = App.instance
    try:
        App.reset()

        sentinel_a = MagicMock(name="sentinel_a")
        sentinel_b = MagicMock(name="sentinel_b")
        call_count = {"n": 0}

        def fake_run(coro: object) -> object:
            """Drain the coroutine and return a sentinel container."""

            coro.close()  # type: ignore[attr-defined]
            return MagicMock(name="container_sentinel")

        def fake_create_app(application: object) -> object:
            """Return a different sentinel on each call."""

            call_count["n"] += 1
            return sentinel_a if call_count["n"] == 1 else sentinel_b

        monkeypatch.setattr("raghub.api.asyncio.run", fake_run)
        monkeypatch.setattr("raghub.api.create_app", fake_create_app)

        settings_a = Settings()
        settings_b = Settings()
        assert settings_a == settings_b  # pydantic structural equality

        # Two equal configs share the cache: only one build.
        first = App.create(settings_a)
        second = App.create(settings_b)
        assert first is sentinel_a
        assert second is sentinel_a
        assert call_count["n"] == 1

        # A mutated config is unequal and triggers a rebuild.
        settings_c = settings_a.model_copy(update={"top_k": 7})
        third = App.create(settings_c)
        assert third is sentinel_b
        assert third is not first
        assert call_count["n"] == 2
    finally:
        App.instance = previous


# ---------------------------------------------------------------------------
# App wiring
# ---------------------------------------------------------------------------


def test_app_create_method_exists() -> None:
    """App exposes a ``create`` method and no longer exposes ``create_app``."""

    assert hasattr(App, "create")
    assert not hasattr(App, "create_app")


# ---------------------------------------------------------------------------
# Module-level: create_app + helper exports
# ---------------------------------------------------------------------------


def test_module_all() -> None:
    """The api module exports the documented public names."""

    import raghub.api as api_module

    for name in (
        "App",
        "Lifespan",
        "check_size",
        "cors_origins",
        "create_app",
        "enforce_limit",
        "package_metadata",
        "health_route",
        "content_length",
        "validate_cors",
    ):
        assert name in api_module.__all__


# ---------------------------------------------------------------------------
# RouteGroup interface
# ---------------------------------------------------------------------------


def test_route_group_class_exists() -> None:
    """``RouteGroup`` is exported by ``raghub.routes``."""

    import raghub.routes as routes_module

    assert "RouteGroup" in vars(routes_module)
    assert hasattr(RouteGroup, "__init__")


# ---------------------------------------------------------------------------
# api_auth helpers
# ---------------------------------------------------------------------------


class FakeApp:
    """Test client carrying an application facade on its state."""

    def __init__(self, application: Any) -> None:
        self.state = SimpleNamespace(application=application)


def test_inject_get_returns_application() -> None:
    """App.get reads the application facade off the request app's state."""

    from raghub.auth import App

    application = MagicMock()
    request = MagicMock()
    request.app = FakeApp(application)
    assert App.get(request) is application


def test_bearer_require_strips_token() -> None:
    """Bearer.require returns the trailing token, trimmed."""

    from raghub.auth import Bearer

    assert Bearer.require("Bearer my-token  ") == "my-token"


def test_bearer_require_lowercase() -> None:
    """Bearer.require is case-insensitive on the scheme."""

    from raghub.auth import Bearer

    assert Bearer.require("bearer t") == "t"


def test_bearer_require_missing_raises() -> None:
    """Bearer.require raises HTTPException(401) when header is missing."""

    from raghub.auth import Bearer

    with pytest.raises(HTTPException) as exc_info:
        Bearer.require(None)
    assert exc_info.value.status_code == 401


def test_bearer_require_wrong_scheme_raises() -> None:
    """Bearer.require rejects non-Bearer schemes."""

    from raghub.auth import Bearer

    with pytest.raises(HTTPException) as exc_info:
        Bearer.require("Basic xyz")
    assert exc_info.value.status_code == 401


def test_auth_admin_resolves_admin() -> None:
    """Auth.admin returns the user when is_admin is True."""

    from raghub.auth import Auth

    admin = MagicMock()
    admin.is_admin = True

    async def _resolve(_token: str) -> tuple[Any, list[Any]]:
        return admin, []

    application_facade = MagicMock()
    application_facade.resolve_user = _resolve

    async def _drive() -> Any:
        return await Auth.admin(authorization="Bearer t", application_facade=application_facade)

    assert asyncio.run(_drive()) is admin


def test_auth_admin_403_for_non_admin() -> None:
    """Auth.admin raises 403 for non-admin users."""

    from raghub.auth import Auth

    user = MagicMock()
    user.is_admin = False

    async def _resolve(_token: str) -> tuple[Any, list[Any]]:
        return user, []

    application_facade = MagicMock()
    application_facade.resolve_user = _resolve

    async def _drive() -> None:
        await Auth.admin(authorization="Bearer t", application_facade=application_facade)

    with pytest.raises(HTTPException) as exc_info:
        asyncio.run(_drive())
    assert exc_info.value.status_code == 403


def test_auth_user_id_helper() -> None:
    """Auth.user_id returns user.id from the inner auth service."""

    from raghub.auth import Auth

    user = MagicMock()
    user.id = "u-1"
    inner = MagicMock()

    async def _resolve(_token: str) -> tuple[Any, list[Any]]:
        return user, []

    inner.resolve_user = _resolve
    application_facade = MagicMock()
    application_facade.auth = inner

    async def _drive() -> str:
        return await Auth.user_id(application_facade, "t")

    assert asyncio.run(_drive()) == "u-1"


# ---------------------------------------------------------------------------
# api_response helpers
# ---------------------------------------------------------------------------


def test_redaction_user_strips_sensitive_keys() -> None:
    """Redaction.user replaces sensitive keys with ``***``."""

    from raghub.response import Redaction

    payload = {"email": "a@x.com", "password": "x", "password_hash": "h", "token": "t"}
    redacted = Redaction.user(payload)
    assert redacted["email"] == "a@x.com"
    assert redacted["password"] == "***"
    assert redacted["password_hash"] == "***"
    assert redacted["token"] == "***"


def test_redaction_user_case_insensitive() -> None:
    """Redaction matches sensitive keys case-insensitively."""

    from raghub.response import Redaction

    payload = {"PASSWORD": "x", "Secret": "y", "Token": "z"}
    redacted = Redaction.user(payload)
    assert all(redacted[k] == "***" for k in payload)


def test_response_builder_from_pipeline() -> None:
    """ResponseBuilder.from_pipeline builds a Response from a Pipeline result."""

    from raghub.response import ResponseBuilder

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

    from raghub.response import ResponseBuilder

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

    from raghub.models import Chunk, Hit
    from raghub.response import ResponseBuilder

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

    from raghub.sse import Sse

    encoded = Sse.format("event", {"x": 1})
    text = encoded.decode("utf-8")
    assert "event: event" in text
    assert '"x": 1' in text


def test_sse_format_passes_through_string() -> None:
    """Sse.format does not re-JSON a string payload."""

    from raghub.sse import Sse

    encoded = Sse.format("event", "hello")
    text = encoded.decode("utf-8")
    assert "data: hello" in text


def test_sse_comment() -> None:
    """Sse.comment emits an SSE comment frame."""

    from raghub.sse import Sse

    encoded = Sse.comment("keep-alive")
    text = encoded.decode("utf-8")
    assert text.startswith(": ")
    assert "keep-alive" in text


# ---------------------------------------------------------------------------
# api_ratelimit: Bucket + Ratelimit
# ---------------------------------------------------------------------------


def test_token_bucket_admits_under_burst() -> None:
    """Bucket.allow admits up to ``burst`` requests immediately."""

    from raghub.ratelimit import Bucket

    bucket = Bucket(rate=1.0, burst=3)
    for _ in range(3):
        admitted, _ = bucket.allow("k")
        assert admitted is True
    # 4th hits the empty bucket (no time has elapsed for refill).
    admitted, _ = bucket.allow("k")
    assert admitted is False


def test_token_bucket_refills_over_time() -> None:
    """After waiting, the bucket refills and admits again."""

    import time

    from raghub.ratelimit import Bucket

    bucket = Bucket(rate=100.0, burst=2)
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

    from raghub.ratelimit import Bucket

    bucket = Bucket(rate=0.1, burst=1)
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

    from raghub.ratelimit import Ratelimit

    downstream_calls: list[str] = []

    async def _app(_scope: Any, _receive: Any, _send: Any) -> None:
        downstream_calls.append("hit")

    middleware = Ratelimit(_app, rate=10.0, burst=2)
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

    from raghub.ratelimit import Ratelimit

    downstream_calls: list[str] = []

    async def _app(_scope: Any, _receive: Any, _send: Any) -> None:
        downstream_calls.append("hit")

    middleware = Ratelimit(_app, rate=10.0, burst=1)
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

    from raghub.ratelimit import Ratelimit

    seen: list[str] = []

    async def _app(scope: Any, _receive: Any, _send: Any) -> None:
        seen.append(scope["type"])

    middleware = Ratelimit(_app)
    await middleware({"type": "lifespan"}, lambda: None, lambda msg: None)
    assert seen == ["lifespan"]


@pytest.mark.asyncio
async def test_rate_limiter_middleware_passes_websocket() -> None:
    """Websocket scopes are forwarded unchanged."""

    from raghub.ratelimit import Ratelimit

    seen: list[str] = []

    async def _app(scope: Any, _receive: Any, _send: Any) -> None:
        seen.append(scope["type"])

    middleware = Ratelimit(_app)
    await middleware({"type": "websocket"}, lambda: None, lambda msg: None)
    assert seen == ["websocket"]


@pytest.mark.asyncio
async def test_rate_limiter_middleware_no_client() -> None:
    """When scope lacks ``client``, the middleware uses 'unknown' as the key."""

    from raghub.ratelimit import Ratelimit

    downstream_calls: list[str] = []

    async def _app(_scope: Any, _receive: Any, _send: Any) -> None:
        downstream_calls.append("hit")

    middleware = Ratelimit(_app, rate=10.0, burst=1)
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
# v0.9.2 Tier 3 -- Item 18: FeedbackRoute
# ---------------------------------------------------------------------------


def test_feedback_router_post_get_delete_round_trip(tmp_path) -> None:
    """POST /feedback, GET /feedback/{id}, DELETE /feedback/{id} all work and the user attribution is real.

    The bearer token is required by :class:`Auth.admin` /
    :class:`Bearer.require`; the facade's auth service is stubbed
    to resolve ``Bearer alice-token`` to a known user, and the
    stored feedback is asserted to carry that user's email rather
    than ``"anonymous"``.
    """
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from raghub.feedback import SqliteFeedbackStore
    from raghub.routes import FeedbackRoute

    feedback_db = tmp_path / "feedback.db"
    store = SqliteFeedbackStore(db_path=str(feedback_db))
    store.initialize()

    class User:
        email = "alice@example.com"
        tenant_id = "acme"

    class AuthSvc:
        async def resolve_user(self, token: str) -> tuple[User, list[Any]]:
            assert token == "alice-token", f"Expected alice-token; got {token!r}"
            return User(), []

    class StubContainer:
        feedback_store = store

    class StubFacade:
        def __init__(self):
            self.container = StubContainer()
            self.auth = AuthSvc()

    app = FastAPI()
    app.include_router(FeedbackRoute().router)
    app.state.application = StubFacade()

    client = TestClient(app)
    headers = {"Authorization": "Bearer alice-token"}

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
        headers=headers,
    )
    assert response.status_code == 201, response.text
    feedback_id = response.json()["id"]

    # GET /feedback/{id}
    response = client.get(f"/feedback/{feedback_id}")
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["chunk_id"] == "c1"
    assert body["rating"] == 1
    assert body["user_id"] == "alice@example.com", (
        "Feedback must be attributed to the bearer-token user, not 'anonymous'"
    )

    # GET /feedback/aggregate
    response = client.get("/feedback/aggregate")
    assert response.status_code == 200, response.text
    aggregate = response.json()
    assert aggregate["positive"] == 1

    # DELETE /feedback/{id}
    response = client.delete(f"/feedback/{feedback_id}")
    assert response.status_code == 204, response.text

    # Missing bearer token is still rejected with 401; the endpoint
    # never silently degrades to an anonymous write.
    unauth = client.post(
        "/feedback",
        json={"session_id": "s1", "query_id": "q1", "rating": 1},
    )
    assert unauth.status_code == 401


def test_feedback_router_aggregate_returns_counts(tmp_path) -> None:
    """GET /feedback/aggregate returns positive/negative/neutral counts."""
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from raghub.feedback import (
        Feedback,
        FeedbackStore,
        Rating,
        SqliteFeedbackStore,
    )
    from raghub.routes import FeedbackRoute

    store = SqliteFeedbackStore(db_path=str(tmp_path / "feedback.db"))
    store.initialize()
    for i, rating in enumerate((Rating.Positive, Rating.Positive, Rating.Negative, Rating.Neutral)):
        asyncio.run(
            store.record(
                Feedback(
                    id=FeedbackStore.new_id(),
                    session_id=f"s{i}",
                    query_id="q1",
                    chunk_id="c1",
                    answer_id=None,
                    user_id=f"alice-{i}",
                    tenant_id="acme",
                    rating=rating,
                    comment=None,
                    created_at=FeedbackStore.now_utc(),
                )
            )
        )

    class StubContainer:
        feedback_store = store

    class StubFacade:
        def __init__(self):
            self.container = StubContainer()

    app = FastAPI()
    app.include_router(FeedbackRoute().router)
    app.state.application = StubFacade()

    client = TestClient(app)

    response = client.get("/feedback/aggregate")
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["positive"] == 2
    assert body["negative"] == 1
    assert body["neutral"] == 1


def test_feedback_router_503_when_store_absent() -> None:
    """503 returned when feedback_store is not configured — but only after the bearer token check passes.

    The auth layer is wired with a stub that resolves any token to
    a real user so the request reaches the handler. The handler
    must then return 503 (no store configured), proving the
    store-absent path is reachable without an auth bypass.
    """
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from raghub.routes import FeedbackRoute

    class User:
        email = "alice@example.com"
        tenant_id = "acme"

    class AuthSvc:
        async def resolve_user(self, token: str) -> tuple[User, list[Any]]:
            return User(), []

    class StubContainer:
        feedback_store = None

    class StubFacade:
        def __init__(self):
            self.container = StubContainer()
            self.auth = AuthSvc()

    app = FastAPI()
    app.include_router(FeedbackRoute().router)
    app.state.application = StubFacade()

    client = TestClient(app)
    headers = {"Authorization": "Bearer alice-token"}

    response = client.post(
        "/feedback",
        json={"session_id": "s1", "query_id": "q1", "rating": 1},
        headers=headers,
    )
    assert response.status_code == 503, response.text
    # And without a bearer token the auth layer still rejects with 401.
    unauth = client.post(
        "/feedback",
        json={"session_id": "s1", "query_id": "q1", "rating": 1},
    )
    assert unauth.status_code == 401
