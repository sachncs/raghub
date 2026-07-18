"""Tests for the legacy dependency-injection container and FastAPI surface.

The legacy ``DynamicRagApplication`` and ``build_application`` paths
are exercised by the integration tests in ``test_platform.py`` (which
boot the full app and drive it through the HTTP lifecycle). This
module covers the smaller, deterministic surfaces that can be unit
tested without spawning the full async stack.
"""

from __future__ import annotations

import os

import pytest

from raghub.config.settings import AppSettings, load_settings
from raghub.core.container import build_application


def test_load_settings_default_profile() -> None:
    """``load_settings()`` returns a populated :class:`AppSettings`."""
    settings = load_settings()
    assert isinstance(settings, AppSettings)
    assert settings.environment == "development"


def test_load_settings_with_profile() -> None:
    """``load_settings("production")`` selects the production profile."""
    os.environ.setdefault("JWT_SECRET", "test-secret")
    settings = load_settings("production")
    assert settings.environment == "production"
    assert settings.allow_passwordless_login is False


def test_build_application_helper_is_callable() -> None:
    """``build_application`` is an async coroutine function."""
    import inspect

    assert inspect.iscoroutinefunction(build_application)


def test_build_container_raises_when_jwt_secret_missing() -> None:
    """``build_container`` raises when ``settings.jwt_secret`` is empty."""
    from raghub.services.application import build_container

    settings = AppSettings(jwt_secret="", environment="production")
    with pytest.raises(RuntimeError):
        import asyncio

        asyncio.run(build_container(settings))


def test_app_factory_creates_fastapi_instance() -> None:
    """``create_app`` returns a :class:`fastapi.FastAPI` with the expected routes.

    The versioned router is mounted under ``/v1`` and the unversioned
    ``/health`` liveness probe is mounted on the app directly. The
    versioned routes are reachable through the TestClient even though
    they show up as ``_IncludedRouter`` placeholders in ``app.routes``
    before the lifespan iterates them.
    """
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from raghub.api.app import create_app
    from raghub.api.dependencies import get_application

    class _MetricsStub:
        def register_app(self, app: object) -> None:
            from fastapi import FastAPI

            if isinstance(app, FastAPI):

                @app.get("/metrics")
                async def metrics_stub() -> dict[str, str]:
                    return {"status": "stub"}

    class _ContainerStub:
        metrics = _MetricsStub()

    class _StubApp:
        container = _ContainerStub()

        def health(self):
            return {"status": "ok"}

    app = create_app(_StubApp())
    app.dependency_overrides[get_application] = lambda: _StubApp()
    assert isinstance(app, FastAPI)
    client = TestClient(app)
    # Both the versioned ``/v1/health`` and the unversioned ``/health``
    # probe are reachable.
    assert client.get("/v1/health").status_code == 200
    assert client.get("/health").status_code == 200
    # The Prometheus ``/metrics`` endpoint is registered when the
    # application container exposes a metrics sink.
    assert client.get("/metrics").status_code == 200


def test_health_route_returns_payload() -> None:
    """``GET /health`` returns the application's health dict."""
    from fastapi.testclient import TestClient

    from raghub.api.app import create_app
    from raghub.api.dependencies import get_application

    class _MetricsStub:
        def register_app(self, _app: object) -> None:
            return None

    class _ContainerStub:
        metrics = _MetricsStub()

    class _StubApp:
        container = _ContainerStub()

        def health(self):
            return {"status": "ok", "vector_store": "InMemoryVectorStore"}

    app = create_app(_StubApp())
    app.dependency_overrides[get_application] = lambda: _StubApp()
    client = TestClient(app)
    response = client.get("/v1/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok", "vector_store": "InMemoryVectorStore"}
    # ``/health`` (unversioned) is the liveness probe used by orchestrators.
    assert client.get("/health").status_code == 200
