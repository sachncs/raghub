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
    """``create_app`` returns a :class:`fastapi.FastAPI` with the expected routes."""
    from fastapi import FastAPI

    from raghub.api.app import create_app
    from raghub.api.dependencies import get_application

    class _StubApp:
        container = type("C", (), {})()

        def health(self):
            return {"status": "ok"}

    app = create_app(_StubApp())
    app.dependency_overrides[get_application] = lambda: _StubApp()
    assert isinstance(app, FastAPI)
    assert "/v1/health" in {route.path for route in app.routes if hasattr(route, "path")}


def test_health_route_returns_payload() -> None:
    """``GET /health`` returns the application's health dict."""
    from fastapi.testclient import TestClient

    from raghub.api.app import create_app
    from raghub.api.dependencies import get_application

    class _StubApp:
        container = type("C", (), {})()

        def health(self):
            return {"status": "ok", "vector_store": "InMemoryVectorStore"}

    app = create_app(_StubApp())
    app.dependency_overrides[get_application] = lambda: _StubApp()
    client = TestClient(app)
    response = client.get("/v1/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok", "vector_store": "InMemoryVectorStore"}
