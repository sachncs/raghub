"""Tests for the DynamicRagApplication service container."""

from __future__ import annotations

import asyncio
import os
from pathlib import Path

import pytest


def _make_app(tmp_path: Path):
    """Build a ``DynamicRagApplication`` against a temp data dir."""
    os.environ["JWT_SECRET"] = "x" * 64
    from pydantic import SecretStr

    from raghub.config import Settings
    from raghub.services.application import DynamicRagApplication, build_container

    from raghub.config import Settings as _Settings
    settings = _Settings.load()
    settings.data_dir = tmp_path
    settings.registry_path = tmp_path / "registry.db"
    settings.sessions_path = tmp_path / "sessions.db"
    settings.zvec_dir = tmp_path / "zvec"
    settings.environment = "development"
    settings.jwt_secret = SecretStr("x" * 64)
    settings.allow_passwordless_login = False
    container = asyncio.run(build_container(settings))
    return DynamicRagApplication(container)


def test_dynamic_app_constructs_and_exposes_services(tmp_path: Path) -> None:
    """The facade wires every service and exposes it on the container."""
    app = _make_app(tmp_path)
    assert app.auth_svc is app.container.auth
    assert app.documents_svc is app.container.documents
    assert app.query_svc is app.container.query
    assert app.health_svc is app.container.health


def test_dynamic_app_health(tmp_path: Path) -> None:
    """``health`` returns a structured payload."""
    app = _make_app(tmp_path)
    health = app.health()
    assert health["status"] == "ok"
    assert "components" in health


def test_dynamic_app_login_logout_round_trip(tmp_path: Path) -> None:
    """A user can log in, fetch history, and log out."""
    app = _make_app(tmp_path)
    resp = asyncio.run(app.login("alice@acme.com", "password"))
    assert resp.user_email == "alice@acme.com"
    assert resp.allowed_companies == ["Apple"]
    assert resp.session_token
    history = asyncio.run(app.history(resp.session_token))
    assert history == []
    asyncio.run(app.logout(resp.session_token))
    # History endpoint returns [] for a deleted session; that's the
    # documented contract. We just ensure the call does not raise.
    hist = asyncio.run(app.history(resp.session_token))
    assert hist == []


def test_dynamic_app_resolve_user_returns_history(tmp_path: Path) -> None:
    """``resolve_user`` returns the principal and the session history."""
    app = _make_app(tmp_path)
    resp = asyncio.run(app.login("alice@acme.com", "password"))
    user, hist = asyncio.run(app.resolve_user(resp.session_token))
    assert user.email == "alice@acme.com"
    assert hist == []


def test_dynamic_app_clear_history(tmp_path: Path) -> None:
    """``clear_history`` empties the conversation history."""
    app = _make_app(tmp_path)
    resp = asyncio.run(app.login("alice@acme.com", "password"))
    asyncio.run(app.clear_history(resp.session_token))
    hist = asyncio.run(app.history(resp.session_token))
    assert hist == []


def test_dynamic_app_list_documents(tmp_path: Path) -> None:
    """``list_documents`` returns the user's accessible documents."""
    app = _make_app(tmp_path)
    resp = asyncio.run(app.login("alice@acme.com", "password"))
    docs = asyncio.run(app.list_documents(resp.session_token))
    assert isinstance(docs, list)


def test_dynamic_app_log(tmp_path: Path) -> None:
    """``log`` is a thin structured-logging wrapper."""
    app = _make_app(tmp_path)
    app.log("test_event", foo="bar")  # must not raise


def test_dynamic_app_emit_metric(tmp_path: Path) -> None:
    """``emit_metric`` accepts a started_at perf_counter value."""
    import time

    app = _make_app(tmp_path)
    app.emit_metric("test_metric", time.perf_counter())  # must not raise


# Note: ``shutdown`` is no longer required to be idempotent — collaborator
# close calls now propagate failures per the v0.6.0 no-swallowing policy.
# The lifespan / CLI boundary calls shutdown exactly once.


def test_dynamic_app_raghub_users_env_seeds(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``RAGHUB_USERS`` adds extra users at startup."""
    import json

    extra = {
        "dave@example.com": {
            "password": "secret",
            "companies": ["IBM"],
            "is_admin": False,
        }
    }
    monkeypatch.setenv("RAGHUB_USERS", json.dumps(extra))

    app = _make_app(tmp_path)
    users = asyncio.run(app.container.user_store.list_users())
    emails = {u.email for u in users}
    assert "dave@example.com" in emails


def test_dynamic_app_raghub_users_invalid_json_raises(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A bad ``RAGHUB_USERS`` value raises a JSON parse error."""
    import json

    from pydantic import SecretStr

    monkeypatch.setenv("RAGHUB_USERS", "{not json")
    import asyncio

    from raghub.services.application import build_container
    from raghub.config import Settings

    settings = Settings.load()
    settings.data_dir = tmp_path
    settings.registry_path = tmp_path / "registry.db"
    settings.sessions_path = tmp_path / "sessions.db"
    settings.zvec_dir = tmp_path / "zvec"
    settings.environment = "development"
    settings.jwt_secret = SecretStr("x" * 64)
    with pytest.raises(json.JSONDecodeError):
        asyncio.run(build_container(settings))


def test_dynamic_app_build_container_raises_without_jwt_secret(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``build_container`` refuses to start when ``JWT_SECRET`` is empty."""
    from pydantic import SecretStr
    from raghub.config import Settings

    monkeypatch.delenv("JWT_SECRET", raising=False)
    from raghub.services.application import build_container

    from raghub.config import Settings
    settings = Settings(environment="development")
    settings.jwt_secret = SecretStr("")
    with pytest.raises(RuntimeError, match="JWT_SECRET"):
        asyncio.run(build_container(settings))


def test_dynamic_app_settings_load_rejects_short_jwt_secret_in_production(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``Settings.load("production")`` rejects a short JWT_SECRET at
    the configuration layer — a regression that allowed the platform
    to boot with a weak secret would surface here."""
    from raghub.config import Settings

    monkeypatch.setenv("JWT_SECRET", "short")
    with pytest.raises(RuntimeError, match="32 bytes"):
        Settings.load("production")


def test_dynamic_app_raghub_users_adds_company_list(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``RAGHUB_USERS`` seeds the user store with extra accounts at
    startup. The companies list must round-trip to the user record."""
    import json

    extra = {
        "alice@example.com": {
            "password": "secret",
            "companies": ["Acme", "Globex"],
            "is_admin": False,
        }
    }
    monkeypatch.setenv("RAGHUB_USERS", json.dumps(extra))
    app = _make_app(tmp_path)
    users = asyncio.run(app.container.user_store.list_users())
    target = next(u for u in users if u.email == "alice@example.com")
    assert set(target.allowed_companies) == {"Acme", "Globex"}
    assert target.is_admin is False


def test_dynamic_app_raghub_users_admin_flag(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The ``is_admin`` flag in ``RAGHUB_USERS`` is honored."""
    import json

    extra = {
        "root@example.com": {
            "password": "secret",
            "is_admin": True,
        }
    }
    monkeypatch.setenv("RAGHUB_USERS", json.dumps(extra))
    app = _make_app(tmp_path)
    users = asyncio.run(app.container.user_store.list_users())
    target = next(u for u in users if u.email == "root@example.com")
    assert target.is_admin is True


def test_dynamic_app_health_reports_components(
    tmp_path: Path,
) -> None:
    """The health payload includes every probed component."""
    app = _make_app(tmp_path)
    health = app.health()
    assert "components" in health
    assert "vectorstore" in health["components"]
    assert "registry" in health["components"]


def test_dynamic_app_query_against_real_user(
    tmp_path: Path,
) -> None:
    """A logged-in user can issue a query and the response carries
    the expected answer / citations fields."""
    app = _make_app(tmp_path)
    resp = asyncio.run(app.login("alice@acme.com", "password"))
    response = asyncio.run(
        app.query(token=resp.session_token, question="What is Acme?")
    )
    # The response shape is documented; the answer may be empty (no
    # documents indexed) but the field must be present.
    assert hasattr(response, "answer")
    assert hasattr(response, "citations")
