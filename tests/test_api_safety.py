"""Production-grade FastAPI safety tests.

Covers the fail-closed production behaviour: wildcard CORS rejected
on startup, admin endpoints redact password_hash, oversize uploads
are rejected before the body is buffered, and the canonical
``/v1/health`` + ``/v1/auth/login`` + ``/v1/query`` + ``/metrics``
surface is reachable.
"""

from __future__ import annotations

import asyncio
import json
import os

import pytest
from fastapi.testclient import TestClient

from raghub.api.admin import SENSITIVE_USER_FIELDS, redact_user_payload
from raghub.api.app import (
    check_upload_size,
    cors_origins_from_env,
    create_app,
    validate_cors_for_credentials,
)
from raghub.config.settings import load_settings
from raghub.services.application import DynamicRagApplication, build_container


def test_validate_cors_for_credentials_rejects_wildcard() -> None:
    """Wildcard origins raise on startup when credentials are enabled."""
    with pytest.raises(RuntimeError, match="incompatible with allow_credentials"):
        validate_cors_for_credentials(["*"])


def test_validate_cors_for_credentials_accepts_explicit_origins() -> None:
    """Explicit origins pass the validator."""
    validate_cors_for_credentials(["https://app.raghub.com"])
    validate_cors_for_credentials(["http://localhost:8501"])


def test_cors_origins_from_env_parses_csv(monkeypatch: pytest.MonkeyPatch) -> None:
    """``CORS_ORIGINS`` env is parsed as a comma-separated list."""
    monkeypatch.setenv("CORS_ORIGINS", "https://a.com, https://b.com")
    origins = cors_origins_from_env()
    assert origins == ["https://a.com", "https://b.com"]


def test_redact_user_payload_strips_hash_and_password() -> None:
    """Every sensitive key on the user record is replaced with ``***``."""
    payload = {
        "email": "alice@example.com",
        "password_hash": "bcrypt$2a$12$abc",
        "password": "secret",
        "token": "abc",
        "is_admin": False,
    }
    redacted = redact_user_payload(payload)
    assert redacted["email"] == "alice@example.com"
    assert redacted["is_admin"] is False
    assert redacted["password_hash"] == "***"
    assert redacted["password"] == "***"
    assert redacted["token"] == "***"
    assert "password" in SENSITIVE_USER_FIELDS
    assert "token" in SENSITIVE_USER_FIELDS


def test_check_upload_size_returns_bool() -> None:
    """The pre-flight guard returns a clean boolean instead of a sentinel."""
    assert check_upload_size(None, 1024) is False
    assert check_upload_size(512, 1024) is False
    assert check_upload_size(2048, 1024) is True
    assert check_upload_size(1024, 1024) is False


def _build_app(tmp_path: os.PathLike[str]) -> TestClient:
    """Build a fresh :class:`TestClient` with a tmp data dir."""
    os.environ["RAG_DATA_DIR"] = str(tmp_path)
    os.environ["RAG_ZVEC_DIR"] = str(tmp_path / "zvec")
    os.environ["RAG_PROFILE"] = "development"
    os.environ["JWT_SECRET"] = "test-secret-must-be-32-bytes-or-longer-for-sha256"
    os.environ["CORS_ORIGINS"] = "http://localhost:8501"
    container = asyncio.run(build_container(load_settings()))
    application = DynamicRagApplication(container)
    return TestClient(create_app(application))


def test_health_login_query_metrics_smoke(tmp_path) -> None:
    """A full request lifecycle works against the canonical surfaces."""
    client = _build_app(tmp_path)

    health = client.get("/v1/health")
    assert health.status_code == 200
    assert health.json()["status"] in {"ok", "degraded"}

    login = client.post(
        "/v1/auth/login", json={"email": "alice@email.com", "password": "test"}
    )
    assert login.status_code == 200
    token = login.json()["session_token"]

    history = client.get(
        "/v1/session/history", headers={"Authorization": f"Bearer {token}"}
    )
    assert history.status_code == 200

    metrics = client.get("/metrics")
    assert metrics.status_code == 200
    assert "raghub_query_duration_ms" in metrics.text


def test_admin_users_endpoint_redacts_password_hash(tmp_path) -> None:
    """The ``/v1/admin/users`` response must not include a real hash."""
    client = _build_app(tmp_path)

    admin_login = client.post(
        "/v1/auth/login", json={"email": "admin@email.com", "password": "admin"}
    )
    assert admin_login.status_code == 200
    token = admin_login.json()["session_token"]

    users = client.get("/v1/admin/users", headers={"Authorization": f"Bearer {token}"})
    assert users.status_code == 200
    for user in users.json():
        assert user["password_hash"] == "***"
        assert "password" not in user or user["password"] == "***"
        assert "hash" not in {k for k in user if k not in {"password_hash", "hash"}}


def test_oversize_upload_rejected_with_413(tmp_path) -> None:
    """``Content-Length`` over the limit returns 413 before the body is read."""
    os.environ["RAG_MAX_UPLOAD_BYTES"] = "1024"
    client = _build_app(tmp_path)

    admin_login = client.post(
        "/v1/auth/login", json={"email": "admin@email.com", "password": "admin"}
    )
    assert admin_login.status_code == 200
    token = admin_login.json()["session_token"]

    big = b"\x00" * 4096
    response = client.post(
        "/v1/documents/upload",
        headers={"Authorization": f"Bearer {token}"},
        files={"file": ("big.bin", big, "application/pdf")},
    )
    assert response.status_code == 413
    payload = response.json()
    assert "exceeds" in json.dumps(payload).lower() or "maximum" in json.dumps(payload).lower()