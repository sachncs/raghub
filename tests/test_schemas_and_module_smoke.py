"""Smoke tests for re-export modules and schema constants."""

from __future__ import annotations

import subprocess
import sys


def test_schemas_module_reexports_canonical_models() -> None:
    """raghub.api.schemas re-exports every public model class."""
    from raghub.api import schemas

    expected = {
        "AuthLoginRequest",
        "AuthLoginResponse",
        "BatchIngestResponse",
        "DocumentUploadResponse",
        "QueryRequest",
        "QueryResponse",
    }
    assert expected.issubset(set(schemas.__all__))
    for name in expected:
        assert hasattr(schemas, name), f"schemas is missing {name}"


def test_schemas_classes_match_models_api() -> None:
    """The re-exported classes are the exact objects in raghub.models.api."""
    from raghub.api import schemas
    from raghub.models import api as api_models

    for name in schemas.__all__:
        assert getattr(schemas, name) is getattr(api_models, name)


def test_sqlite_schema_string_contains_required_tables() -> None:
    """The bundled schema declares every table the stores expect."""
    from raghub.storage.sqlite_schema import SQLITE_SCHEMA

    assert "CREATE TABLE IF NOT EXISTS documents" in SQLITE_SCHEMA
    assert "CREATE TABLE IF NOT EXISTS chunks" in SQLITE_SCHEMA
    assert "CREATE TABLE IF NOT EXISTS sessions" in SQLITE_SCHEMA
    assert "CREATE TABLE IF NOT EXISTS users" in SQLITE_SCHEMA


def test_raghub_package_exposes_public_api() -> None:
    """Top-level ``raghub`` package surfaces the public facade."""
    import raghub

    assert hasattr(raghub, "RAG") or hasattr(raghub, "__all__")


def test_cli_module_invocation_returns_exit_code() -> None:
    """``python -m raghub.cli version`` exits with status 0."""
    result = subprocess.run(
        [sys.executable, "-m", "raghub.cli", "version"],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr