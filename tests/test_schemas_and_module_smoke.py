"""Schema constant tests."""

from __future__ import annotations

import subprocess
import sys


def test_sqlite_schema_string_contains_required_tables() -> None:
    """The bundled schema declares every table the stores expect."""
    from raghub.storage.sqlite_schema import SQLITE_SCHEMA

    assert "CREATE TABLE IF NOT EXISTS documents" in SQLITE_SCHEMA
    assert "CREATE TABLE IF NOT EXISTS chunks" in SQLITE_SCHEMA
    assert "CREATE TABLE IF NOT EXISTS sessions" in SQLITE_SCHEMA
    assert "CREATE TABLE IF NOT EXISTS users" in SQLITE_SCHEMA


def test_cli_module_invocation_returns_exit_code() -> None:
    """``python -m raghub.cli version`` exits with status 0."""
    result = subprocess.run(
        [sys.executable, "-m", "raghub.cli", "version"],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr
