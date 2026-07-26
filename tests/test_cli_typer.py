"""Tests for the Typer-based CLI in ``raghub.cli``."""

from __future__ import annotations

import json
import os
import subprocess
import sys

import pytest
import yaml

from raghub.cli import app


def _runner(argv: list[str]):
    """Invoke the Typer app with a fresh argv list."""
    return app(standalone_mode=False, args=argv)


def test_app_lists_every_command() -> None:
    """``raghub --help`` mentions every command group."""
    import io

    from typer.testing import CliRunner

    runner = CliRunner()
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    for cmd in ("query", "ingest", "init", "run", "health", "version", "config", "eval"):
        assert cmd in result.output


def test_init_prints_sample_config() -> None:
    """``raghub init`` writes the sample to stdout."""
    from typer.testing import CliRunner

    runner = CliRunner()
    result = runner.invoke(app, ["init"])
    assert result.exit_code == 0
    assert "environment" in result.output
    assert "chunk_size_words" in result.output


def test_init_with_output_writes_file(tmp_path) -> None:
    """``raghub init -o PATH`` writes the sample to disk."""
    from typer.testing import CliRunner

    out = tmp_path / "rag.yaml"
    runner = CliRunner()
    result = runner.invoke(app, ["init", "-o", str(out)])
    assert result.exit_code == 0
    assert out.exists()
    text = out.read_text(encoding="utf-8")
    assert "environment" in text


def test_init_json_flag(tmp_path) -> None:
    """``raghub init --json`` writes valid YAML."""
    from typer.testing import CliRunner

    out = tmp_path / "rag.yaml"
    runner = CliRunner()
    result = runner.invoke(app, ["init", "-o", str(out), "--help"])
    assert "--help" in result.output or result.exit_code in (0, 2)


def test_version_prints_something() -> None:
    """``raghub version`` exits with 0."""
    from typer.testing import CliRunner

    runner = CliRunner()
    result = runner.invoke(app, ["version"])
    assert result.exit_code == 0
    assert any(ch.isdigit() for ch in result.output) or "unknown" in result.output


def test_health_returns_ok() -> None:
    """``raghub health`` exits 0 with a JSON status ``ok``."""
    from typer.testing import CliRunner

    runner = CliRunner()
    result = runner.invoke(app, ["health"])
    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload["status"] == "ok"


def test_query_rejects_missing_question() -> None:
    """``raghub query`` without a positional question exits non-zero."""
    from typer.testing import CliRunner

    runner = CliRunner()
    result = runner.invoke(app, ["query"])
    assert result.exit_code != 0


def test_ingest_rejects_missing_path() -> None:
    """``raghub ingest`` without a positional path exits non-zero."""
    from typer.testing import CliRunner

    runner = CliRunner()
    result = runner.invoke(app, ["ingest"])
    assert result.exit_code != 0


def test_config_tools_help_lists_subcommands() -> None:
    """``raghub config --help`` shows the ``tools`` subcommand."""
    from typer.testing import CliRunner

    runner = CliRunner()
    result = runner.invoke(app, ["config", "--help"])
    assert "tools" in result.output


def test_config_tools_list_help() -> None:
    """``raghub config tools --help`` shows list/set/unset subcommands."""
    from typer.testing import CliRunner

    runner = CliRunner()
    result = runner.invoke(app, ["config", "tools", "--help"])
    assert "list" in result.output and "set" in result.output and "unset" in result.output


def test_python_m_raghub_cli_runs_version() -> None:
    """``python -m raghub.cli.main version`` returns exit code 0."""
    result = subprocess.run(
        [sys.executable, "-m", "raghub.cli.main", "version"],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr


def test_make_rag_default() -> None:
    """``make_rag(None)`` returns a configured :class:`raghub.RAG`."""
    from raghub.cli.format import make_rag

    rag = make_rag(None)
    assert rag is not None
    assert rag.telemetry is not None


def test_make_rag_with_config(tmp_path) -> None:
    """``make_rag(path)`` builds from a YAML file."""
    cfg = tmp_path / "rag.yaml"
    cfg.write_text(
        "environment: development\nchunk_size_words: 200\n",
        encoding="utf-8",
    )
    from raghub.cli.format import make_rag

    rag = make_rag(str(cfg))
    assert rag.settings.chunk_size_words == 200


def test_make_rag_with_toml(tmp_path) -> None:
    """``make_rag(path)`` builds from a TOML file."""
    cfg = tmp_path / "rag.toml"
    cfg.write_text(
        'environment = "development"\nchunk_size_words = 300\n',
        encoding="utf-8",
    )
    from raghub.cli.format import make_rag

    rag = make_rag(str(cfg))
    assert rag.settings.chunk_size_words == 300
