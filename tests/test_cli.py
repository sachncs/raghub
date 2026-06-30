"""Tests for the public CLI surface (in-process)."""

from __future__ import annotations

import argparse
import io
import json
from contextlib import redirect_stdout
from pathlib import Path

import pytest

from raghub.cli import (
    eval_cmd,
    ingest_cmd,
    init_cmd,
    main,
    query_cmd,
    system,
)
from raghub.cli.common import (
    load_settings_or_path,
    print_json,
    run_async,
)


def _make_parser() -> argparse.ArgumentParser:
    """Build a fresh top-level parser for tests."""
    return main.build_parser()


def _invoke(command: str, *args: str) -> tuple[int, str]:
    """Run ``raghub <command> ...`` and capture stdout.

    Args:
        command: Subcommand name.
        *args: Remaining arguments.

    Returns:
        A ``(returncode, stdout)`` tuple.
    """
    parser = _make_parser()
    ns = parser.parse_args([command, *args])
    handler = getattr(ns, "handler", None)
    assert handler is not None
    buf = io.StringIO()
    with redirect_stdout(buf):
        rc = handler(ns)
    return rc, buf.getvalue()


def test_help_lists_all_subcommands() -> None:
    """``--help`` shows every subcommand."""
    parser = _make_parser()
    help_text = parser.format_help()
    for cmd in ("init", "ingest", "query", "eval", "health", "version"):
        assert cmd in help_text


def test_init_prints_sample_when_no_output() -> None:
    """``init`` with no ``-o`` writes to stdout."""
    rc, out = _invoke("init")
    assert rc == 0
    assert "environment" in out
    assert "chunk_size_words" in out


def test_init_writes_sample_to_file(tmp_path: Path) -> None:
    """``init -o PATH`` writes the sample to disk."""
    out = tmp_path / "rag.yaml"
    rc, stdout = _invoke("init", "-o", str(out))
    assert rc == 0
    assert out.exists()
    content = out.read_text(encoding="utf-8")
    assert "environment" in content
    assert "Wrote" in stdout or "Wrote" in str(out)


def test_health_prints_json() -> None:
    """``health`` prints a JSON status dict."""
    rc, out = _invoke("health")
    assert rc == 0
    payload = json.loads(out)
    assert payload["status"] == "ok"
    assert "vector_store" in payload


def test_version_prints_something() -> None:
    """``version`` exits 0 (the value is environment-dependent)."""
    rc, _ = _invoke("version")
    assert rc == 0


def test_ingest_query_round_trip(tmp_path: Path) -> None:
    """Ingest a file, then query it via the CLI."""
    doc = tmp_path / "doc.txt"
    doc.write_text("Revenue grew 25% YoY in 2024.", encoding="utf-8")
    cfg = tmp_path / "rag.yaml"
    cfg.write_text(
        f"environment: development\n"
        f"data_dir: {tmp_path / 'data'}\n"
        f"chunk_size_words: 200\n"
        f"chunk_overlap_words: 10\n"
        f"embedding_model: hashing-bge\n"
        f"llm_model: heuristic-llm\n",
        encoding="utf-8",
    )
    rc, out = _invoke("ingest", "--config", str(cfg), str(doc))
    assert rc == 0, out
    rc, out = _invoke("query", "--config", str(cfg), "revenue")
    assert rc == 0, out
    payload = json.loads(out)
    assert "answer" in payload


def test_load_settings_or_path_uses_active_profile() -> None:
    """``load_settings_or_path(None)`` reads the active profile."""
    settings = load_settings_or_path(None)
    assert settings.environment == "development"


def test_load_settings_or_path_reads_yaml(tmp_path: Path) -> None:
    """``load_settings_or_path(path)`` parses the YAML file."""
    cfg = tmp_path / "rag.yaml"
    cfg.write_text(
        "environment: development\nchunk_size_words: 200\n",
        encoding="utf-8",
    )
    settings = load_settings_or_path(str(cfg))
    assert settings.chunk_size_words == 200


def test_load_settings_or_path_reads_toml(tmp_path: Path) -> None:
    """``load_settings_or_path(path)`` parses a TOML file when the extension is .toml."""
    pytest.importorskip("tomllib")
    cfg = tmp_path / "rag.toml"
    cfg.write_text(
        'environment = "development"\nchunk_size_words = 200\n',
        encoding="utf-8",
    )
    settings = load_settings_or_path(str(cfg))
    assert settings.chunk_size_words == 200


def test_run_async_executes_coroutine() -> None:
    """``run_async`` runs a coroutine to completion."""

    async def _coro() -> int:
        return 42

    assert run_async(_coro()) == 42


def test_print_json_emits_compact_payload() -> None:
    """``print_json`` writes valid JSON to stdout."""
    buf = io.StringIO()
    with redirect_stdout(buf):
        print_json({"a": 1, "b": [1, 2]})
    assert json.loads(buf.getvalue()) == {"a": 1, "b": [1, 2]}


def test_init_cmd_add_parser_registers() -> None:
    """``init_cmd.add_parser`` registers the subcommand."""
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command")
    init_cmd.add_parser(sub)
    ns = parser.parse_args(["init"])
    assert ns.command == "init"


def test_ingest_cmd_add_parser_registers() -> None:
    """``ingest_cmd.add_parser`` registers the subcommand."""
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command")
    ingest_cmd.add_parser(sub)
    ns = parser.parse_args(["ingest", "doc.pdf"])
    assert ns.command == "ingest"
    assert ns.path == "doc.pdf"


def test_query_cmd_add_parser_registers() -> None:
    """``query_cmd.add_parser`` registers the subcommand."""
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command")
    query_cmd.add_parser(sub)
    ns = parser.parse_args(["query", "What?"])
    assert ns.command == "query"
    assert ns.question == "What?"


def test_eval_cmd_add_parser_registers() -> None:
    """``eval_cmd.add_parser`` registers the subcommand."""
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command")
    eval_cmd.add_parser(sub)
    ns = parser.parse_args(["eval", "financebench"])
    assert ns.command == "eval"
    assert ns.benchmark == "financebench"


def test_system_add_parser_registers() -> None:
    """``system.add_parser`` registers the subcommands."""
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command")
    system.add_parser(sub)
    health_ns = parser.parse_args(["health"])
    assert health_ns.command == "health"
    version_ns = parser.parse_args(["version"])
    assert version_ns.command == "version"
