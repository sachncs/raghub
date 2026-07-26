"""Tests for the top-level ``raghub.cli.main`` entrypoint."""
from __future__ import annotations

import argparse
import io
import json
import sys
from contextlib import redirect_stdout
from pathlib import Path

import pytest

from raghub.cli.main import (
    CLI_LIMITER,
    RATE_LIMIT_EXEMPT_COMMANDS,
    build_parser,
    main,
)
from raghub.cli.rate_limiter import CLIRateLimiter, RateLimitExceeded


# ---------------------------------------------------------------------------
# build_parser
# ---------------------------------------------------------------------------


def test_build_parser_returns_argument_parser() -> None:
    """``build_parser`` returns an ``ArgumentParser`` instance."""
    parser = build_parser()
    assert isinstance(parser, argparse.ArgumentParser)


def test_build_parser_program_is_raghub() -> None:
    """The parser's program name is ``raghub``."""
    parser = build_parser()
    assert parser.prog == "raghub"


def test_build_parser_help_lists_all_subcommands() -> None:
    """The help text mentions every registered subcommand."""
    parser = build_parser()
    help_text = parser.format_help()
    for cmd in ("init", "ingest", "query", "eval", "config"):
        assert cmd in help_text


def test_build_parser_requires_subcommand() -> None:
    """No subcommand is reported as an error."""
    parser = build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args([])


def test_build_parser_parses_init() -> None:
    """``init`` subcommand parses successfully."""
    parser = build_parser()
    ns = parser.parse_args(["init"])
    assert ns.command == "init"
    assert callable(getattr(ns, "handler", None))


def test_build_parser_parses_health() -> None:
    """``health`` subcommand parses successfully."""
    parser = build_parser()
    ns = parser.parse_args(["health"])
    assert ns.command == "health"


def test_build_parser_parses_version() -> None:
    """``version`` subcommand parses successfully."""
    parser = build_parser()
    ns = parser.parse_args(["version"])
    assert ns.command == "version"


def test_build_parser_parses_run() -> None:
    """``run`` subcommand parses successfully."""
    parser = build_parser()
    ns = parser.parse_args(["run"])
    assert ns.command == "run"


def test_build_parser_parses_query_with_question() -> None:
    """``query`` subcommand accepts a question positional argument."""
    parser = build_parser()
    ns = parser.parse_args(["query", "What is revenue?"])
    assert ns.command == "query"
    assert ns.question == "What is revenue?"


def test_build_parser_parses_ingest_with_path() -> None:
    """``ingest`` subcommand accepts a path positional argument."""
    parser = build_parser()
    ns = parser.parse_args(["ingest", "report.pdf"])
    assert ns.command == "ingest"
    assert ns.path == "report.pdf"


def test_build_parser_parses_eval_with_benchmark() -> None:
    """``eval`` subcommand accepts a benchmark positional argument."""
    parser = build_parser()
    ns = parser.parse_args(["eval", "financebench"])
    assert ns.command == "eval"
    assert ns.benchmark == "financebench"


def test_build_parser_parses_config_subcommand() -> None:
    """The ``config`` group subcommand parses successfully."""
    parser = build_parser()
    ns = parser.parse_args(["config", "tools", "list", "--email", "alice@acme.com"])
    assert ns.command == "config"


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------


def test_cli_limiter_is_a_rate_limiter_instance() -> None:
    """The module-level ``CLI_LIMITER`` is a :class:`CLIRateLimiter`."""
    assert isinstance(CLI_LIMITER, CLIRateLimiter)


def test_rate_limit_exempt_commands_contains_expected_names() -> None:
    """The exempt set contains the documented always-on commands."""
    assert "health" in RATE_LIMIT_EXEMPT_COMMANDS
    assert "version" in RATE_LIMIT_EXEMPT_COMMANDS
    assert "run" in RATE_LIMIT_EXEMPT_COMMANDS


# ---------------------------------------------------------------------------
# main()
# ---------------------------------------------------------------------------


def test_main_help_returns_zero() -> None:
    """Calling ``main`` with ``--help`` exits with code ``0``."""
    test_argv = sys.argv
    sys.argv = ["raghub", "--help"]
    try:
        with pytest.raises(SystemExit) as exc_info:
            main()
    finally:
        sys.argv = test_argv
    assert exc_info.value.code == 0


def test_main_runs_health_handler(monkeypatch: pytest.MonkeyPatch) -> None:
    """``main`` runs the ``health`` handler (rate-limit exempt)."""
    monkeypatch.setattr(sys, "argv", ["raghub", "health"])
    buf = io.StringIO()
    with redirect_stdout(buf):
        rc = main()
    assert rc == 0
    output = buf.getvalue()
    # Should contain some JSON payload describing the health status.
    decoder = json.JSONDecoder()
    obj, _ = decoder.raw_decode(output[output.find("{"):])
    assert obj["status"] == "ok"


def test_main_runs_version_handler(monkeypatch: pytest.MonkeyPatch) -> None:
    """``main`` runs the ``version`` handler (rate-limit exempt)."""
    monkeypatch.setattr(sys, "argv", ["raghub", "version"])
    rc = main()
    assert rc == 0


def test_main_runs_run_handler(monkeypatch: pytest.MonkeyPatch) -> None:
    """``main`` dispatches the ``run`` command (which actually boots uvicorn).

    We don't need to actually boot uvicorn — we just verify that
    ``main`` reaches the run handler without rejecting it at the
    rate-limit stage. We achieve that by patching the run handler to
    return ``0`` immediately.
    """
    import raghub.cli.run_cmd as run_cmd

    monkeypatch.setattr(run_cmd, "run_subcommand", lambda ns: 0)
    monkeypatch.setattr(sys, "argv", ["raghub", "run"])
    rc = main()
    assert rc == 0


def test_main_returns_nonzero_on_rate_limit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When the rate limiter rejects the command, ``main`` returns ``1``."""

    def _raise(_: str) -> None:
        raise RateLimitExceeded("limited")

    monkeypatch.setattr(CLI_LIMITER, "check", _raise)
    monkeypatch.setattr(sys, "argv", ["raghub", "init"])
    rc = main()
    assert rc == 1


def test_main_runs_init_handler(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """``main`` dispatches ``init`` end-to-end and writes the sample config."""
    monkeypatch.setattr(sys, "argv", ["raghub", "init", "-o", str(tmp_path / "rag.yaml")])
    rc = main()
    assert rc == 0
    assert (tmp_path / "rag.yaml").exists()


# ---------------------------------------------------------------------------
# Re-exports
# ---------------------------------------------------------------------------


def test_rate_limit_exceeded_is_importable() -> None:
    """``RateLimitExceeded`` can be imported from the same path."""
    from raghub.cli.rate_limiter import RateLimitExceeded as RLE

    assert RLE is RateLimitExceeded
    assert issubclass(RLE, Exception)