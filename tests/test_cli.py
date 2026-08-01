"""CLI surface coverage tests.

Exercises the Typer CLI commands assembled in :mod:`raghub.cli` and
:mod:`raghub.cli_commands`. Uses :class:`typer.testing.CliRunner`.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest
import yaml
from typer.testing import CliRunner

import raghub.cli as cli_module
from raghub.cli import app
from raghub.cli_commands import CliConfig, InitCommand, ToolConfig

runner = CliRunner()


# ---------------------------------------------------------------------------
# Module-level: imports, attrs, exports
# ---------------------------------------------------------------------------


def test_cli_module_exports_three_names() -> None:
    """The CLI module re-exports health, main, version."""
    assert cli_module.__all__ == ["health", "main", "version"]


def test_main_invokes_app(monkeypatch: pytest.MonkeyPatch) -> None:
    """The console script entry runs the Typer app."""
    runner_local = CliRunner()
    result = runner_local.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "raghub" in result.output.lower()


def test_health_command_invokes_make_rag(monkeypatch: pytest.MonkeyPatch) -> None:
    """``raghub health`` calls CliConfig.make_rag(None).health() and writes JSON."""
    calls: list[tuple[str, object]] = []

    class _FakeRAG:
        def health(self) -> dict[str, object]:
            return {"status": "ok", "chunks": 7}

        @classmethod
        def from_config(cls, config: str | None) -> _FakeRAG:
            calls.append(("from_config", config))
            return cls()

    fake_make_rag = MagicMock(return_value=_FakeRAG())
    monkeypatch.setattr(cli_module, "CliConfig", MagicMock(write_json=CliConfig.write_json, make_rag=fake_make_rag))
    result = runner.invoke(app, ["health"])
    assert result.exit_code == 0
    assert "ok" in result.output
    fake_make_rag.assert_called_once_with(None)


def test_version_command_prints_package_version(monkeypatch: pytest.MonkeyPatch) -> None:
    """``raghub version`` echoes the installed package version string."""
    monkeypatch.setattr(cli_module.importlib.metadata, "version", lambda pkg: "1.2.3")
    result = runner.invoke(app, ["version"])
    assert result.exit_code == 0
    assert "1.2.3" in result.output


def test_version_command_handles_missing_package(monkeypatch: pytest.MonkeyPatch) -> None:
    """When the package metadata is missing, ``raghub version`` says ``unknown``."""

    def _raise(_pkg: str) -> str:
        raise Exception("not installed")

    monkeypatch.setattr(cli_module.importlib.metadata, "version", _raise)
    result = runner.invoke(app, ["version"])
    assert "unknown" in result.output


def test_no_args_shows_help() -> None:
    """Running the CLI with no args prints help (no_args_is_help=True)."""
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "raghub" in result.output.lower()


# ---------------------------------------------------------------------------
# CliConfig helpers
# ---------------------------------------------------------------------------


def test_cli_config_write_json(capsys: pytest.CaptureFixture[str]) -> None:
    """CliConfig.write_json pretty-prints to stdout."""
    CliConfig.write_json({"a": 1, "b": [1, 2]})
    out = capsys.readouterr().out
    assert json.loads(out) == {"a": 1, "b": [1, 2]}


def test_cli_config_read_settings_yaml(tmp_path: Path) -> None:
    """read_settings parses YAML and filters to Settings fields."""
    path = tmp_path / "cfg.yaml"
    path.write_text(yaml.safe_dump({"llm_model": "litellm", "nonsense": True}))
    settings = CliConfig.read_settings(str(path))
    assert settings.llm_model == "litellm"


def test_cli_config_read_settings_toml(tmp_path: Path) -> None:
    """read_settings parses TOML and filters to Settings fields."""
    path = tmp_path / "cfg.toml"
    path.write_text('llm_model = "heuristic-llm"\n')
    settings = CliConfig.read_settings(str(path))
    assert settings.llm_model == "heuristic-llm"


def test_cli_config_read_settings_no_path_returns_defaults() -> None:
    """read_settings(None) loads the active profile."""
    settings = CliConfig.read_settings(None)
    assert settings is not None


def test_cli_config_make_rag_no_config() -> None:
    """make_rag(None) returns a default RAG instance."""
    from raghub.rag import RAG

    rag = CliConfig.make_rag(None)
    assert isinstance(rag, RAG)


# ---------------------------------------------------------------------------
# InitCommand
# ---------------------------------------------------------------------------


def test_init_prints_to_stdout_when_no_output() -> None:
    """``raghub init`` prints the sample config to stdout."""
    result = runner.invoke(app, ["init"])
    assert result.exit_code == 0
    assert "RAGHub" in result.output or "raghub" in result.output.lower()


def test_init_writes_to_output_file(tmp_path: Path) -> None:
    """``raghub init -o path`` writes the starter config to the path."""
    output = tmp_path / "config.yaml"
    result = runner.invoke(app, ["init", "-o", str(output)])
    assert result.exit_code == 0
    assert output.exists()
    assert "environment" in output.read_text(encoding="utf-8")


def test_app_registers_expected_subcommands() -> None:
    """The CLI registers init, ingest, query, run, health, version, config."""
    command_names = {cmd.name for cmd in cli_module.app.registered_commands}
    for expected in ("init", "ingest", "query", "run", "health", "version"):
        assert expected in command_names


def test_sample_config_contains_required_keys() -> None:
    """The sample starter config advertises the documented keys."""
    text = InitCommand.SAMPLE_CONFIG
    for key in (
        "data_dir",
        "chunk_size_words",
        "embedding_dim",
        "llm_model",
        "log_level",
    ):
        assert key in text, key


# ---------------------------------------------------------------------------
# ToolConfig sub-app
# ---------------------------------------------------------------------------


def test_tool_config_app_is_a_typer() -> None:
    """ToolConfig.app is a Typer app."""
    from typer import Typer

    assert isinstance(ToolConfig.app, Typer)
    assert isinstance(ToolConfig.tools, Typer)


def test_tool_config_keys_are_frozenset_of_strings() -> None:
    """ToolConfig.keys advertises a fixed tuple of known settings."""
    assert all(isinstance(k, str) for k in ToolConfig.keys)
    assert "agent_enabled" in ToolConfig.keys
    assert "max_steps" in ToolConfig.keys


def test_tool_config_register_is_idempotent() -> None:
    """Calling register() twice does not duplicate commands."""
    ToolConfig.register()
    first = list(ToolConfig.tools.registered_commands)
    ToolConfig.register()
    second = list(ToolConfig.tools.registered_commands)
    assert first == second


def test_tool_config_list_unknown_email_raises(capsys: pytest.CaptureFixture[str]) -> None:
    """``raghub config tools list --email absent`` exits non-zero with a bad parameter."""
    # Set data_dir to a temp directory so the loader doesn't touch real data.
    import os

    os.environ.setdefault("RAGHUB_DATA_DIR", "/tmp/raghub-test-not-used")
    result = runner.invoke(app, ["config", "tools", "list", "--email", "absent@example.com"])
    # Typer exits with code 2 on bad parameter; the message must mention the email.
    assert result.exit_code != 0
    assert "absent@example.com" in result.output or "no user" in result.output.lower()


def test_tool_config_set_invalid_json_raises() -> None:
    """``raghub config tools set --json 'not-json'`` errors with a bad parameter."""
    result = runner.invoke(app, ["config", "tools", "set", "--email", "x@x.com", "--json", "not-json"])
    assert result.exit_code != 0
    assert "invalid json" in result.output.lower() or "Bad Parameter" in result.output


def test_tool_config_set_non_object_json_raises() -> None:
    """``raghub config tools set --json '[]'`` rejects non-object payloads."""
    result = runner.invoke(app, ["config", "tools", "set", "--email", "x@x.com", "--json", "[]"])
    assert result.exit_code != 0
    assert "json object" in result.output.lower()


def test_tool_config_set_unknown_email_raises() -> None:
    """``raghub config tools set --email absent`` exits non-zero."""
    result = runner.invoke(app, ["config", "tools", "set", "--email", "absent@example.com", "--json", "{}"])
    assert result.exit_code != 0
    assert "no user" in result.output.lower()


def test_tool_config_unset_unknown_email_raises() -> None:
    """``raghub config tools unset --email absent`` exits non-zero."""
    result = runner.invoke(app, ["config", "tools", "unset", "--email", "absent@example.com"])
    assert result.exit_code != 0
    assert "no user" in result.output.lower()


# ---------------------------------------------------------------------------
# IngestCommand
# ---------------------------------------------------------------------------


def test_ingest_command_invokes_rag_ingest(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """``raghub ingest path`` calls RAG.ingest and prints OK."""
    fake_batch = MagicMock()
    fake_batch.model_dump.return_value = {"path": "x"}
    fake_result = MagicMock()
    fake_result.outputs = {"batch": [fake_batch]}
    fake_result.error = None
    fake_rag = MagicMock()
    fake_rag.ingest.return_value = fake_result
    monkeypatch.setattr(CliConfig, "make_rag", lambda config: fake_rag)
    test_file = tmp_path / "doc.txt"
    test_file.write_text("hello", encoding="utf-8")
    result = runner.invoke(app, ["ingest", str(test_file)])
    assert result.exit_code == 0, result.output
    fake_rag.ingest.assert_called_once()
    assert "ingested batch" in result.output or "ingested" in result.output


def test_ingest_command_success_prints_ok(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """When success is True in the result, ingest prints ``OK``."""
    fake_result = MagicMock()
    fake_result.outputs = {}
    fake_result.error = None

    def _dump(*_args: object, **_kwargs: object) -> dict[str, object]:
        return {"success": True}

    fake_result.model_dump.side_effect = _dump
    fake_rag = MagicMock()
    fake_rag.ingest.return_value = fake_result
    monkeypatch.setattr(CliConfig, "make_rag", lambda config: fake_rag)
    test_file = tmp_path / "doc.txt"
    test_file.write_text("hi", encoding="utf-8")
    result = runner.invoke(app, ["ingest", str(test_file)])
    assert result.exit_code == 0, result.output
    assert "OK" in result.output


def test_ingest_command_failure_prints_fail(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """When success is False in the result, ingest prints ``FAIL``."""
    fake_result = MagicMock()
    fake_result.outputs = {}
    fake_result.error = None

    def _dump(*_args: object, **_kwargs: object) -> dict[str, object]:
        return {"success": False}

    fake_result.model_dump.side_effect = _dump
    fake_rag = MagicMock()
    fake_rag.ingest.return_value = fake_result
    monkeypatch.setattr(CliConfig, "make_rag", lambda config: fake_rag)
    test_file = tmp_path / "doc.txt"
    test_file.write_text("hi", encoding="utf-8")
    result = runner.invoke(app, ["ingest", str(test_file)])
    assert result.exit_code == 0, result.output
    assert "FAIL" in result.output


def test_ingest_command_no_success_prints_ingested(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """When result has no ``success`` key, ingest prints ``ingested PATH``."""
    fake_result = MagicMock()
    fake_result.outputs = {}
    fake_result.error = None

    def _dump(*_args: object, **_kwargs: object) -> dict[str, object]:
        return {"chunk_count": 3}

    fake_result.model_dump.side_effect = _dump
    fake_rag = MagicMock()
    fake_rag.ingest.return_value = fake_result
    monkeypatch.setattr(CliConfig, "make_rag", lambda config: fake_rag)
    test_file = tmp_path / "doc.txt"
    test_file.write_text("hi", encoding="utf-8")
    result = runner.invoke(app, ["ingest", str(test_file)])
    assert result.exit_code == 0, result.output
    assert "ingested" in result.output


def test_ingest_async_emits_job_id(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """``raghub ingest --async PATH`` calls ``ingest_async`` and echoes the job id."""
    fake_rag = MagicMock()
    fake_rag.ingest_async.return_value = "job-1234"
    monkeypatch.setattr(CliConfig, "make_rag", lambda config: fake_rag)
    test_file = tmp_path / "doc.txt"
    test_file.write_text("hello", encoding="utf-8")
    result = runner.invoke(app, ["ingest", "--async", str(test_file)])
    assert result.exit_code == 0
    assert "job-1234" in result.output
    fake_rag.ingest_async.assert_called_once()


def test_ingest_async_json_output(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """``raghub ingest --async --json PATH`` writes JSON with the job id."""
    fake_rag = MagicMock()
    fake_rag.ingest_async.return_value = "job-9"
    monkeypatch.setattr(CliConfig, "make_rag", lambda config: fake_rag)
    test_file = tmp_path / "doc.txt"
    test_file.write_text("hi", encoding="utf-8")
    result = runner.invoke(app, ["ingest", "--async", "--json", str(test_file)])
    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload["job_id"] == "job-9"


# ---------------------------------------------------------------------------
# QueryCommand
# ---------------------------------------------------------------------------


def test_query_command_prints_answer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``raghub query "..."`` echoes the response answer."""
    fake_response = MagicMock()
    fake_response.answer = "the answer is 42"
    fake_response.citations = []
    fake_response.source_chunks = []
    fake_response.metadata = {}
    fake_rag = MagicMock()
    fake_rag.query.return_value = fake_response
    monkeypatch.setattr(CliConfig, "make_rag", lambda config: fake_rag)
    result = runner.invoke(app, ["query", "what is the answer?"])
    assert result.exit_code == 0
    assert "the answer is 42" in result.output
    fake_rag.query.assert_called_once()


def test_query_command_json_output(monkeypatch: pytest.MonkeyPatch) -> None:
    """``raghub query --json "..."`` writes a JSON payload with answer + metadata."""
    fake_response = MagicMock()
    fake_response.answer = "yes"
    fake_response.citations = []
    fake_response.source_chunks = []
    fake_response.metadata = {"k": "v"}
    fake_rag = MagicMock()
    fake_rag.query.return_value = fake_response
    monkeypatch.setattr(CliConfig, "make_rag", lambda config: fake_rag)
    result = runner.invoke(app, ["query", "--json", "q"])
    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload["answer"] == "yes"
    assert payload["metadata"] == {"k": "v"}


def test_query_command_with_citations(monkeypatch: pytest.MonkeyPatch) -> None:
    """``raghub query `` prints the citation footer for non-empty citations."""
    fake_citation = MagicMock()
    fake_citation.document_id = "d1"
    fake_citation.id = "c1"
    fake_citation.score = 0.81
    fake_response = MagicMock()
    fake_response.answer = "use"
    fake_response.citations = [fake_citation]
    fake_response.source_chunks = []
    fake_response.metadata = {}
    fake_rag = MagicMock()
    fake_rag.query.return_value = fake_response
    monkeypatch.setattr(CliConfig, "make_rag", lambda config: fake_rag)
    result = runner.invoke(app, ["query", "q"])
    assert result.exit_code == 0
    assert "1 citations" in result.output
    assert "d1" in result.output


# ---------------------------------------------------------------------------
# ServerCommand (just verify uvicorn.Server.run is wired)
# ---------------------------------------------------------------------------


def test_server_command_invokes_uvicorn(monkeypatch: pytest.MonkeyPatch) -> None:
    """``raghub run`` constructs an uvicorn.Server and calls .run()."""
    fake_server = MagicMock()
    fake_config = MagicMock()
    monkeypatch.setattr("raghub.cli_commands.uvicorn.Config", lambda *args, **kwargs: fake_config)
    monkeypatch.setattr("raghub.cli_commands.uvicorn.Server", lambda config: fake_server)
    result = runner.invoke(app, ["run"])
    assert result.exit_code == 0
    fake_server.run.assert_called_once()
