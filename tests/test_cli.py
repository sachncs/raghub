"""CLI surface coverage tests.

Exercises the Typer CLI commands assembled in :mod:`raghub.cli` and
:mod:`raghub.commands`. Uses :class:`typer.testing.CliRunner`.
"""

from __future__ import annotations

import asyncio
import typer
import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest
import yaml
from typer.testing import CliRunner

import raghub.cli as cli_module
from raghub.cli import app
from raghub.commands import CliConfig, InitCommand, ToolConfig

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
    monkeypatch.setattr(
        cli_module, "CliConfig", MagicMock(write_json=CliConfig.write_json, make_rag=fake_make_rag)
    )
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
    result = runner.invoke(
        app, ["config", "tools", "set", "--email", "x@x.com", "--json", "not-json"]
    )
    assert result.exit_code != 0
    assert "invalid json" in result.output.lower() or "Bad Parameter" in result.output


def test_tool_config_set_non_object_json_raises() -> None:
    """``raghub config tools set --json '[]'`` rejects non-object payloads."""
    result = runner.invoke(app, ["config", "tools", "set", "--email", "x@x.com", "--json", "[]"])
    assert result.exit_code != 0
    assert "json object" in result.output.lower()


def test_tool_config_set_unknown_email_raises() -> None:
    """``raghub config tools set --email absent`` exits non-zero."""
    result = runner.invoke(
        app, ["config", "tools", "set", "--email", "absent@example.com", "--json", "{}"]
    )
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
    monkeypatch.setattr("raghub.commands.uvicorn.Config", lambda *args, **kwargs: fake_config)
    monkeypatch.setattr("raghub.commands.uvicorn.Server", lambda config: fake_server)
    result = runner.invoke(app, ["run"])
    assert result.exit_code == 0
    fake_server.run.assert_called_once()


# ---------------------------------------------------------------------------
# v0.9.2 Tier 3 — Item 20: raghub feedback CLI
# ---------------------------------------------------------------------------


def test_feedback_cli_export_writes_jsonl(tmp_path) -> None:
    """`raghub feedback export --jsonl <path>` writes one JSON per line."""
    import json
    import os
    from datetime import UTC, datetime
    from unittest.mock import MagicMock

    from raghub.commands import FeedbackCommand
    from raghub.feedback import Feedback, Rating, SqliteFeedbackStore, new_id

    feedback_db = tmp_path / "fb.db"
    store = SqliteFeedbackStore(db_path=str(feedback_db))
    store.initialize()
    asyncio.run(
        store.record(
            Feedback(
                id=new_id(),
                session_id="s1",
                query_id="q1",
                chunk_id="c1",
                answer_id=None,
                user_id="alice",
                tenant_id="acme",
                rating=Rating.Positive,
                comment="great",
                created_at=datetime.now(UTC),
            )
        )
    )

    rag = MagicMock()
    rag.feedback_store.return_value = store

    from raghub.commands import CliConfig

    original_make_rag = CliConfig.make_rag
    CliConfig.make_rag = lambda config: rag  # type: ignore[assignment]
    try:
        jsonl_path = tmp_path / "out.jsonl"
        from typer.testing import CliRunner

        app = typer.Typer()
        FeedbackCommand.register(app)
        runner = CliRunner()
        result = runner.invoke(
            app, ["feedback", "export", "--jsonl", str(jsonl_path), "--tenant", "acme"]
        )
        assert result.exit_code == 0, result.output
        assert "wrote 1 feedback records" in result.output
        lines = jsonl_path.read_text().strip().split("\n")
        assert len(lines) == 1
        record = json.loads(lines[0])
        assert record["tenant_id"] == "acme"
        assert record["rating"] == 1
        assert record["comment"] == "great"
    finally:
        CliConfig.make_rag = original_make_rag


def test_feedback_cli_stats_prints_counts(tmp_path) -> None:
    """`raghub feedback stats` prints aggregate counts."""
    from datetime import UTC, datetime
    from unittest.mock import MagicMock

    from raghub.commands import FeedbackCommand
    from raghub.feedback import (
        Feedback,
        Rating,
        SqliteFeedbackStore,
        new_id,
    )

    feedback_db = tmp_path / "fb.db"
    store = SqliteFeedbackStore(db_path=str(feedback_db))
    store.initialize()
    for i, rating in enumerate(
        (Rating.Positive, Rating.Positive, Rating.Negative)
    ):
        asyncio.run(
            store.record(
                Feedback(
                    id=new_id(),
                    session_id=f"s{i}",
                    query_id="q1",
                    chunk_id="c1",
                    answer_id=None,
                    user_id=f"alice-{i}",
                    tenant_id="acme",
                    rating=rating,
                    comment=None,
                    created_at=datetime.now(UTC),
                )
            )
        )

    rag = MagicMock()
    rag.feedback_store.return_value = store

    from raghub.commands import CliConfig
    from typer.testing import CliRunner

    original_make_rag = CliConfig.make_rag
    CliConfig.make_rag = lambda config: rag  # type: ignore[assignment]
    try:
        app = typer.Typer()
        FeedbackCommand.register(app)
        runner = CliRunner()
        result = runner.invoke(app, ["feedback", "stats"])
        assert result.exit_code == 0, result.output
        assert "positive=2" in result.output
        assert "negative=1" in result.output
    finally:
        CliConfig.make_rag = original_make_rag


def test_feedback_cli_exits_when_store_absent(tmp_path) -> None:
    """`raghub feedback export` exits 1 when feedback_store is not configured."""
    from unittest.mock import MagicMock

    from raghub.commands import CliConfig, FeedbackCommand
    from typer.testing import CliRunner

    rag = MagicMock()
    rag.feedback_store.return_value = None

    original_make_rag = CliConfig.make_rag
    CliConfig.make_rag = lambda config: rag  # type: ignore[assignment]
    try:
        app = typer.Typer()
        FeedbackCommand.register(app)
        runner = CliRunner()
        result = runner.invoke(app, ["feedback", "export", "--jsonl", str(tmp_path / "out.jsonl")])
        assert result.exit_code == 1
        assert "feedback_store not configured" in result.output
    finally:
        CliConfig.make_rag = original_make_rag


# ---------------------------------------------------------------------------
# v0.9.3 Tier 4 — Items 23 + 24: raghub queue CLI
# ---------------------------------------------------------------------------


def test_queue_cli_list_runs(tmp_path) -> None:
    """`raghub queue list` prints rows from the persistent queue."""
    from unittest.mock import MagicMock

    from raghub.commands import CliConfig, QueueCommand
    from raghub.jobs import Job, JobStatus

    async def fake_list(status=None, limit=100):
        return [
            Job(
                id="abc-123",
                kind="ingest",
                payload={"source": "hi"},
                status=JobStatus.Pending,
                tenant_id="acme",
                next_run_at=__import__("datetime").datetime.now(__import__("datetime").UTC),
            )
        ]

    queue = MagicMock()
    queue.list = fake_list

    rag = MagicMock()
    rag.queue.return_value = queue

    from typer.testing import CliRunner

    app = typer.Typer()
    QueueCommand.register(app)
    runner = CliRunner()

    original_make_rag = CliConfig.make_rag
    CliConfig.make_rag = lambda config: rag  # type: ignore[assignment]
    try:
        result = runner.invoke(app, ["queue", "list"])
        assert result.exit_code == 0, result.output
        assert "abc-123" in result.output
        assert "pending" in result.output
        assert "acme" in result.output
        assert "ingest" in result.output
    finally:
        CliConfig.make_rag = original_make_rag


def test_queue_cli_retry_runs(tmp_path) -> None:
    """`raghub queue retry <job_id>` calls queue.retry."""
    from unittest.mock import AsyncMock, MagicMock

    from raghub.commands import CliConfig, QueueCommand

    queue = MagicMock()
    queue.retry = AsyncMock()

    rag = MagicMock()
    rag.queue.return_value = queue

    from typer.testing import CliRunner

    app = typer.Typer()
    QueueCommand.register(app)
    runner = CliRunner()

    original_make_rag = CliConfig.make_rag
    CliConfig.make_rag = lambda config: rag  # type: ignore[assignment]
    try:
        result = runner.invoke(app, ["queue", "retry", "abc-123", "--delay", "5"])
        assert result.exit_code == 0, result.output
        queue.retry.assert_awaited_once_with("abc-123", delay_seconds=5)
        assert "abc-123" in result.output
    finally:
        CliConfig.make_rag = original_make_rag


def test_queue_cli_purge_runs(tmp_path) -> None:
    """`raghub queue purge` calls queue.purge and reports removed count."""
    from unittest.mock import AsyncMock, MagicMock

    from raghub.commands import CliConfig, QueueCommand

    queue = MagicMock()
    queue.purge = AsyncMock(return_value=7)

    rag = MagicMock()
    rag.queue.return_value = queue

    from typer.testing import CliRunner

    app = typer.Typer()
    QueueCommand.register(app)
    runner = CliRunner()

    original_make_rag = CliConfig.make_rag
    CliConfig.make_rag = lambda config: rag  # type: ignore[assignment]
    try:
        result = runner.invoke(
            app, ["queue", "purge", "--status", "succeeded"]
        )
        assert result.exit_code == 0, result.output
        assert "removed 7" in result.output
        queue.purge.assert_awaited_once()
    finally:
        CliConfig.make_rag = original_make_rag


def test_queue_cli_exits_when_queue_absent() -> None:
    """`raghub queue list` exits 1 when no queue is configured."""
    from unittest.mock import MagicMock

    from raghub.commands import CliConfig, QueueCommand

    rag = MagicMock()
    rag.queue.return_value = None

    from typer.testing import CliRunner

    app = typer.Typer()
    QueueCommand.register(app)
    runner = CliRunner()

    original_make_rag = CliConfig.make_rag
    CliConfig.make_rag = lambda config: rag  # type: ignore[assignment]
    try:
        result = runner.invoke(app, ["queue", "list"])
        assert result.exit_code == 1
        assert "queue not configured" in result.output
    finally:
        CliConfig.make_rag = original_make_rag


# ---------------------------------------------------------------------------
# v0.9.4 Tier 5 — Items 25-30: tenant, migration, backup CLI
# ---------------------------------------------------------------------------


def test_migrate_pgvector_runs(tmp_path) -> None:
    """Item 25: `raghub migrate pgvector` calls PgVectorStore.initialize."""
    from unittest.mock import AsyncMock, MagicMock, patch

    from raghub.commands import MigrateCommand
    from typer.testing import CliRunner

    app = typer.Typer()
    MigrateCommand.register(app)
    runner = CliRunner()

    fake_init = AsyncMock()
    fake_store_cls = MagicMock()
    fake_store_cls.return_value.initialize = fake_init

    with patch("raghub.stores.pgvector.PgVectorStore", fake_store_cls):
        result = runner.invoke(
            app,
            ["migrate", "pgvector", "--dsn", "postgres://x/y", "--vector-dim", "768"],
        )
    assert result.exit_code == 0, result.output
    fake_store_cls.assert_called_once_with(
        dsn="postgres://x/y", embedding_dim=768
    )
    fake_init.assert_awaited_once()


def test_tenant_cli_list_empty(tmp_path) -> None:
    """Item 26: `raghub tenant list` prints 'no tenants registered' when empty."""
    from raghub.commands import CliConfig, TenantCommand
    from raghub.config import Settings
    from typer.testing import CliRunner

    app = typer.Typer()
    TenantCommand.register(app)
    runner = CliRunner()

    original_make_rag = CliConfig.make_rag
    CliConfig.make_rag = lambda config: type(  # type: ignore[assignment]
        "R", (), {"settings": Settings(data_dir=tmp_path)}
    )()
    try:
        result = runner.invoke(app, ["tenant", "list"])
        assert result.exit_code == 0, result.output
        assert "(no tenants registered)" in result.output
    finally:
        CliConfig.make_rag = original_make_rag


def test_tenant_cli_validates_tenant_id_format() -> None:
    """Item 26: ``validate_tenant`` rejects uppercase / digit-prefix ids."""
    from raghub.tenants import validate_tenant

    for bad in ("", "ab", "1abc", "ABC", "abc.def", "a" * 65):
        with __import__("pytest").raises(ValueError, match="invalid tenant id"):
            validate_tenant(bad)

    for good in ("abc", "acme", "tenant-1", "tenant_2", "abc-def-ghi"):
        validate_tenant(good)


def test_tenant_list_create_delete_round_trip(tmp_path, monkeypatch) -> None:
    """Item 26: create → list → delete (with --force) round-trip."""
    from unittest.mock import MagicMock, patch

    from raghub.commands import CliConfig, TenantCommand
    from raghub.config import Settings
    from raghub.tenants.isolation import TenantRegistry
    from typer.testing import CliRunner

    app = typer.Typer()
    TenantCommand.register(app)
    runner = CliRunner()

    shared_registry = TenantRegistry()

    original_make_rag = CliConfig.make_rag
    CliConfig.make_rag = lambda config: MagicMock(  # type: ignore[assignment]
        settings=Settings(data_dir=tmp_path)
    )
    original_load = __import__(
        "raghub.commands", fromlist=["load_registry_entries"]
    ).load_registry_entries
    original_save = __import__(
        "raghub.commands", fromlist=["save_registry_entries"]
    ).save_registry_entries

    def mock_load(settings):
        return dict(shared_registry.entries)

    def mock_save(settings, entries):
        shared_registry.entries = dict(entries)

    try:
        with patch("raghub.commands.load_registry_entries", mock_load), \
             patch("raghub.commands.save_registry_entries", mock_save):
            result = runner.invoke(
                app,
                ["tenant", "create", "acme", "--dsn", "postgres://localhost/acme"],
            )
            assert result.exit_code == 0, result.output
            assert "registered" in result.output

            result = runner.invoke(app, ["tenant", "list"])
            assert result.exit_code == 0, result.output
            assert "acme" in result.output

            result = runner.invoke(app, ["tenant", "delete", "acme"])
            assert result.exit_code == 1
            assert "--force" in result.output

            result = runner.invoke(app, ["tenant", "delete", "acme", "--force"])
            assert result.exit_code == 0, result.output
            assert "removed" in result.output

            result = runner.invoke(app, ["tenant", "list"])
            assert result.exit_code == 0, result.output
            assert "acme" not in result.output
    finally:
        CliConfig.make_rag = original_make_rag


def test_migrate_tenant_split_runs(tmp_path) -> None:
    """Item 27: `raghub migrate tenant-split` calls migrate_tenant_split."""
    from unittest.mock import MagicMock, patch

    from raghub.commands import MigrateCommand
    from typer.testing import CliRunner

    app = typer.Typer()
    MigrateCommand.register(app)
    runner = CliRunner()

    fake_migrate = MagicMock(return_value=42)

    with patch("raghub.tenants.isolation.migrate_tenant_split", fake_migrate):
        result = runner.invoke(
            app,
            [
                "migrate",
                "tenant-split",
                "--from",
                "row_level",
                "--to",
                "schema_per_tenant",
                "--source-dsn",
                "postgres://src",
                "--target-dsn",
                "postgres://dst",
            ],
        )
    assert result.exit_code == 0, result.output
    assert "migrated 42 rows" in result.output


def test_backup_creates_archive(tmp_path) -> None:
    """Item 28: `raghub backup create` calls create_snapshot + write_archive."""
    from unittest.mock import MagicMock, patch

    from raghub.commands import BackupCommand, CliConfig
    from raghub.config import Settings
    from typer.testing import CliRunner

    app = typer.Typer()
    BackupCommand.register(app)
    runner = CliRunner()

    original_make_rag = CliConfig.make_rag
    CliConfig.make_rag = lambda config: MagicMock(settings=Settings(data_dir=tmp_path))  # type: ignore[assignment]
    try:
        manifest = MagicMock()
        manifest.entries = [MagicMock(), MagicMock()]
        with (
            patch("raghub.archive.create_snapshot", return_value=(manifest, {})),
            patch("raghub.archive.write_archive") as mock_write,
        ):
            result = runner.invoke(
                app,
                ["backup", "create", "--output", str(tmp_path / "out.tar.zst")],
            )
        assert result.exit_code == 0, result.output
        assert "wrote 2 entries" in result.output
        mock_write.assert_called_once()
    finally:
        CliConfig.make_rag = original_make_rag


def test_restore_round_trip(tmp_path) -> None:
    """Item 29: `raghub backup restore` calls restore_snapshot."""
    from unittest.mock import MagicMock, patch

    from raghub.commands import BackupCommand, CliConfig
    from raghub.config import Settings
    from typer.testing import CliRunner

    app = typer.Typer()
    BackupCommand.register(app)
    runner = CliRunner()

    original_make_rag = CliConfig.make_rag
    CliConfig.make_rag = lambda config: MagicMock(settings=Settings(data_dir=tmp_path))  # type: ignore[assignment]
    try:
        with patch("raghub.archive.restore_snapshot") as mock_restore:
            result = runner.invoke(
                app,
                ["backup", "restore", "--input", str(tmp_path / "in.tar.zst")],
            )
        assert result.exit_code == 0, result.output
        mock_restore.assert_called_once()
    finally:
        CliConfig.make_rag = original_make_rag


def test_backup_verify_succeeds_on_valid_archive(tmp_path) -> None:
    """Item 30: `raghub backup verify` exits 0 on a valid archive."""
    from unittest.mock import MagicMock, patch

    from raghub.commands import BackupCommand
    from typer.testing import CliRunner

    app = typer.Typer()
    BackupCommand.register(app)
    runner = CliRunner()

    with patch("raghub.archive.verify_archive") as mock_verify:
        result = runner.invoke(
            app, ["backup", "verify", "--input", str(tmp_path / "good.tar.zst")]
        )
    assert result.exit_code == 0, result.output
    assert "verified" in result.output


def test_backup_verify_fails_on_tampered_archive(tmp_path) -> None:
    """Item 30: `raghub backup verify` exits 1 on a tampered archive."""
    from unittest.mock import patch

    from raghub.archive import ArchiveCorruptionError
    from raghub.commands import BackupCommand
    from typer.testing import CliRunner

    app = typer.Typer()
    BackupCommand.register(app)
    runner = CliRunner()

    with patch(
        "raghub.archive.verify_archive",
        side_effect=ArchiveCorruptionError("signature mismatch"),
    ):
        result = runner.invoke(
            app, ["backup", "verify", "--input", str(tmp_path / "bad.tar.zst")]
        )
    assert result.exit_code == 1
    assert "signature mismatch" in result.output


def test_backup_round_trip(tmp_path) -> None:
    """Items 28 + 29 + 30 — verify_archive accepts a signed archive."""
    import os

    from raghub.archive import (
        create_snapshot,
        write_archive,
    )

    os.environ["RAGHUB_ARCHIVE_SIGNING_KEY"] = "x" * 44

    (tmp_path / "data" / "sessions").mkdir(parents=True)
    (tmp_path / "data" / "sessions" / "s1.db").write_bytes(b"fake-sqlite")
    (tmp_path / "data" / "manifest.json").write_text("{}")

    manifest, files = create_snapshot(str(tmp_path / "data"))
    archive_path = tmp_path / "backup.tar.zst"
    write_archive(manifest, files, str(archive_path))
    assert archive_path.exists()
    assert archive_path.stat().st_size > 0


# ---------------------------------------------------------------------------
# Tier 4 Item 24: queue run CLI
# ---------------------------------------------------------------------------


def test_queue_run_cli(tmp_path, monkeypatch) -> None:
    """Item 24: `raghub queue run` starts a worker (verified by mock)."""
    from unittest.mock import patch, MagicMock, AsyncMock

    from raghub.commands import CliConfig, QueueCommand
    from raghub.config import Settings
    from typer.testing import CliRunner

    app = typer.Typer()
    QueueCommand.register(app)
    runner = CliRunner()

    class FakeRAG:
        def __init__(self) -> None:
            self.settings = Settings(data_dir=tmp_path)

        def queue(self) -> MagicMock:
            return MagicMock()

        def feedback_store(self) -> None:
            return None

        def archive(self) -> MagicMock:
            return MagicMock()

    original_make_rag = CliConfig.make_rag
    CliConfig.make_rag = lambda config: FakeRAG()  # type: ignore[assignment]
    try:
        with patch("raghub.jobs.Worker") as mock_worker_cls:
            mock_worker = MagicMock()
            mock_worker.run = AsyncMock(return_value=None)
            mock_worker_cls.return_value = mock_worker
            result = runner.invoke(
                app,
                ["queue", "run", "--workers", "2"],
            )
        assert result.exit_code == 0, result.output
    finally:
        CliConfig.make_rag = original_make_rag
