"""Targeted tests for uncovered CLI lines."""

from __future__ import annotations

import argparse
import io
import json
import sys
from contextlib import redirect_stdout
from importlib.metadata import PackageNotFoundError
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch


from raghub.cli import eval_cmd, ingest_cmd, main, system
from raghub.cli.common import load_settings_or_path
from raghub.models import PipelineResult


# =======================================================================
# main.py  —  lines 28–34, 38
# =======================================================================


def test_main_with_handler_returns_int():
    """main() returns int(handler(args)) when a handler is set."""
    with patch("argparse.ArgumentParser.parse_args") as mock_parse:
        mock_parse.return_value = argparse.Namespace(handler=lambda ns: 42)
        rc = main.main()
    assert rc == 42


def test_main_without_handler_prints_help():
    """main() prints usage and returns 0 when no handler is set."""
    with patch("argparse.ArgumentParser.parse_args") as mock_parse:
        mock_parse.return_value = argparse.Namespace(handler=None)
        buf = io.StringIO()
        with redirect_stdout(buf):
            rc = main.main()
    assert rc == 0
    assert "usage:" in buf.getvalue().lower()


def test_main_module_entry_via_subprocess():
    """The ``if __name__ == '__main__'`` block calls main()."""
    import subprocess

    result = subprocess.run(
        [sys.executable, "-m", "raghub.cli", "version"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0


# =======================================================================
# ingest_cmd.py  —  lines 41, 44–46, 51–52
# =======================================================================


def test_ingest_no_config_uses_default_rag():
    """run_subcommand creates RAG() when no --config is given."""
    args = argparse.Namespace(path="doc.pdf", config=None, async_job=False)
    with patch("raghub.cli.ingest_cmd.RAG") as mock_rag_cls:
        mock_rag = MagicMock()
        mock_rag_cls.return_value = mock_rag
        mock_rag.ingest.return_value = MagicMock(outputs={})
        rc = ingest_cmd.run_subcommand(args)
    assert rc == 0
    mock_rag_cls.assert_called_once_with()


def test_ingest_async_prints_job_id():
    """run_subcommand with --async prints job_id JSON."""
    args = argparse.Namespace(path="doc.pdf", config=None, async_job=True)
    with patch("raghub.cli.ingest_cmd.RAG") as mock_rag_cls:
        mock_rag = MagicMock()
        mock_rag_cls.return_value = mock_rag
        mock_rag.ingest_async.return_value = "job-42"
        buf = io.StringIO()
        with redirect_stdout(buf):
            rc = ingest_cmd.run_subcommand(args)
    assert rc == 0
    assert json.loads(buf.getvalue()) == {"job_id": "job-42"}


def test_ingest_batch_output_prints_list():
    """run_subcommand prints a JSON list for directory (batch) ingest."""
    inner = PipelineResult(pipeline_id="p1", pipeline_name="ingest", outputs={})
    batch_result = PipelineResult(
        pipeline_id="batch",
        pipeline_name="ingest",
        outputs={"batch": [inner]},
    )
    args = argparse.Namespace(path="mydir", config=None, async_job=False)
    with patch("raghub.cli.ingest_cmd.RAG") as mock_rag_cls:
        mock_rag = MagicMock()
        mock_rag_cls.return_value = mock_rag
        mock_rag.ingest.return_value = batch_result
        buf = io.StringIO()
        with redirect_stdout(buf):
            rc = ingest_cmd.run_subcommand(args)
    assert rc == 0
    payload = json.loads(buf.getvalue())
    assert isinstance(payload, list)
    assert len(payload) == 1


# =======================================================================
# eval_cmd.py  —  lines 50–80, 93, 102–106
# =======================================================================


def test_eval_build_console_namespace():
    """build_console_namespace populates the benchmark field."""
    ns = argparse.Namespace(examples=3)
    result = eval_cmd.build_console_namespace(ns)
    assert result.benchmark == "financebench"
    assert result.examples == 3


def test_eval_main_parses_args_and_runs():
    """eval_cmd.main() parses CLI args and runs the subcommand."""
    with patch("argparse.ArgumentParser.parse_args") as mock_parse:
        mock_parse.return_value = argparse.Namespace(examples=5)
        with patch.object(eval_cmd, "run_subcommand") as mock_run:
            mock_run.return_value = 0
            rc = eval_cmd.main()
    assert rc == 0
    mock_run.assert_called_once()


def test_eval_run_subcommand_zero_examples():
    """run_subcommand handles examples=0 (skip ensure_examples)."""
    args = argparse.Namespace(benchmark="financebench", examples=0)
    mock_result = MagicMock()
    mock_result.passed = True
    mock_result.metrics = {"accuracy": 0.95}

    with patch("raghub.cli.eval_cmd.FinanceBenchEvaluator") as mock_cls:
        mock_eval = MagicMock()
        mock_eval.benchmark = "financebench"
        mock_cls.return_value = mock_eval
        mock_eval.evaluate = AsyncMock(return_value=[mock_result])

        buf = io.StringIO()
        with redirect_stdout(buf):
            rc = eval_cmd.run_subcommand(args)
    assert rc == 0
    payload = json.loads(buf.getvalue())
    assert payload["summary"]["benchmark"] == "financebench"
    assert payload["summary"]["count"] == 1


def test_eval_run_subcommand_with_examples():
    """run_subcommand calls ensure_examples and limits rows."""
    args = argparse.Namespace(benchmark="financebench", examples=2)
    rows = [{"q": "a"}, {"q": "b"}, {"q": "c"}]
    mock_result = MagicMock()
    mock_result.passed = True
    mock_result.metrics = {"accuracy": 0.95}

    with patch("raghub.cli.eval_cmd.FinanceBenchEvaluator") as mock_cls:
        mock_eval = MagicMock()
        mock_eval.benchmark = "financebench"
        mock_cls.return_value = mock_eval
        mock_eval.ensure_examples.return_value = rows
        mock_eval.evaluate = AsyncMock(return_value=[mock_result, mock_result])

        buf = io.StringIO()
        with redirect_stdout(buf):
            rc = eval_cmd.run_subcommand(args)
    assert rc == 0
    payload = json.loads(buf.getvalue())
    assert payload["summary"]["count"] == 2


# =======================================================================
# common.py  —  line 36
# =======================================================================


def test_load_settings_or_path_toml_legacy_python(tmp_path: Path):
    """load_settings_or_path loads TOML on Python < 3.11 using tomli."""
    cfg = tmp_path / "test.toml"
    cfg.write_text('environment = "development"\nchunk_size_words = 200\n', encoding="utf-8")
    mock_tomli = MagicMock()
    mock_tomli.loads.return_value = {"environment": "development", "chunk_size_words": 200}

    with patch.dict("sys.modules", {"tomli": mock_tomli}):
        with patch.object(sys, "version_info", (3, 10, 0)):
            settings = load_settings_or_path(str(cfg))
    assert settings.chunk_size_words == 200


# =======================================================================
# system.py  —  lines 46–47
# =======================================================================


def test_handle_version_package_not_found():
    """handle_version prints 'unknown' when the package is not installed."""
    with patch("importlib.metadata.version", side_effect=PackageNotFoundError("x")):
        buf = io.StringIO()
        with redirect_stdout(buf):
            rc = system.handle_version(MagicMock())
        assert rc == 0
        assert buf.getvalue().strip() == "unknown"
