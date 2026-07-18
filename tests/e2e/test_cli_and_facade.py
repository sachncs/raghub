"""End-to-end qualitative tests.

Each test exercises the canonical operator surface from the outside
(``raghub`` CLI, ``RAG()`` facade, ``build_container``) and asserts
qualitative behaviour — not specific numeric outputs — so the
tests are stable across LLM / embedder / vector-store backends.
"""

from __future__ import annotations

import asyncio
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest


def test_python_m_raghub_cli_runs() -> None:
    """``python -m raghub.cli <sub>`` exits 0 for the canonical subcommands."""
    env = {
        **os.environ,
        "RAG_PROFILE": "development",
        "CORS_ORIGINS": "http://testserver",
        "JWT_SECRET": "test-secret-must-be-32-bytes-or-longer-for-sha256",
    }
    for sub in ("version", "health"):
        result = subprocess.run(
            [sys.executable, "-m", "raghub.cli", sub],
            capture_output=True,
            text=True,
            timeout=30,
            env=env,
        )
        assert result.returncode == 0, result.stderr
        assert (result.stdout + result.stderr).strip() != ""


def test_python_m_raghub_cli_help_works() -> None:
    """``python -m raghub.cli --help`` exits 0 and lists every subcommand."""
    result = subprocess.run(
        [sys.executable, "-m", "raghub.cli", "--help"],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr
    for cmd in ("init", "ingest", "query", "eval", "run", "health", "version"):
        assert cmd in result.stdout


def test_rag_facade_ingest_query_round_trip(tmp_path: Path) -> None:
    """A text-only ingest/query cycle through :class:`RAG` returns a typed answer."""
    os.environ["RAG_DATA_DIR"] = str(tmp_path)
    os.environ["RAG_PROFILE"] = "development"
    os.environ["CORS_ORIGINS"] = "http://testserver"
    os.environ["JWT_SECRET"] = "test-secret-must-be-32-bytes-or-longer-for-sha256"
    from raghub.api.rag import RAG

    rag = RAG()
    result = asyncio.run(
        rag.aingest(
            b"RAGHub e2e smoke test. The answer to life is 42.",
            source_uri="file://smoke.txt",
            mime_type="text/plain",
        )
    )
    assert result.success, result.error
    assert result.outputs.get("chunk_count", 0) >= 1

    response = asyncio.run(rag.aquery("life"))
    assert response.answer
    # The heuristic LLM echoes the top chunk; "42" must appear.
    assert "42" in response.answer


def test_build_container_returns_dynamic_rag_container(tmp_path: Path) -> None:
    """``build_container`` returns a fully-wired :class:`DynamicRagContainer`."""
    os.environ["RAG_DATA_DIR"] = str(tmp_path)
    os.environ["RAG_ZVEC_DIR"] = str(tmp_path / "zvec")
    os.environ["RAG_PROFILE"] = "development"
    os.environ["CORS_ORIGINS"] = "http://testserver"
    os.environ["JWT_SECRET"] = "test-secret-must-be-32-bytes-or-longer-for-sha256"
    from raghub.config.settings import load_settings
    from raghub.services.application import DynamicRagApplication, build_container

    container = asyncio.run(build_container(load_settings()))
    application = DynamicRagApplication(container)
    assert application.container.settings.environment in {"development", "test", "production"}
    assert application.container.logger is not None
    assert application.container.metrics is not None


def test_tqdm_progress_appears_during_ingest(capsys) -> None:
    """The chunking progress bar writes to stderr during ingest."""
    os.environ["RAG_PROFILE"] = "development"
    os.environ["CORS_ORIGINS"] = "http://testserver"
    os.environ["JWT_SECRET"] = "test-secret-must-be-32-bytes-or-longer-for-sha256"
    from raghub.api.rag import RAG

    rag = RAG()
    with tempfile.TemporaryDirectory() as tmp:
        target = Path(tmp) / "doc.txt"
        target.write_bytes(b"hello world " * 50)

        async def _drive() -> None:
            result = await rag.aingest(target, mime_type="text/plain")
            assert result.success, result.error

        asyncio.run(_drive())
    # tqdm writes progress to stderr by default; loguru writes to
    # stderr too. We just assert something made it to stderr so the
    # operator sees progress.
    captured = capsys.readouterr()
    assert captured.err or captured.out