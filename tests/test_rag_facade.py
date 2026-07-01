"""Tests for the RAG facade and the plugin registry."""

from __future__ import annotations

from pathlib import Path

import pytest

from raghub import RAG
from raghub.plugins.registry import PluginRegistry


def test_rag_default_construction() -> None:
    """A facade can be built with no arguments (offline-only defaults)."""
    rag = RAG()
    assert rag.health()["status"] == "ok"


def test_rag_from_config(tmp_path: Path) -> None:
    """A facade can be built from a YAML config."""
    cfg = tmp_path / "rag.yaml"
    cfg.write_text(
        "environment: development\nchunk_size_words: 200\nchunk_overlap_words: 10\n",
        encoding="utf-8",
    )
    rag = RAG.from_config(cfg)
    assert rag.settings.environment == "development"
    assert rag.settings.chunk_size_words == 200


def test_rag_ingest_query_smoke() -> None:
    """Smoke: ingest plain text and ask a question."""
    from raghub.converters.plaintext import PlainTextConverter
    from raghub.ingestion.chunkers.word_window import WordWindowChunker

    rag = RAG()
    # Use the plain-text converter + a tiny chunker so this smoke
    # test works without a real PDF or LLM endpoint.
    rag.converter = PlainTextConverter()
    rag.ingest_pipeline.converter = rag.converter
    rag.chunker = WordWindowChunker(chunk_size=5, chunk_overlap=1)
    rag.ingest_pipeline.chunker = rag.chunker
    rag.ingest(b"revenue grew by 10% in Q3", source_uri="mem://text")
    response = rag.query("revenue")
    assert response.answer is not None


def test_rag_ingest_rejects_empty_bytes() -> None:
    """Empty bytes raise a clear ``RagHubError``."""
    import asyncio

    from raghub.exceptions import RagHubError

    rag = RAG()
    with pytest.raises(RagHubError, match="empty bytes"):
        rag.ingest(b"", source_uri="mem://empty")
    with pytest.raises(RagHubError, match="empty bytes"):
        asyncio.run(rag.aingest(b"", source_uri="mem://empty"))


def test_rag_evaluate_calls_evaluator() -> None:
    """The evaluate() helper returns a list of EvaluationResult."""
    rag = RAG()

    async def _factory(example):
        return example.get("answer", "")

    results = rag.evaluate(
        benchmark="financebench",
        response_factory=_factory,
        examples=[{"id": "0", "question": "x", "answer": "y"}],
    )
    assert results
    assert results[0].benchmark == "financebench"


def test_rag_evaluate_unknown_benchmark() -> None:
    """An unknown benchmark raises ConfigurationError."""
    from raghub.exceptions import ConfigurationError

    rag = RAG()
    with pytest.raises(ConfigurationError):
        rag.evaluate(benchmark="wat", examples=[])


def test_rag_shutdown_issafe_call() -> None:
    """Calling shutdown() twice should be safe."""
    rag = RAG()
    rag.shutdown()
    rag.shutdown()


def test_plugin_registry_records() -> None:
    """A registry stores registrations by category."""
    reg = PluginRegistry()
    reg.register_factory("noop", lambda: None)
    assert reg.factories["noop"]() is None
    assert reg.discover_entrypoints(group="raghub.plugins.does.not.exist") == 0


# ---------------------------------------------------------------------------
# Coverage for uncovered lines in raghub/api/rag.py
# ---------------------------------------------------------------------------


def test_rag_from_config_toml(tmp_path: Path) -> None:
    """from_config with a .toml file exercises the tomllib path."""
    cfg = tmp_path / "rag.toml"
    cfg.write_text('environment = "development"\n', encoding="utf-8")
    rag = RAG.from_config(cfg)
    assert rag.settings.environment == "development"


def test_rag_initialize() -> None:
    """initialize() calls create_collection / initialize on collaborators."""
    rag = RAG()
    rag.initialize()


def test_rag_shutdown_telemetry_error() -> None:
    """shutdown() swallows telemetry.end_trace() exceptions."""
    rag = RAG()

    class _BadTelemetry:
        @staticmethod
        def end_trace() -> None:
            raise RuntimeError("telemetry crashed")

    rag.telemetry = _BadTelemetry()
    rag.shutdown()


def test_rag_shutdown_async_close() -> None:
    """shutdown() handles collaborators whose close() returns a coroutine."""
    rag = RAG()

    class _AsyncCloser:
        @staticmethod
        async def close() -> None:
            return None

    rag.vector_store = _AsyncCloser()
    rag.knowledge_repo = None
    rag.shutdown()


def test_rag_ingest_directory_sync(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """ingest() with a directory path exercises ingest_directory_sync."""
    rag = RAG()
    (tmp_path / "a.txt").write_bytes(b"hello")
    (tmp_path / "b.txt").write_bytes(b"world")

    from raghub.models import PipelineResult

    async def _mock_run(*args: object, **kwargs: object) -> PipelineResult:
        return PipelineResult(pipeline_id="t", pipeline_name="ingest", success=True, outputs={})

    monkeypatch.setattr(rag.ingest_pipeline, "run", _mock_run)

    result = rag.ingest(tmp_path)
    assert result.success


def test_rag_aingest_directory(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """aingest() with a directory path exercises ingest_directory_async."""
    rag = RAG()
    (tmp_path / "a.txt").write_bytes(b"hello")

    from raghub.models import PipelineResult

    async def _mock_run(*args: object, **kwargs: object) -> PipelineResult:
        return PipelineResult(pipeline_id="t", pipeline_name="ingest", success=True, outputs={})

    monkeypatch.setattr(rag.ingest_pipeline, "run", _mock_run)

    import asyncio

    result = asyncio.run(rag.aingest(tmp_path))
    assert result.success


def test_rag_aquery_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    """aquery() raises RagHubError when the pipeline returns failure."""
    from raghub.exceptions import RagHubError
    from raghub.models import PipelineResult

    rag = RAG()

    async def _mock_run(*args: object, **kwargs: object) -> PipelineResult:
        return PipelineResult(pipeline_id="q", pipeline_name="query", success=False, error="LLM timeout")

    monkeypatch.setattr(rag.query_pipeline, "run", _mock_run)

    import asyncio

    with pytest.raises(RagHubError, match="LLM timeout"):
        asyncio.run(rag.aquery("test question"))


def test_rag_evaluate_without_factory(monkeypatch: pytest.MonkeyPatch) -> None:
    """evaluate() with no response_factory calls aquery internally."""
    from raghub.models import CanonicalResponse

    rag = RAG()

    async def _mock_aquery(*args: object, **kwargs: object) -> CanonicalResponse:
        return CanonicalResponse(answer="42", citations=[])

    monkeypatch.setattr(rag, "aquery", _mock_aquery)

    results = rag.evaluate(
        benchmark="financebench",
        examples=[{"id": "0", "question": "x", "answer": "y"}],
    )
    assert len(results) == 1


def test_rag_evaluate_with_sync_factory() -> None:
    """evaluate() with a sync factory skips await."""
    rag = RAG()

    def _factory(example: dict) -> str:
        return example.get("answer", "")

    results = rag.evaluate(
        benchmark="financebench",
        response_factory=_factory,
        examples=[{"id": "0", "question": "x", "answer": "y"}],
    )
    assert len(results) == 1


def test_rag_sync_index_not_directory(tmp_path: Path) -> None:
    """sync_index() raises when path is not a directory."""
    from raghub.exceptions import RagHubError

    rag = RAG()
    f = tmp_path / "file.txt"
    f.write_text("x")
    with pytest.raises(RagHubError, match="not a directory"):
        rag.sync_index(f)


def test_rag_sync_index_skips_non_files(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """sync_index() skips subdirectories in the rglob loop."""
    rag = RAG()
    (tmp_path / "sub").mkdir()
    (tmp_path / "real.txt").write_text("hello")

    class _MockManifest:
        def __contains__(self, uri: object) -> bool:
            return False

        @staticmethod
        def sources() -> list[str]:
            return []

        @staticmethod
        def save() -> None:
            return None

        @staticmethod
        def record(uri: str, bundle_id: str, checksum: str) -> None:
            return None

    rag.manifest = _MockManifest()
    monkeypatch.setattr(rag, "ingest", lambda *a, **kw: None)

    summary = rag.sync_index(tmp_path)
    assert "added" in summary


def test_rag_sync_index_skips_external_uris(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """sync_index() skips manifest URIs outside the target directory."""
    rag = RAG()
    (tmp_path / "doc.txt").write_text("data")

    class _MockManifest:
            def __contains__(self, uri: object) -> bool:
                return True

            @staticmethod
            def sources() -> list[str]:
                return ["/outside/doc.txt", "/other/path.pdf"]

            @staticmethod
            def save() -> None:
                return None

            @staticmethod
            def remove(uri: str) -> None:
                return None

            @staticmethod
            def record(uri: str, bundle_id: str, checksum: str) -> None:
                return None

            def __getitem__(self, uri: str) -> dict:
                return {"bundle_id": "old", "checksum": "old"}

    rag.manifest = _MockManifest()
    monkeypatch.setattr(rag, "ingest", lambda *a, **kw: None)
    monkeypatch.setattr(rag, "delete", lambda *a, **kw: None)

    summary = rag.sync_index(tmp_path)
    assert len(summary["removed"]) == 0


def test_rag_ingest_async_with_bytes(monkeypatch: pytest.MonkeyPatch) -> None:
    """ingest_async() works with raw bytes and creates background service on demand."""
    class _MockBgService:
        def __init__(self, **kwargs: object) -> None:
            pass

        @staticmethod
        def submit(*args: object, **kwargs: object) -> str:
            return "mock-job-1"

    monkeypatch.setattr("raghub.api.rag.ResumableBackgroundIngestionService", _MockBgService)

    rag = RAG()
    rag.settings.data_dir.mkdir(parents=True, exist_ok=True)

    job_id = rag.ingest_async(b"test content")
    assert job_id == "mock-job-1"
    assert rag.background_ingestion is not None


def test_rag_job_status_no_background() -> None:
    """job_status() returns None when no background service exists."""
    rag = RAG()
    assert rag.background_ingestion is None
    assert rag.job_status("some-job") is None
