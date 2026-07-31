"""Tests for the RAG facade and the plugin registry."""

from __future__ import annotations

from pathlib import Path

import pytest

from raghub.lifecycle import PlainTextConverter
from raghub.rag import RAG


def test_rag_default_construction() -> None:
    """A facade can be built with no arguments (offline-only defaults)."""
    rag = RAG(converter=PlainTextConverter())
    assert rag.health()["status"] == "ok"


def test_rag_from_config(tmp_path: Path) -> None:
    """A facade can be built from a YAML config."""
    cfg = tmp_path / "rag.yaml"
    cfg.write_text(
        "environment: development\nchunk_size_words: 200\nchunk_overlap_words: 10\n",
        encoding="utf-8",
    )
    rag = RAG.from_config(cfg)
    rag.converter = PlainTextConverter()
    rag.ingest_pipeline.converter = rag.converter
    assert rag.settings.environment == "development"
    assert rag.settings.chunk_size_words == 200


def test_rag_ingest_query_smoke() -> None:
    """Smoke: ingest plain text and ask a question."""
    from raghub.lifecycle import PlainTextConverter

    rag = RAG(converter=PlainTextConverter())
    rag.converter = PlainTextConverter()
    rag.ingest_pipeline.converter = rag.converter
    rag.ingest(b"revenue grew by 10% in Q3", source_uri="mem://text")
    response = rag.query("revenue")
    assert response.answer is not None


def test_rag_ingest_rejects_empty_bytes() -> None:
    """Empty bytes raise a clear :class:`IngestionError`."""
    import asyncio

    from raghub.errors import IngestionError

    rag = RAG(converter=PlainTextConverter())
    with pytest.raises(IngestionError, match="empty"):
        rag.ingest(b"", source_uri="mem://empty")
    with pytest.raises(IngestionError, match="empty"):
        asyncio.run(rag.aingest(b"", source_uri="mem://empty"))


def test_rag_ingestion_failure_raises_typed_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import asyncio

    from raghub.errors import IngestionError
    from raghub.models import PipelineResult

    rag = RAG(converter=PlainTextConverter())

    async def fail_run(*args: object, **kwargs: object) -> PipelineResult:
        return PipelineResult(
            pipeline_id="i",
            pipeline_name="ingest",
            success=False,
            error="vector store unavailable",
        )

    monkeypatch.setattr(rag.ingest_pipeline, "run", fail_run)
    with pytest.raises(IngestionError, match="vector store unavailable"):
        rag.ingest(b"data", source_uri="mem://failed")
    with pytest.raises(IngestionError, match="vector store unavailable"):
        asyncio.run(rag.aingest(b"data", source_uri="mem://failed"))


def test_rag_evaluate_calls_evaluator() -> None:
    """The evaluate() helper dispatches to the named benchmark."""
    rag = RAG(converter=PlainTextConverter())

    class _FakeEvaluator:
        def __init__(self) -> None:
            self.called_with: tuple | None = None

        async def evaluate(self, examples, *, response_factory):
            self.called_with = (tuple(examples), response_factory)
            from raghub.models import Result

            return [Result(benchmark="financebench", example_id="0", predicted="y")]

    _monkey = rag.evaluate.__func__.__globals__.get("Finance")
    original = rag.evaluate.__globals__.get("Finance")
    rag_evaluate_globals = rag.evaluate.__globals__
    rag_evaluate_globals["Finance"] = _FakeEvaluator
    try:
        results = rag.evaluate(
            benchmark="financebench",
            examples=[{"id": "0", "question": "x", "answer": "y"}],
        )
    finally:
        rag_evaluate_globals["Finance"] = original
    assert results
    assert results[0].benchmark == "financebench"


def test_rag_evaluate_unknown_benchmark() -> None:
    """An unknown benchmark raises ConfigurationError."""
    from raghub.errors import ConfigurationError

    rag = RAG(converter=PlainTextConverter())
    with pytest.raises(ConfigurationError):
        rag.evaluate(benchmark="wat", examples=[])


def test_rag_shutdown_is_safe_call() -> None:
    """Calling shutdown() twice should be safe."""
    rag = RAG(converter=PlainTextConverter())
    rag.shutdown()
    rag.shutdown()


def test_plugin_registry_records() -> None:
    """A registry stores registrations by category."""
    from raghub.plugins import PluginRegistry

    reg = PluginRegistry()
    reg.register_factory("noop", lambda: None)
    assert reg.factories["noop"]() is None
    assert reg.discover_entrypoints(group="raghub.plugins.does.not.exist") == 0


# ---------------------------------------------------------------------------
# Coverage for uncovered lines in raghub/rag.py
# ---------------------------------------------------------------------------


def test_rag_from_config_toml(tmp_path: Path) -> None:
    """from_config with a .toml file exercises the tomllib path."""
    cfg = tmp_path / "rag.toml"
    cfg.write_text('environment = "development"\n', encoding="utf-8")
    rag = RAG.from_config(cfg)
    rag.converter = PlainTextConverter()
    rag.ingest_pipeline.converter = rag.converter
    assert rag.settings.environment == "development"


def test_rag_initialize() -> None:
    """initialize() calls create_collection / initialize on collaborators."""
    rag = RAG(converter=PlainTextConverter())
    rag.initialize()


def test_rag_shutdown_telemetry_error() -> None:
    """shutdown() surfaces telemetry.end_trace() failures (no swallowing)."""
    rag = RAG(converter=PlainTextConverter())

    class _BadTelemetry:
        @staticmethod
        def end_trace() -> None:
            raise RuntimeError("telemetry crashed")

    rag.telemetry = _BadTelemetry()
    with pytest.raises(RuntimeError, match="telemetry crashed"):
        rag.shutdown()


def test_rag_shutdown_async_close() -> None:
    """shutdown() runs coroutine-typed close() via asyncio.run."""
    rag = RAG(converter=PlainTextConverter())

    class _AsyncCloser:
        @staticmethod
        async def close() -> None:
            return None

    rag.vector_store = _AsyncCloser()
    rag.knowledge_repo = None
    rag.shutdown()


def test_rag_ingest_directory_sync(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """ingest() with a directory path exercises ingest_directory_sync."""
    rag = RAG(converter=PlainTextConverter())
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
    import asyncio

    rag = RAG(converter=PlainTextConverter())
    (tmp_path / "a.txt").write_bytes(b"hello")

    from raghub.models import PipelineResult

    async def _mock_run(*args: object, **kwargs: object) -> PipelineResult:
        return PipelineResult(pipeline_id="t", pipeline_name="ingest", success=True, outputs={})

    monkeypatch.setattr(rag.ingest_pipeline, "run", _mock_run)

    result = asyncio.run(rag.aingest(tmp_path))
    assert result.success


def test_rag_aquery_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    """aquery() raises RagHubError when the pipeline returns failure."""
    import asyncio

    from raghub.errors import RagHubError
    from raghub.models import PipelineResult

    rag = RAG(converter=PlainTextConverter())

    async def _mock_run(*args: object, **kwargs: object) -> PipelineResult:
        return PipelineResult(
            pipeline_id="q", pipeline_name="query", success=False, error="LLM timeout"
        )

    monkeypatch.setattr(rag.query_pipeline, "run", _mock_run)

    with pytest.raises(RagHubError, match="LLM timeout"):
        asyncio.run(rag.aquery("test question"))


def test_rag_evaluate_without_factory(monkeypatch: pytest.MonkeyPatch) -> None:
    """evaluate() with no response_factory calls aquery internally."""
    from raghub.models import Response, Result

    rag = RAG(converter=PlainTextConverter())

    async def _mock_aquery(*args: object, **kwargs: object) -> Response:
        return Response(answer="42", citations=[])

    monkeypatch.setattr(rag, "aquery", _mock_aquery)

    class _FakeEvaluator:
        async def evaluate(self, examples, *, response_factory):
            for ex in examples:
                await response_factory(ex)
            return [Result(benchmark="financebench", example_id="0", predicted="y")]

    globs = rag.evaluate.__globals__
    original = globs["Finance"]
    globs["Finance"] = _FakeEvaluator
    try:
        results = rag.evaluate(
            benchmark="financebench",
            examples=[{"id": "0", "question": "x", "answer": "y"}],
        )
    finally:
        globs["Finance"] = original
    assert len(results) == 1


def test_rag_evaluate_with_sync_factory() -> None:
    """evaluate() with a sync factory skips await."""
    rag = RAG(converter=PlainTextConverter())

    class _FakeEvaluator:
        async def evaluate(self, examples, *, response_factory):
            for ex in examples:
                response_factory(ex)
            return []

    globs = rag.evaluate.__globals__
    original = globs["Finance"]
    globs["Finance"] = _FakeEvaluator
    try:

        def _factory(example: dict) -> str:
            return example.get("answer", "")

        results = rag.evaluate(
            benchmark="financebench",
            response_factory=_factory,
            examples=[{"id": "0", "question": "x", "answer": "y"}],
        )
    finally:
        globs["Finance"] = original
    assert results == []


def test_sync_index_does_not_record_failed_ingest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from raghub.errors import IngestionError
    from raghub.knowledge import Manifest
    from raghub.models import PipelineResult

    directory = tmp_path / "documents"
    directory.mkdir()
    document = directory / "failed.txt"
    document.write_text("data", encoding="utf-8")
    manifest = Manifest(tmp_path / "state" / "manifest.json")
    rag = RAG(manifest=manifest, converter=PlainTextConverter())
    monkeypatch.setattr(
        rag,
        "ingest",
        lambda *args, **kwargs: PipelineResult(
            pipeline_id="i",
            pipeline_name="ingest",
            success=False,
            error="failed",
        ),
    )

    with pytest.raises(IngestionError, match="failed"):
        rag.sync_index(directory)

    assert str(document.resolve()) not in manifest
    assert not manifest.path.exists()


def test_rag_ingest_async_with_bytes(monkeypatch: pytest.MonkeyPatch) -> None:
    """ingest_async() works with raw bytes and creates background service on demand."""

    class _MockBgService:
        def __init__(self, **kwargs: object) -> None:
            pass

        @staticmethod
        def submit(*args: object, **kwargs: object) -> str:
            return "mock-job-1"

    monkeypatch.setattr("raghub.rag.Resumable", _MockBgService)

    rag = RAG(converter=PlainTextConverter())
    rag.settings.data_dir.mkdir(parents=True, exist_ok=True)

    job_id = rag.ingest_async(b"test content")
    assert job_id == "mock-job-1"
    assert rag.background_ingestion is not None


def test_rag_job_status_no_background() -> None:
    """job_status() returns None when no background service exists."""
    rag = RAG(converter=PlainTextConverter())
    assert rag.background_ingestion is None
    assert rag.job_status("some-job") is None
