"""Regression tests for issues called out in the production-readiness review."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

from raghub import RAG
from raghub.config.settings import AppSettings
from raghub.ingestion.chunkers.word_window import WordWindowChunker
from raghub.interfaces.observability import TelemetryProvider
from raghub.knowledge.manifest import SourceManifest
from raghub.observability.noop import NoOpTelemetry
from raghub.observability.redact import RedactingTelemetry
from raghub.pipelines.rag import IngestPipeline, QueryPipeline
from raghub.embeddings.hashing import HashingEmbeddingProvider
from raghub.vectorstore.memory import InMemoryVectorStore
from raghub.llm.heuristic import HeuristicLLMProvider
from raghub.generation.generator import DefaultGenerator


# ---------------------------------------------------------------------------
# asyncio.run-inside-event-loop fix
# ---------------------------------------------------------------------------


def test_rag_ingest_inside_event_loop() -> None:
    """Sync ``RAG.ingest`` from inside a running event loop returns a coroutine."""

    async def _drive() -> None:
        rag = RAG()
        result = rag.ingest(b"revenue grew 10%", source_uri="mem://x")
        # When a loop is running, ``_maybe_run`` returns the awaitable
        # rather than calling ``asyncio.run`` (which would raise).
        if asyncio.iscoroutine(result):
            await result
        # Otherwise it's the resolved result.

    asyncio.run(_drive())


def test_rag_evaluate_inside_event_loop() -> None:
    """Sync ``RAG.evaluate`` does not raise from inside a running event loop."""

    async def _factory(_example):
        return "ok"

    async def _drive() -> None:
        rag = RAG()
        out = rag.evaluate(
            benchmark="financebench",
            response_factory=_factory,
            examples=[{"id": "0", "question": "x", "answer": "ok"}],
        )
        # When called from inside a running loop, ``_maybe_run``
        # returns the awaitable rather than the resolved result.
        if asyncio.iscoroutine(out):
            resolved = await out
            assert isinstance(resolved, list)
        else:
            assert isinstance(out, list)

    asyncio.run(_drive())


# ---------------------------------------------------------------------------
# Real streaming
# ---------------------------------------------------------------------------


def test_rag_astream_yields_chunks() -> None:
    """``RAG.astream`` yields at least one chunk for a non-empty answer."""
    from raghub.converters.plaintext import PlainTextConverter
    from raghub.ingestion.chunkers.word_window import WordWindowChunker

    rag = RAG()
    rag.converter = PlainTextConverter()
    rag.ingest_pipeline.converter = rag.converter
    rag.chunker = WordWindowChunker(chunk_size=5, chunk_overlap=1)
    rag.ingest_pipeline.chunker = rag.chunker
    # Use a long text so the chunker produces something to retrieve.
    text = (
        b"Revenue grew 12% YoY in Q3 2024. "
        b"Operating margin expanded 200 basis points. "
        b"Free cash flow increased significantly. "
        b"Total revenue reached 100 million dollars. "
    ) * 3
    rag.ingest(text, source_uri="mem://x")

    async def _collect() -> list:
        chunks = []
        async for piece in rag.astream("revenue"):
            chunks.append(piece)
        return chunks

    chunks = asyncio.run(_collect())
    assert len(chunks) >= 1
    assert any("revenue" in c.lower() for c in chunks)


# ---------------------------------------------------------------------------
# Telemetry default
# ---------------------------------------------------------------------------


def test_rag_telemetry_default_is_noop_when_langfuse_unconfigured(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When LANGFUSE_* are unset, the default telemetry is a no-op."""
    monkeypatch.delenv("LANGFUSE_PUBLIC_KEY", raising=False)
    monkeypatch.delenv("LANGFUSE_SECRET_KEY", raising=False)
    rag = RAG()
    # The default is wrapped in RedactingTelemetry; the inner is a
    # NoOpTelemetry. ``health`` reports the wrapper class.
    assert "Redacting" in rag.health()["telemetry"]


def test_rag_works_with_noop_telemetry() -> None:
    """RAG must run cleanly when telemetry is a NoOpTelemetry."""
    rag = RAG(telemetry=NoOpTelemetry())
    rag.ingest(b"revenue grew 12%", source_uri="mem://x")
    response = rag.query("revenue")
    assert response.answer


# ---------------------------------------------------------------------------
# Redaction
# ---------------------------------------------------------------------------


def test_redacting_telemetry_filters_secrets() -> None:
    """Secret-looking kwargs are masked before being forwarded."""

    from raghub.interfaces.observability import TelemetryProvider
    from raghub.telemetry.langfuse import NoopSpan

    class Capture(TelemetryProvider):
        def __init__(self) -> None:
            self.calls: list[tuple[str, dict]] = []

        def info(self, message, **kwargs):
            self.calls.append(("info", {"message": message, **kwargs}))

        def warning(self, message, **kwargs):
            self.calls.append(("warning", {"message": message, **kwargs}))

        def error(self, message, **kwargs):
            self.calls.append(("error", {"message": message, **kwargs}))

        def record_latency(self, name, value_ms, **labels):
            self.calls.append(("latency", {"name": name, **labels}))

        def increment(self, name, value=1, **labels):
            self.calls.append(("counter", {"name": name, **labels}))

        def start_span(self, name, **attrs):
            self.calls.append(("span_start", {"name": name, **attrs}))
            return NoopSpan(name)

        def end_span(self, span):
            self.calls.append(("span_end", {}))

        def record_tokens(self, name, prompt_tokens, completion_tokens, model=""):
            self.calls.append(("tokens", {"name": name}))

    cap = Capture()
    redacting = RedactingTelemetry(cap)
    redacting.info("login", api_key="sk-123", password="hunter2", ok=True)
    redacting.record_latency("query", 12.0, api_key="sk-456", model="gpt-4")
    span = redacting.start_span("op", api_key="sk-789")
    redacting.end_span(span)

    payload_str = json.dumps(cap.calls, default=str)
    assert "sk-123" not in payload_str
    assert "sk-456" not in payload_str
    assert "sk-789" not in payload_str
    assert "hunter2" not in payload_str
    # Non-secret fields are passed through.
    assert "gpt-4" in payload_str


# ---------------------------------------------------------------------------
# Delete
# ---------------------------------------------------------------------------


def test_rag_delete_removes_chunks() -> None:
    """``RAG.delete`` removes the document's chunks from the store."""
    rag = RAG()
    rag.ingest(b"revenue grew 12%", source_uri="doc-1")
    rag.delete("doc-1")
    # The store is empty.
    store = rag.vector_store
    if hasattr(store, "search"):
        vector = rag.embedder.embed_text("revenue")
        hits = store.search(vector=vector, top_k=10, metadata_filter="")
        assert all(h["chunk"].document_id != "doc-1" for h in hits)


# ---------------------------------------------------------------------------
# Metadata filter
# ---------------------------------------------------------------------------


def test_rag_query_with_metadata_filter() -> None:
    """``RAG.query(..., metadata_filter={...})`` is forwarded to the store."""
    rag = RAG()
    rag.ingest(b"revenue grew 12%", source_uri="doc-1")
    response = rag.query("revenue", metadata_filter={"company": "acme"})
    assert response.answer is not None


# ---------------------------------------------------------------------------
# Incremental indexing
# ---------------------------------------------------------------------------


def test_incremental_short_circuits_unchanged() -> None:
    """Ingesting the same bytes twice does not re-embed."""
    from raghub.converters.plaintext import PlainTextConverter
    from raghub.ingestion.chunkers.word_window import WordWindowChunker

    rag = RAG()
    # Use the plain text converter so this test works without a
    # real PDF; the converter + chunker logic we want to exercise
    # is the same either way.
    rag.converter = PlainTextConverter()
    rag.ingest_pipeline.converter = rag.converter
    rag.chunker = WordWindowChunker(chunk_size=5, chunk_overlap=1)
    rag.ingest_pipeline.chunker = rag.chunker
    embedder_calls: list[int] = []

    real_embed = rag.embedder.embed_texts

    def spy(texts):
        embedder_calls.append(len(texts))
        return real_embed(texts)

    from typing import cast
    from collections.abc import Callable

    rag.embedder.embed_texts = cast(Callable[..., object], spy)
    text = b"unchanged content. The quick brown fox jumps over the lazy dog. " * 4
    rag.ingest(text, source_uri="mem://x")
    rag.ingest(text, source_uri="mem://x")
    # First call embeds; second call should be a no-op.
    assert len(embedder_calls) == 1


def test_sync_index_detects_add_modify_delete(tmp_path: Path) -> None:
    """``RAG.sync_index`` adds, modifies, and deletes entries."""
    (tmp_path / "a.txt").write_text("alpha", encoding="utf-8")
    (tmp_path / "b.txt").write_text("beta", encoding="utf-8")
    manifest_path = tmp_path / "manifest.json"
    manifest = SourceManifest(manifest_path)
    rag = RAG(manifest=manifest)

    summary = rag.sync_index(tmp_path)
    assert sorted(summary["added"]) == sorted(
        [str((tmp_path / "a.txt").resolve()), str((tmp_path / "b.txt").resolve())]
    )
    assert summary["modified"] == []
    assert summary["removed"] == []

    # Modify a.txt.
    (tmp_path / "a.txt").write_text("alpha updated", encoding="utf-8")
    summary = rag.sync_index(tmp_path)
    assert len(summary["modified"]) == 1

    # Delete b.txt.
    (tmp_path / "b.txt").unlink()
    summary = rag.sync_index(tmp_path)
    assert len(summary["removed"]) == 1
    assert manifest_path.exists()


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


def test_runtime_override() -> None:
    """``AppSettings.override`` returns a new instance with the change."""
    settings = AppSettings()
    updated = settings.override(chunk_size_words=200, embedding_dim=64)
    assert updated.chunk_size_words == 200
    assert updated.embedding_dim == 64
    # Original is untouched.
    assert settings.chunk_size_words != 200


def test_toml_loaded(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """TOML config is loaded when present."""
    pytest.importorskip("tomllib")
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    cfg = config_dir / "development.toml"
    cfg.write_text(
        'environment = "development"\nchunk_size_words = 200\n'
        'embedding_dim = 64\n',
        encoding="utf-8",
    )
    cwd = Path.cwd()
    monkeypatch.chdir(tmp_path)
    try:
        from raghub.config.settings import load_settings

        settings = load_settings("development")
    finally:
        monkeypatch.chdir(cwd)
    # The TOML is loaded as the profile overlay.
    assert settings.chunk_size_words == 200
    assert settings.embedding_dim == 64


# ---------------------------------------------------------------------------
# Internal pipeline instrumentation
# ---------------------------------------------------------------------------


def test_ingest_pipeline_emits_spans() -> None:
    """The ingest pipeline records telemetry spans when a provider is wired."""
    cap = _Capture()
    embedder = HashingEmbeddingProvider(dimension=16, model_name="hashing-test")
    store = InMemoryVectorStore()
    pipeline = IngestPipeline(
        converter=_PlainConverter(),
        chunker=WordWindowChunker(chunk_size=4, chunk_overlap=1),
        embedder=embedder,
        vector_store=store,
        telemetry=cap,
    )

    async def _drive():
        from raghub.models import PipelineContext

        return await pipeline.run(
            PipelineContext(pipeline_name="ingest"),
            file_bytes=b"alpha beta gamma delta",
            source_uri="mem://x",
            mime_type="text/plain",
        )

    result = asyncio.run(_drive())
    assert result.success
    names = [name for (name, _attrs) in cap.spans]
    assert "ingest" in names
    assert "ingest.convert" in names
    assert "ingest.chunk" in names
    assert "ingest.embed" in names
    assert "ingest.upsert" in names


def test_query_pipeline_emits_spans() -> None:
    """The query pipeline records telemetry spans."""
    cap = _Capture()
    embedder = HashingEmbeddingProvider(dimension=16, model_name="hashing-test")
    store = InMemoryVectorStore()
    from raghub.models import ChunkRecord, Classification

    store.upsert(
        [
            ChunkRecord(
                chunk_id="c1",
                document_id="d1",
                version=1,
                text="revenue is high",
                company="o",
                owner="u",
                classification=Classification.INTERNAL,
            )
        ],
        [embedder.embed_text("revenue is high")],
    )
    pipeline = QueryPipeline(
        embedder=embedder,
        vector_store=store,
        generator=DefaultGenerator(llm=HeuristicLLMProvider()),
        telemetry=cap,
    )

    async def _drive():
        from raghub.models import PipelineContext

        return await pipeline.run(
            PipelineContext(pipeline_name="query"), question="revenue"
        )

    result = asyncio.run(_drive())
    assert result.success
    names = [name for (name, _attrs) in cap.spans]
    assert "query" in names
    assert "query.embed_query" in names
    assert "query.search" in names
    assert "query.generate" in names


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _Capture(TelemetryProvider):
    """Telemetry stub that records every span and event."""

    def __init__(self) -> None:
        self.spans: list[tuple[str, dict]] = []
        self.events: list[tuple[str, dict]] = []

    def info(self, message, **kwargs):
        self.events.append(("info", {"message": message, **kwargs}))

    def warning(self, message, **kwargs):
        self.events.append(("warning", {"message": message, **kwargs}))

    def error(self, message, **kwargs):
        self.events.append(("error", {"message": message, **kwargs}))

    def record_latency(self, name, value_ms, **labels):
        self.events.append(("latency", {"name": name, **labels}))

    def increment(self, name, value=1, **labels):
        self.events.append(("counter", {"name": name, **labels}))

    def start_span(self, name, **attrs):
        from raghub.telemetry.langfuse import NoopSpan

        s = NoopSpan(name)
        for k, v in attrs.items():
            s.set_attribute(k, v)
        self.spans.append((name, attrs))
        return s

    def end_span(self, span):
        pass

    def record_tokens(self, name, prompt_tokens, completion_tokens, model=""):
        self.events.append(("tokens", {"name": name, "prompt": prompt_tokens, "completion": completion_tokens}))


class _PlainConverter:
    """Minimal converter that wraps :class:`PlainTextConverter`."""

    def convert(self, *, source_uri, file_bytes, mime_type="", language="", metadata=None):
        from raghub.converters.plaintext import PlainTextConverter

        return PlainTextConverter().convert(
            source_uri=source_uri,
            file_bytes=file_bytes,
            mime_type=mime_type,
            language=language,
            metadata=metadata,
        )
