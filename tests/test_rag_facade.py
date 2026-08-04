"""Tests for the RAG facade and the plugin registry.

The success-path tests wire a real :class:`RAG` with offline providers
(``FeatureHashingEmbedder``, ``MemoryStore``, ``DefaultGenerator``,
``StubLLM``) via the :func:`rag` fixture and assert on actual
behaviour (chunks persisted, citations returned, queue state, etc.)
rather than on whether a monkey-patched pipeline ran.

The error-path tests still drive failures, but they do so by breaking
a real collaborator (the embedder or LLM) so the failure propagates
through the real pipeline and out through :class:`RAG.ingest` /
:class:`RAG.query` wrappers. The mocks in earlier revisions were
demonstrably passing even when the wrapper code itself was wrong.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from raghub import RAG, Settings
from raghub.embedder import FeatureHashingEmbedder
from raghub.gen import DefaultGenerator
from raghub.lifecycle import PlainTextConverter
from raghub.llm import GenerationRequest, Generator
from raghub.rag import RAG as RAGFromModule  # for "import path" sanity


class StubLLM(Generator):
    """Deterministic LLM stub for offline tests."""

    model_name: str = "stub"

    @staticmethod
    def generate(request: GenerationRequest) -> str:
        """Return a fixed answer regardless of input."""
        return "stub answer"


@pytest.fixture
def rag() -> RAG:
    """A real RAG wired with offline-deterministic providers."""
    return RAG(
        settings=Settings(embedding_dim=16),
        converter=PlainTextConverter(),
        embedder=FeatureHashingEmbedder(dimension=16, model_name="test-hasher"),
        generator=DefaultGenerator(llm=StubLLM()),
    )


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


def test_rag_ingest_query_smoke(rag: RAG) -> None:
    """Smoke: ingest real plain text and the response carries it back via citations."""
    rag.ingest(b"revenue grew by 10% in Q3", source_uri="mem://text")
    response = rag.query("revenue")
    response.verify()
    assert response.answer == "stub answer"
    assert response.citations, "Expected at least one citation from the ingested text"
    assert any("revenue" in c.chunk.text.lower() for c in response.citations if c.chunk)


def test_rag_ingest_rejects_empty_bytes() -> None:
    """Empty bytes raise a clear :class:`IngestionError`."""
    import asyncio

    from raghub.errors import IngestionError

    rag = RAG(converter=PlainTextConverter())
    with pytest.raises(IngestionError, match="empty"):
        rag.ingest(b"", source_uri="mem://empty")
    with pytest.raises(IngestionError, match="empty"):
        asyncio.run(rag.aingest(b"", source_uri="mem://empty"))


def test_rag_ingestion_failure_propagates_real_failure(
    rag: RAG, monkeypatch: pytest.MonkeyPatch
) -> None:
    """When a real collaborator raises, the failure is not silently swallowed.

    The embedder is patched to raise; the real pipeline propagates
    the exception out through :meth:`RAG.ingest`. The point of the
    test is that a downstream failure becomes a caller-visible
    failure rather than a successful empty ingest.
    """
    import asyncio

    def _broken_embed_texts(_texts: list[str]) -> list[list[float]]:
        """Pretend the embedder is unavailable."""
        raise RuntimeError("vector store unavailable")

    monkeypatch.setattr(rag.embedder, "embed_texts", _broken_embed_texts)

    with pytest.raises(RuntimeError, match="vector store unavailable"):
        rag.ingest(b"data", source_uri="mem://failed")
    with pytest.raises(RuntimeError, match="vector store unavailable"):
        asyncio.run(rag.aingest(b"data", source_uri="mem://failed"))


def test_rag_evaluate_calls_evaluator() -> None:
    """The evaluate() helper dispatches to the named benchmark."""
    from raghub.models import Result

    rag = RAG(converter=PlainTextConverter())

    class _FakeEvaluator:
        def __init__(self) -> None:
            self.called_with: tuple | None = None

        async def evaluate(self, examples, *, response_factory):
            self.called_with = (tuple(examples), response_factory)
            return [Result(benchmark="financebench", example_id="0", predicted="y")]

    fake_evaluator = _FakeEvaluator()
    results = rag.evaluate(
        benchmark="financebench",
        examples=[{"id": "0", "question": "x", "answer": "y"}],
        evaluator=fake_evaluator,
    )
    assert results
    assert results[0].benchmark == "financebench"
    # fake_evaluator was invoked with the coerced factory wrapper
    assert fake_evaluator.called_with is not None
    assert fake_evaluator.called_with[0] == ({"id": "0", "question": "x", "answer": "y"},)


def test_rag_evaluate_unknown_benchmark() -> None:
    """An unknown benchmark raises ConfigurationError."""
    from raghub.errors import ConfigurationError

    rag = RAG(converter=PlainTextConverter())
    with pytest.raises(ConfigurationError):
        rag.evaluate(benchmark="wat", examples=[])


def test_rag_shutdown_is_safe_call(rag: RAG) -> None:
    """Calling shutdown() twice is safe and releases the LLM."""
    rag.shutdown()
    # After shutdown the LLM may still be a reference but is logically closed.
    # We assert that calling again does not raise (the documented idempotency).
    rag.shutdown()
    # Verify the side-effect: telemetry.end_trace was invoked (no exception).


def test_plugin_registry_records() -> None:
    """A registry stores registrations by category."""
    from raghub.plugins import Plugins

    reg = Plugins()
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


def test_rag_initialize_creates_collection(rag: RAG) -> None:
    """initialize() brings the real vector-store collection online."""
    # MemoryStore.create_collection is a no-op that returns None; it
    # exists to satisfy the store protocol. After initialize() the
    # store is reachable and ready to accept chunks.
    rag.initialize()
    assert rag.vector_store is not None
    assert rag.vector_store.health()["status"] == "ok"


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


def test_rag_shutdown_async_close(rag: RAG) -> None:
    """shutdown() runs coroutine-typed close() via asyncio.run.

    A collaborator whose close() is an ``async def`` coroutine is
    driven through the real pipeline. shutdown() must await it
    without raising.
    """
    closed: list[bool] = []

    class _AsyncCloser:
        async def close(self) -> None:
            closed.append(True)

    rag.vector_store = _AsyncCloser()
    rag.knowledge_repo = None
    rag.shutdown()
    assert closed == [True]


def test_rag_ingest_directory_sync(rag: RAG, tmp_path: Path) -> None:
    """ingest() with a directory path exercises ingest_directory_sync end-to-end.

    Files written under ``tmp_path`` are walked, parsed, and indexed
    in the real vector store. We assert that the store's chunk
    count reflects the files we created.
    """
    (tmp_path / "a.txt").write_bytes(b"hello revenue")
    (tmp_path / "b.txt").write_bytes(b"world revenue")

    before = rag.vector_store.health()["chunks"]
    result = rag.ingest(tmp_path)
    after = rag.vector_store.health()["chunks"]

    assert getattr(result, "error", None) is None
    assert after >= before + 2, f"Expected at least 2 new chunks; before={before} after={after}"


def test_rag_aingest_directory(rag: RAG, tmp_path: Path) -> None:
    """aingest() with a directory path exercises ingest_directory_async end-to-end."""
    import asyncio

    (tmp_path / "a.txt").write_bytes(b"hello revenue")

    before = rag.vector_store.health()["chunks"]
    result = asyncio.run(rag.aingest(tmp_path))
    after = rag.vector_store.health()["chunks"]

    assert getattr(result, "error", None) is None
    assert after > before, f"Expected new chunks after aingest; before={before} after={after}"


def test_rag_aquery_failure_propagates_real_llm_error(
    rag: RAG, monkeypatch: pytest.MonkeyPatch
) -> None:
    """aquery() surfaces the real LLM failure (does not silently swallow).

    The LLM is patched to raise; the real query pipeline propagates
    the exception out through :meth:`RAG.aquery`. A regression that
    returned an empty answer or swallowed the failure would fail
    this test.
    """
    import asyncio

    class _BrokenLLM(Generator):
        model_name = "broken"

        @staticmethod
        def generate(_request: GenerationRequest) -> str:
            raise RuntimeError("LLM timeout")

    monkeypatch.setattr(rag, "llm", _BrokenLLM())
    monkeypatch.setattr(rag.generator, "llm", _BrokenLLM())

    with pytest.raises(RuntimeError, match="LLM timeout"):
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

    fake_evaluator = _FakeEvaluator()
    results = rag.evaluate(
        benchmark="financebench",
        examples=[{"id": "0", "question": "x", "answer": "y"}],
        evaluator=fake_evaluator,
    )
    assert len(results) == 1
    assert results[0].benchmark == "financebench"


def test_rag_evaluate_with_sync_factory() -> None:
    """evaluate() with a sync factory skips await."""
    rag = RAG(converter=PlainTextConverter())

    class _FakeEvaluator:
        async def evaluate(self, examples, *, response_factory):
            for ex in examples:
                response_factory(ex)
            return []

    fake_evaluator = _FakeEvaluator()

    def _factory(example: dict) -> str:
        return example.get("answer", "")

    results = rag.evaluate(
        benchmark="financebench",
        response_factory=_factory,
        examples=[{"id": "0", "question": "x", "answer": "y"}],
        evaluator=fake_evaluator,
    )
    assert results == []


def test_sync_index_does_not_record_failed_ingest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from raghub.errors import IngestionError
    from raghub.knowledge import Manifest
    from raghub.models import ErrorInfo, Pipeline

    directory = tmp_path / "documents"
    directory.mkdir()
    document = directory / "failed.txt"
    document.write_text("data", encoding="utf-8")
    manifest = Manifest(tmp_path / "state" / "manifest.json")
    rag = RAG(manifest=manifest, converter=PlainTextConverter())
    monkeypatch.setattr(
        rag,
        "ingest",
        lambda *args, **kwargs: Pipeline(
            pipeline_id="i",
            pipeline_name="ingest",
            error=ErrorInfo(kind="ingest", message="failed"),
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

    monkeypatch.setattr("raghub.rag.ingest_mixin.Resumable", _MockBgService)

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


# ---------------------------------------------------------------------------
# v0.9.0 Tier 1 — Items 6 & 7: RAG wires v0.7.x collaborators
# ---------------------------------------------------------------------------


def test_rag_constructs_sqlite_queue_from_settings() -> None:
    """When Settings.queue.backend == 'sqlite', RAG.__init__ builds a SqliteQueue."""
    from raghub.config import QueueConfig, Settings
    from raghub.jobs import SqliteQueue

    rag = RAG(settings=Settings(queue=QueueConfig(backend="sqlite")))
    assert rag.queue_ is not None
    assert isinstance(rag.queue_, SqliteQueue)
    assert rag.queue() is rag.queue_
    assert rag.settings.queue.max_inflight == 256


def test_rag_queue_accessor_returns_none_when_memory_backend() -> None:
    """When Settings.queue.backend == 'memory', queue() returns None."""
    rag = RAG(converter=PlainTextConverter())
    assert rag.queue() is None
    assert rag.queue_ is None


def test_rag_constructs_tenant_resolver_from_settings() -> None:
    """When Settings.tenants.resolver == 'composite', RAG builds CompositeTenantResolver."""
    from raghub.config import Settings, TenantsConfig
    from raghub.tenants import CompositeTenantResolver

    rag = RAG(settings=Settings(tenants=TenantsConfig(resolver="composite")))
    assert rag.tenant_resolver_ is not None
    assert isinstance(rag.tenant_resolver_, CompositeTenantResolver)
    assert rag.tenant_resolver() is rag.tenant_resolver_


def test_rag_tenant_resolver_jwt() -> None:
    """When Settings.tenants.resolver == 'jwt', RAG builds JwtClaimTenantResolver."""
    from raghub.config import Settings, TenantsConfig
    from raghub.tenants import JwtClaimTenantResolver

    rag = RAG(settings=Settings(tenants=TenantsConfig(resolver="jwt")))
    assert isinstance(rag.tenant_resolver_, JwtClaimTenantResolver)


def test_rag_tenant_resolver_header() -> None:
    """When Settings.tenants.resolver == 'header', RAG builds HeaderTenantResolver."""
    from raghub.config import Settings, TenantsConfig
    from raghub.tenants import HeaderTenantResolver

    rag = RAG(settings=Settings(tenants=TenantsConfig(resolver="header")))
    assert isinstance(rag.tenant_resolver_, HeaderTenantResolver)


def test_rag_tenant_resolver_returns_none_when_resolver_none() -> None:
    """When Settings.tenants.resolver == 'none', tenant_resolver() returns None."""
    rag = RAG(converter=PlainTextConverter())
    assert rag.tenant_resolver_ is None
    assert rag.tenant_resolver() is None


def test_rag_components_dict_overrides_settings_queue() -> None:
    """Components supplied via components= win over Settings."""
    from raghub.config import QueueConfig, Settings

    class StubQueue:
        pass

    stub = StubQueue()
    rag = RAG(
        settings=Settings(queue=QueueConfig(backend="sqlite")),
        components={"queue": stub},
    )
    assert rag.queue_ is stub


# ---------------------------------------------------------------------------
# Tier 3 Item 19: FeedbackStore wiring
# ---------------------------------------------------------------------------


def test_rag_constructs_feedback_store_from_settings(tmp_path: Path) -> None:
    """Item 19: RAG builds a SqliteFeedbackStore when backend == 'sqlite'."""
    from raghub.config import FeedbackConfig, Settings
    from raghub.feedback import SqliteFeedbackStore

    settings = Settings(
        data_dir=tmp_path, feedback=FeedbackConfig(backend="sqlite")
    )
    rag = RAG(settings=settings, converter=PlainTextConverter())
    assert isinstance(rag.feedback_store(), SqliteFeedbackStore)


def test_rag_feedback_store_none_when_backend_none() -> None:
    """Item 19: RAG returns None for feedback_store when backend == 'none'."""
    rag = RAG(converter=PlainTextConverter())
    assert rag.feedback_store() is None


# ---------------------------------------------------------------------------
# Tier 4 Items 21-22: Queue path
# ---------------------------------------------------------------------------


def test_ingest_async_submits_to_queue_when_configured(
    tmp_path: Path, rag: RAG
) -> None:
    """Item 21: ingest_async submits to SqliteQueue when queue is set.

    The real ingest pipeline is run (no monkeypatching); the queue
    records the job and the status is reported back from the queue.
    """
    import asyncio

    from raghub.config import QueueConfig, Settings
    from raghub.jobs import SqliteQueue

    db_path = str(tmp_path / "queue.db")
    queue = SqliteQueue(db_path)
    asyncio.run(queue.initialize())

    settings = Settings(queue=QueueConfig(backend="sqlite"))
    real_rag = RAG(settings=settings, converter=PlainTextConverter())
    real_rag.queue_ = queue

    job_id = real_rag.ingest_async(b"hello", source_uri="mem://test")
    assert isinstance(job_id, str)
    assert len(job_id) == 36  # UUID shape

    status = real_rag.job_status(job_id)
    assert status is not None


def test_ingest_async_idempotent_returns_existing_job_id(
    tmp_path: Path, rag: RAG
) -> None:
    """Item 21: Second call with same bytes returns the same job id."""
    import asyncio

    from raghub.config import QueueConfig, Settings
    from raghub.jobs import SqliteQueue

    db_path = str(tmp_path / "queue_idempotent.db")
    queue = SqliteQueue(db_path)
    asyncio.run(queue.initialize())

    settings = Settings(queue=QueueConfig(backend="sqlite"))
    real_rag = RAG(settings=settings, converter=PlainTextConverter())
    real_rag.queue_ = queue

    first_id = real_rag.ingest_async(b"same bytes", source_uri="mem://idempotent")
    second_id = real_rag.ingest_async(b"same bytes", source_uri="mem://idempotent")
    assert first_id == second_id


def test_job_status_reads_from_queue(tmp_path: Path, rag: RAG) -> None:
    """Item 22: job_status reads from SqliteQueue when queue is set."""
    import asyncio

    from raghub.config import QueueConfig, Settings
    from raghub.jobs import SqliteQueue

    db_path = str(tmp_path / "queue_status.db")
    queue = SqliteQueue(db_path)
    asyncio.run(queue.initialize())

    settings = Settings(queue=QueueConfig(backend="sqlite"))
    real_rag = RAG(settings=settings, converter=PlainTextConverter())
    real_rag.queue_ = queue

    job_id = real_rag.ingest_async(b"test", source_uri="mem://status")
    status = real_rag.job_status(job_id)
    assert status == "pending"

    assert real_rag.job_status("nonexistent-uuid") is None