"""Comprehensive tests for observability, knowledge, workers, and API modules."""

from __future__ import annotations

import json
from concurrent.futures import Future
from unittest.mock import MagicMock, patch

import pytest

from raghub.exceptions import KnowledgeError
from raghub.knowledge.okf import from_okf, loads, to_okf
from raghub.models import (
    BlockKind,
    DocumentBlock,
    DocumentSection,
    KnowledgeBundle,
    PipelineResult,
)
from raghub.services.workers import (
    InMemoryTaskQueue,
    SynchronousWorker,
    ThreadPoolWorker,
)


# =======================================================================
# metrics.py
# =======================================================================


class TestNullMetrics:
    def test_record_latency_noops(self):
        from raghub.observability.metrics import NullMetrics

        m = NullMetrics()
        assert m.record_latency("x", 1.0) is None

    def test_increment_noops(self):
        from raghub.observability.metrics import NullMetrics

        m = NullMetrics()
        assert m.increment("x", 1) is None

    def test_increment_with_labels(self):
        from raghub.observability.metrics import NullMetrics

        m = NullMetrics()
        assert m.increment("x", 5, error_type="test") is None


class TestPrometheusMetrics:
    def test_register_app_with_fastapi(self):
        from fastapi import FastAPI
        from raghub.observability.metrics import PrometheusMetrics

        app = FastAPI()
        PrometheusMetrics(app=app)
        assert any(r.path == "/metrics" for r in app.routes)

    def test_register_app_with_non_fastapi(self):
        from raghub.observability.metrics import PrometheusMetrics

        metrics = PrometheusMetrics()
        metrics.register_app(None)  # no-op, not FastAPI
        metrics.register_app(object())  # no-op, not FastAPI

    def test_record_latency(self):
        from raghub.observability.metrics import PrometheusMetrics

        m = PrometheusMetrics()
        m.record_latency("test_latency", 42.0)
        m.record_latency("test_latency", 100.0, extra_label="val")

    def test_increment_counter(self):
        from raghub.observability.metrics import PrometheusMetrics

        m = PrometheusMetrics()
        m.increment("test_error", 1)
        m.increment("test_error", 3)

    def test_record_query(self):
        from raghub.observability.metrics import PrometheusMetrics

        m = PrometheusMetrics()
        m.record_query(150.0, 10)
        m.record_query(300.0, 5)

    def test_record_ingestion(self):
        from raghub.observability.metrics import PrometheusMetrics

        m = PrometheusMetrics()
        m.record_ingestion(600.0, 42)

    def test_record_auth(self):
        from raghub.observability.metrics import PrometheusMetrics

        m = PrometheusMetrics()
        m.record_auth(12.0, True)
        m.record_auth(8.0, False)

    def test_record_error(self):
        from raghub.observability.metrics import PrometheusMetrics

        m = PrometheusMetrics()
        m.record_error("rate_limit")
        m.record_error("timeout")

    def test_duplicate_init_idempotent(self):
        from prometheus_client import REGISTRY
        from raghub.observability.metrics import PrometheusMetrics

        # Clear registry first
        collectors = list(REGISTRY._names_to_collectors.keys())
        for name in collectors:
            if name.startswith("raghub_"):
                REGISTRY._names_to_collectors.pop(name, None)

        m1 = PrometheusMetrics()
        m2 = PrometheusMetrics()  # should not raise
        assert m1.query_duration is m2.query_duration


class TestPrometheusMetricsWithMockApp:
    def test_register_app_with_mock_fastapi(self):
        from raghub.observability.metrics import PrometheusMetrics

        metrics = PrometheusMetrics()
        # register_app with non-FastAPI objects should be a no-op
        metrics.register_app(None)
        metrics.register_app(object())

    def test_register_app_with_fastapi_mock(self):
        from raghub.observability.metrics import PrometheusMetrics

        metrics = PrometheusMetrics()
        # Verify it does not crash when app is not FastAPI
        mock_app = MagicMock()
        # isinstance(mock_app, FastAPI) will be False, so it's still a no-op
        metrics.register_app(mock_app)
        mock_app.get.assert_not_called()


# =======================================================================
# redact.py
# =======================================================================


def test_scrub_secrets():
    from raghub.observability.redact import scrub_secrets

    result = scrub_secrets({"api_key": "sk-123", "safe": "hello"})
    assert result == {"api_key": "***", "safe": "hello"}


def test_scrub_secrets_case_insensitive():
    from raghub.observability.redact import scrub_secrets

    result = scrub_secrets({"API_KEY": "val", "Password": "secret"})
    assert result == {"API_KEY": "***", "Password": "***"}


def test_scrub_secrets_nested_dict():
    from raghub.observability.redact import scrub_secrets

    result = scrub_secrets({"outer": {"inner_password": "s3cret"}})
    assert result == {"outer": {"inner_password": "***"}}


def test_scrub_secrets_deeply_nested():
    from raghub.observability.redact import scrub_secrets

    result = scrub_secrets({"a": {"b": {"access_token": "tok"}}})
    assert result == {"a": {"b": {"access_token": "***"}}}


def test_scrub_secrets_authorization():
    from raghub.observability.redact import scrub_secrets

    result = scrub_secrets({"authorization": "Bearer tok"})
    assert result == {"authorization": "***"}


def test_scrub_secrets_jwt():
    from raghub.observability.redact import scrub_secrets

    result = scrub_secrets({"jwt": "eyJ.eyJ.sig"})
    assert result == {"jwt": "***"}


def test_scrub_secrets_apikey():
    from raghub.observability.redact import scrub_secrets

    result = scrub_secrets({"apikey": "abc123"})
    assert result == {"apikey": "***"}


class TestRedactingTelemetry:
    @pytest.fixture
    def inner(self):
        return MagicMock()

    @pytest.fixture
    def telemetry(self, inner):
        from raghub.observability.redact import RedactingTelemetry

        return RedactingTelemetry(inner)

    def test_info_redacts(self, telemetry, inner):
        telemetry.info("hello", api_key="sk-123", user="alice")
        inner.info.assert_called_once_with("hello", api_key="***", user="alice")

    def test_warning_redacts(self, telemetry, inner):
        telemetry.warning("warn", password="p@ss")
        inner.warning.assert_called_once_with("warn", password="***")

    def test_error_redacts(self, telemetry, inner):
        telemetry.error("boom", secret="s3cret")
        inner.error.assert_called_once_with("boom", secret="***")

    def test_record_latency_redacts(self, telemetry, inner):
        telemetry.record_latency("query", 1.0, access_token="tok")
        inner.record_latency.assert_called_once_with("query", 1.0, access_token="***")

    def test_increment_redacts(self, telemetry, inner):
        telemetry.increment("errors", 1, api_key="key")
        inner.increment.assert_called_once_with("errors", 1, api_key="***")

    def test_start_span_redacts(self, telemetry, inner):
        telemetry.start_span("op", refresh_token="rtok")
        inner.start_span.assert_called_once_with("op", refresh_token="***")

    def test_end_span(self, telemetry, inner):
        span = MagicMock()
        telemetry.end_span(span)
        inner.end_span.assert_called_once_with(span)

    def test_record_tokens(self, telemetry, inner):
        telemetry.record_tokens("gpt-4", 100, 50, "gpt-4")
        inner.record_tokens.assert_called_once_with("gpt-4", 100, 50, "gpt-4")

    def test_record_tokens_default_model(self, telemetry, inner):
        telemetry.record_tokens("gpt-4", 100, 50)
        inner.record_tokens.assert_called_once_with("gpt-4", 100, 50, "")


# =======================================================================
# okf.py
# =======================================================================


def test_from_okf_invalid_json_string():
    with pytest.raises(KnowledgeError, match="Invalid OKF JSON"):
        from_okf("{bad json")


def test_from_okf_non_dict_payload():
    with pytest.raises(KnowledgeError, match="must be a dict"):
        from_okf("[]")


def test_from_okf_parses_json_string():
    payload = json.dumps({"source_uri": "s3://bucket/doc.pdf"})
    bundle = from_okf(payload)
    assert bundle.source_uri == "s3://bucket/doc.pdf"


def test_from_okf_section_not_dict():
    payload = {"source_uri": "x", "sections": ["not a dict"]}
    with pytest.raises(KnowledgeError, match="sections must be dicts"):
        from_okf(payload)


def test_from_okf_block_not_dict():
    payload = {
        "source_uri": "x",
        "sections": [{"index": 0, "blocks": ["not a dict"]}],
    }
    with pytest.raises(KnowledgeError, match="blocks must be dicts"):
        from_okf(payload)


def test_from_okf_sections_none():
    bundle = from_okf({"source_uri": "x", "sections": None})
    assert bundle.sections == []


def test_from_okf_blocks_none():
    payload = {"source_uri": "x", "sections": [{"index": 0, "blocks": None}]}
    bundle = from_okf(payload)
    assert bundle.sections[0].blocks == []


def test_from_okf_unknown_block_kind():
    payload = {
        "source_uri": "x",
        "sections": [{"index": 0, "blocks": [{"kind": "unknown_kind"}]}],
    }
    with pytest.raises(KnowledgeError, match="Unknown OKF block kind"):
        from_okf(payload)


def test_loads_invalid_json():
    with pytest.raises(KnowledgeError, match="Invalid OKF JSON"):
        loads("{{broken")


def test_loads_valid_json():
    bundle = loads(json.dumps({"source_uri": "s3://test"}))
    assert bundle.source_uri == "s3://test"


def test_to_okf_schema_version_override():
    bundle = KnowledgeBundle(
        source_uri="file://x", schema_version="0.2", sections=[]
    )
    payload = to_okf(bundle)
    assert payload["$schema"] == "okf/0.2"


def test_to_okf_all_block_kinds():
    bundle = KnowledgeBundle(
        source_uri="file://x",
        sections=[
            DocumentSection(
                index=0,
                heading="All kinds",
                blocks=[
                    DocumentBlock(kind=BlockKind.TEXT, content="text"),
                    DocumentBlock(kind=BlockKind.TABLE, content="|a|b|"),
                    DocumentBlock(kind=BlockKind.IMAGE, content="fig.png"),
                    DocumentBlock(kind=BlockKind.EQUATION, content="E=mc^2"),
                    DocumentBlock(kind=BlockKind.CODE, content="print('hi')"),
                    DocumentBlock(kind=BlockKind.METADATA, content="meta"),
                ],
            )
        ],
    )
    payload = to_okf(bundle)
    kinds = [b["kind"] for b in payload["sections"][0]["blocks"]]
    assert kinds == ["text", "table", "image", "equation", "code", "metadata"]


def test_round_trip_with_section_defaults():
    """Sections/blocks with missing fields survive round-trip."""
    bundle = KnowledgeBundle(source_uri="s3://b")
    payload = to_okf(bundle)
    restored = from_okf(payload)
    assert restored.bundle_id == bundle.bundle_id
    assert restored.schema_version == bundle.schema_version


def test_from_okf_missing_bundle_id():
    bundle = from_okf({"source_uri": "x"})
    assert bundle.bundle_id == ""
    assert bundle.source_uri == "x"


# =======================================================================
# workers.py
# =======================================================================


class TestSynchronousWorker:
    def test_submit_returns_result(self):
        w = SynchronousWorker()
        result = w.submit(lambda a, b: a + b, 1, 2)
        assert result == 3

    def test_submit_with_kwargs(self):
        w = SynchronousWorker()
        result = w.submit(lambda x, y=10: x * y, 5, y=3)
        assert result == 15

    def test_submit_propagates_exception(self):
        w = SynchronousWorker()

        def fail():
            raise ValueError("boom")

        with pytest.raises(ValueError, match="boom"):
            w.submit(fail)


class TestThreadPoolWorker:
    def test_submit_returns_future(self):
        w = ThreadPoolWorker(max_workers=2)
        future = w.submit(lambda: 42)
        assert isinstance(future, Future)
        assert future.result() == 42

    def test_submit_with_args(self):
        w = ThreadPoolWorker(max_workers=2)
        future = w.submit(lambda a, b: a + b, 10, 20)
        assert future.result() == 30

    def test_submit_with_kwargs(self):
        w = ThreadPoolWorker(max_workers=2)
        future = w.submit(lambda x, y=100: x + y, 1, y=2)
        assert future.result() == 3

    def test_error_propagation(self):
        w = ThreadPoolWorker(max_workers=2)

        def fail():
            raise RuntimeError("thread fail")

        future = w.submit(fail)
        with pytest.raises(RuntimeError, match="thread fail"):
            future.result()

    def test_shutdown_cleans_up(self):
        w = ThreadPoolWorker(max_workers=1)
        w.executor.shutdown(wait=True)
        # submitting after shutdown raises
        with pytest.raises(RuntimeError):
            w.submit(lambda: 1).result()


class TestInMemoryTaskQueue:
    def test_init_creates_queue(self):
        q = InMemoryTaskQueue()
        assert q.queue is not None

    def test_enqueue_returns_name(self):
        q = InMemoryTaskQueue()
        result = q.enqueue("my_task", {"key": "value"})
        assert result == "my_task"

    def test_enqueue_stores_item(self):
        q = InMemoryTaskQueue()
        q.enqueue("task_a", {"x": 1})
        name, payload = q.queue.get_nowait()
        assert name == "task_a"
        assert payload == {"x": 1}

    def test_enqueue_multiple(self):
        q = InMemoryTaskQueue()
        q.enqueue("a", {"id": 1})
        q.enqueue("b", {"id": 2})
        assert q.queue.qsize() == 2


# =======================================================================
# response.py
# =======================================================================


def test_build_response_no_structured():
    from raghub.models import Citation

    result = PipelineResult(
        pipeline_id="p1",
        pipeline_name="query",
        outputs={
            "answer": "42",
            "citations": [
                Citation(
                    chunk_id="c1",
                    document_id="d1",
                    page=1,
                    source_uri="doc1",
                )
            ],
            "hits": [],
        },
    )
    from raghub.api.response import build_response

    resp = build_response(result)
    assert resp.answer == "42"
    assert len(resp.citations) == 1
    assert resp.citations[0].chunk_id == "c1"
    assert resp.citations[0].document_id == "d1"
    assert resp.source_chunks == []
    assert resp.structured is None
    assert resp.metadata == {"pipeline_id": "p1", "structured": False}


def test_build_response_with_hits():
    from raghub.models import ChunkRecord
    from raghub.api.response import build_response

    chunk = ChunkRecord(
        chunk_id="c1",
        document_id="d1",
        version=1,
        text="some chunk",
        company="Acme",
        owner="u1",
    )
    hit = MagicMock()
    hit.chunk_id = "c1"
    hit.score = 0.95
    hit.chunk = chunk

    result = PipelineResult(
        pipeline_id="p1",
        pipeline_name="query",
        outputs={
            "answer": "answer",
            "hits": [hit],
        },
    )
    resp = build_response(result)
    assert len(resp.source_chunks) == 1
    assert resp.source_chunks[0].chunk_id == "c1"
    assert resp.source_chunks[0].score == 0.95
    assert resp.source_chunks[0].chunk is chunk
    assert resp.metadata == {"pipeline_id": "p1", "structured": False}


class TestBuildResponseWithStructured:
    def test_structured_model_dump(self):
        from pydantic import BaseModel
        from raghub.api.response import build_response

        class MyModel(BaseModel):
            name: str
            value: int

        structured = MyModel(name="test", value=42)
        result = PipelineResult(
            pipeline_id="p1",
            pipeline_name="query",
            outputs={
                "answer": "fallback",
                "structured": structured,
            },
        )
        resp = build_response(result)
        assert resp.answer == structured.model_dump_json()
        assert resp.structured == {"name": "test", "value": 42}
        assert resp.metadata == {"pipeline_id": "p1", "structured": True}

    def test_structured_model_dump_error_fallback(self):
        from raghub.api.response import build_response

        class BrokenModel:
            """A non-Pydantic object that still has model_dump methods."""

            def model_dump_json(self):
                raise ValueError("broken")

            def model_dump(self):
                raise ValueError("broken")

            def __str__(self):
                return "fallback-str"

        structured = BrokenModel()
        result = PipelineResult(
            pipeline_id="p1",
            pipeline_name="query",
            outputs={
                "answer": "original",
                "structured": structured,
            },
        )
        resp = build_response(result)
        assert resp.answer == "fallback-str"
        assert resp.structured is None
        assert resp.metadata == {"pipeline_id": "p1", "structured": True}


# =======================================================================
# defaults.py
# =======================================================================


class TestDefaultConverter:
    def test_returns_marker_converter_when_importable(self):
        with patch("raghub.converters.marker.MarkerConverter") as mock_mc:
            from raghub.api.defaults import default_converter

            result = default_converter()
            assert result is mock_mc.return_value

    def test_returns_plaintext_when_config_error(self):
        from raghub.exceptions import ConfigurationError

        with patch(
            "raghub.converters.marker.MarkerConverter",
            side_effect=ConfigurationError("not configured"),
        ):
            from raghub.api.defaults import default_converter
            from raghub.converters.plaintext import PlainTextConverter

            result = default_converter()
            assert isinstance(result, PlainTextConverter)


@patch.dict("os.environ", {}, clear=True)
class TestDefaultEmbedder:
    def test_no_api_key_uses_hashing(self):
        from raghub.api.defaults import default_embedder
        from raghub.embeddings.hashing import HashingEmbeddingProvider

        result = default_embedder("test-model", 128)
        assert isinstance(result, HashingEmbeddingProvider)
        assert result.dimension == 128

    def test_with_litellm_api_key(self):
        with patch.dict("os.environ", {"OPENAI_API_KEY": "sk-test"}, clear=True):
            with patch("raghub.embeddings.litellm.LiteLLMEmbeddingProvider") as mock_llm:
                from raghub.api.defaults import default_embedder

                result = default_embedder("gpt-4", 256)
                assert result is mock_llm.return_value

    def test_litellm_config_error_falls_back_to_hashing(self):
        from raghub.exceptions import ConfigurationError

        with patch.dict("os.environ", {"OPENAI_API_KEY": "sk-test"}, clear=True):
            with patch(
                "raghub.embeddings.litellm.LiteLLMEmbeddingProvider",
                side_effect=ConfigurationError("fail"),
            ):
                from raghub.api.defaults import default_embedder
                from raghub.embeddings.hashing import HashingEmbeddingProvider

                result = default_embedder("gpt-4", 256)
                assert isinstance(result, HashingEmbeddingProvider)
                assert result.dimension == 256


@patch.dict("os.environ", {}, clear=True)
class TestDefaultLLM:
    def test_heuristic_model_returns_heuristic(self):
        from raghub.api.defaults import default_llm
        from raghub.llm.heuristic import HeuristicLLMProvider

        result = default_llm("heuristic")
        assert isinstance(result, HeuristicLLMProvider)

    def test_empty_model_returns_heuristic(self):
        from raghub.api.defaults import default_llm
        from raghub.llm.heuristic import HeuristicLLMProvider

        result = default_llm("")
        assert isinstance(result, HeuristicLLMProvider)

    def test_no_api_key_returns_heuristic(self):
        from raghub.api.defaults import default_llm
        from raghub.llm.heuristic import HeuristicLLMProvider

        result = default_llm("gpt-4")
        assert isinstance(result, HeuristicLLMProvider)

    def test_with_api_key_returns_litellm(self):
        with patch.dict("os.environ", {"OPENAI_API_KEY": "sk-test"}, clear=True):
            with patch("raghub.llm.litellm.LiteLLMProvider") as mock_llm:
                from raghub.api.defaults import default_llm

                result = default_llm("gpt-4")
                assert result is mock_llm.return_value

    def test_litellm_config_error_falls_back(self):
        from raghub.exceptions import ConfigurationError

        with patch.dict("os.environ", {"OPENAI_API_KEY": "sk-test"}, clear=True):
            with patch(
                "raghub.llm.litellm.LiteLLMProvider",
                side_effect=ConfigurationError("fail"),
            ):
                from raghub.api.defaults import default_llm
                from raghub.llm.heuristic import HeuristicLLMProvider

                result = default_llm("gpt-4")
                assert isinstance(result, HeuristicLLMProvider)


@patch.dict("os.environ", {}, clear=True)
class TestDefaultVectorStore:
    def test_no_qdrant_url_uses_memory(self):
        from raghub.api.defaults import default_vector_store
        from raghub.vectorstore.memory import InMemoryVectorStore

        result = default_vector_store(384)
        assert isinstance(result, InMemoryVectorStore)

    def test_with_qdrant_url(self):
        with patch.dict("os.environ", {"QDRANT_URL": "http://localhost:6333"}, clear=True):
            with patch("raghub.vectorstore.qdrant.QdrantVectorStore") as mock_qdrant:
                from raghub.api.defaults import default_vector_store

                result = default_vector_store(384)
                mock_qdrant.assert_called_once_with(embedding_dim=384)
                assert result is mock_qdrant.return_value

    def test_qdrant_config_error_falls_back(self):
        from raghub.exceptions import ConfigurationError

        with patch.dict("os.environ", {"QDRANT_URL": "http://localhost:6333"}, clear=True):
            with patch(
                "raghub.vectorstore.qdrant.QdrantVectorStore",
                side_effect=ConfigurationError("fail"),
            ):
                from raghub.api.defaults import default_vector_store
                from raghub.vectorstore.memory import InMemoryVectorStore

                result = default_vector_store(384)
                assert isinstance(result, InMemoryVectorStore)


@patch.dict("os.environ", {}, clear=True)
class TestDefaultStructured:
    def test_no_api_key_returns_none(self):
        from raghub.api.defaults import default_structured

        result = default_structured()
        assert result is None

    def test_with_api_key_returns_instructor(self):
        with patch.dict("os.environ", {"OPENAI_API_KEY": "sk-test"}, clear=True):
            with patch(
                "raghub.structured.instructor.InstructorStructuredOutputProvider"
            ) as mock_inst:
                from raghub.api.defaults import default_structured

                result = default_structured()
                assert result is mock_inst.return_value

    def test_import_error_returns_none(self):
        with patch.dict("os.environ", {"OPENAI_API_KEY": "sk-test"}, clear=True):
            with patch(
                "raghub.structured.instructor.InstructorStructuredOutputProvider",
                side_effect=ImportError("not installed"),
            ):
                from raghub.api.defaults import default_structured

                result = default_structured()
                assert result is None

    def test_config_error_returns_none(self):
        from raghub.exceptions import ConfigurationError

        with patch.dict("os.environ", {"OPENAI_API_KEY": "sk-test"}, clear=True):
            with patch(
                "raghub.structured.instructor.InstructorStructuredOutputProvider",
                side_effect=ConfigurationError("fail"),
            ):
                from raghub.api.defaults import default_structured

                result = default_structured()
                assert result is None


@patch.dict("os.environ", {}, clear=True)
class TestDefaultTelemetry:
    def test_import_error_returns_noop(self):

        with patch.dict("sys.modules", {"raghub.telemetry.langfuse": None}, clear=False):
            # Force import to fail so the except ImportError branch is taken
            from raghub.api.defaults import default_telemetry
            from raghub.observability.noop import NoOpTelemetry

            result = default_telemetry()
            assert isinstance(result, NoOpTelemetry)

    def test_not_configured_returns_noop(self):
        mock_langfuse = MagicMock()
        mock_langfuse.is_configured.return_value = False

        with patch(
            "raghub.telemetry.langfuse.LangfuseTelemetryProvider",
            mock_langfuse,
        ):
            from raghub.api.defaults import default_telemetry
            from raghub.observability.noop import NoOpTelemetry

            result = default_telemetry()
            assert isinstance(result, NoOpTelemetry)

    def test_configured_returns_langfuse(self):
        mock_langfuse = MagicMock()
        mock_langfuse.is_configured.return_value = True

        with patch(
            "raghub.telemetry.langfuse.LangfuseTelemetryProvider",
            mock_langfuse,
        ):
            from raghub.api.defaults import default_telemetry

            result = default_telemetry()
            assert result is mock_langfuse.return_value

    def test_import_succeeds_but_not_configured(self):
        """When langfuse imports ok but is_configured() returns False."""
        mock_provider = MagicMock()
        mock_provider.is_configured.return_value = False

        with patch(
            "raghub.telemetry.langfuse.LangfuseTelemetryProvider",
            mock_provider,
        ):
            from raghub.api.defaults import default_telemetry
            from raghub.observability.noop import NoOpTelemetry

            result = default_telemetry()
            assert isinstance(result, NoOpTelemetry)


# =======================================================================
# workers.py  —  targeted coverage for explicitly stated lines
# =======================================================================


class TestSynchronousWorkerCoverage:
    """Exercises SynchronousWorker.submit (line 47)."""

    def test_submit_positional_args(self):
        w = SynchronousWorker()
        assert w.submit(lambda a, b, c: a + b + c, 1, 2, 3) == 6

    def test_submit_no_args(self):
        w = SynchronousWorker()
        assert w.submit(lambda: 99) == 99


class TestThreadPoolWorkerCoverage:
    """Exercises ThreadPoolWorker.__init__ (line 63) and .submit (line 76)."""

    def test_init_default_max_workers(self):
        w = ThreadPoolWorker()
        assert w.executor._max_workers == 4

    def test_init_custom_max_workers(self):
        w = ThreadPoolWorker(max_workers=1)
        assert w.executor._max_workers == 1

    def test_submit_with_args(self):
        w = ThreadPoolWorker(max_workers=2)
        future = w.submit(lambda x, y: x * y, 6, 7)
        assert future.result() == 42


class TestInMemoryTaskQueueCoverage:
    """Exercises InMemoryTaskQueue.__init__ (line 92) and .enqueue (lines 104-105)."""

    def test_init_sets_queue(self):
        q = InMemoryTaskQueue()
        assert q.queue is not None

    def test_enqueue_puts_and_returns_name(self):
        q = InMemoryTaskQueue()
        name = q.enqueue("task_name", {"answer": 42})
        assert name == "task_name"
        retrieved_name, payload = q.queue.get_nowait()
        assert retrieved_name == "task_name"
        assert payload == {"answer": 42}
