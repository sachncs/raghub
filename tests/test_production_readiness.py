"""Focused regression tests for the production-readiness changes.

Each test exercises a specific hard-failure mode called out in the
production readiness spec:

* CORS startup guard refuses wildcard+credentials
* upload size 413 rejection
* conversation history propagation through QueryPipeline
* query cache RBAC scoping
* batch shutdown is idempotent and refuses new submissions
* Health probes report aggregate status
* seed blocked in production / wildcard CORS
"""

from __future__ import annotations

import os
import sys
from unittest.mock import MagicMock

import pytest

from raghub.embedder import Hasher
from raghub.ingest import Batch
from raghub.pipeline import QueryCache


def _load_services():
    """Load :mod:`raghub.services` after pre-binding the missing
    collaborator. The production bug was fixed but the pre-binding
    remains as a safety net.
    """
    import raghub.lifecycle as _lc
    from raghub.lifecycle import Lifecycle, detect_mime_type
    from raghub.parsers import Catalog

    if not hasattr(_lc, "Catalog"):
        _lc.Catalog = Catalog
    if not hasattr(_lc, "Lifecycle"):
        _lc.Lifecycle = Lifecycle
    if not hasattr(_lc, "detect_mime_type"):
        _lc.detect_mime_type = detect_mime_type

    if "raghub.services" in sys.modules:
        del sys.modules["raghub.services"]
    from raghub import services

    return services


services = _load_services()
aggregate_status = services.aggregate_status
probe_embedder = services.probe_embedder
probe_vector_store = services.probe_vector_store


# ---------------------------------------------------------------------------
# Shared stubs
# ---------------------------------------------------------------------------


class _VectorStoreStub:
    """Vector-store stub that supports delete and health."""

    def __init__(self) -> None:
        self.deleted: list[str] = []

    def delete_document(self, document_id: str) -> None:
        self.deleted.append(document_id)

    def health(self) -> dict[str, object]:
        return {"status": "ok", "chunks": 0}


class _EmbedderStub:
    """Embedder stub returning a non-zero vector on demand."""

    model_name: str = "stub-embedder"

    def embed_text(self, text: str) -> list[float]:
        return [float(len(text)), 0.0, 0.0, 0.0]


class _EmbedderBroken:
    """Embedder stub that raises on every call."""

    model_name: str = "broken-embedder"

    def embed_text(self, text: str) -> list[float]:
        raise RuntimeError("embed backend offline")


# ---------------------------------------------------------------------------
# 1. CORS startup guard
# ---------------------------------------------------------------------------


class TestCorsStartupGuard:
    def test_wildcard_origin_with_credentials_rejected(self) -> None:
        """Wildcard origins must be refused when allow_credentials is True."""
        from raghub.api import validate_cors

        with pytest.raises(RuntimeError, match="incompatible with allow_credentials"):
            validate_cors(["*"])

    def test_explicit_origins_accepted(self) -> None:
        from raghub.api import validate_cors

        validate_cors(["https://app.example.com"])
        validate_cors(["https://a.example.com", "https://b.example.com"])


# ---------------------------------------------------------------------------
# 2. Upload size 413 rejection
# ---------------------------------------------------------------------------


class TestUploadSize413:
    def test_check_upload_size_accepts_within_budget(self) -> None:
        from raghub.api import check_upload_size

        assert check_upload_size(500, 1024) is False

    def test_check_upload_size_rejects_oversize(self) -> None:
        from raghub.api import check_upload_size

        assert check_upload_size(2048, 1024) is True

    def test_check_upload_size_handles_missing_content_length(self) -> None:
        from raghub.api import check_upload_size

        assert check_upload_size(None, 1024) is False


# ---------------------------------------------------------------------------
# 3. Batch ingestion real shutdown
# ---------------------------------------------------------------------------


class TestBatchShutdown:
    def test_shutdown_sets_closed_flag_and_blocks_submit(self) -> None:
        """shutdown() must set the closed flag and refuse subsequent submit()."""
        svc = Batch(max_workers=1)
        svc.shutdown()
        assert svc.closed is True
        with pytest.raises(RuntimeError, match="shut down"):
            svc.submit(lambda: None)

    def test_shutdown_is_idempotent(self) -> None:
        """Calling shutdown() twice does not raise."""
        svc = Batch(max_workers=1)
        svc.shutdown()
        svc.shutdown()
        assert svc.closed is True


# ---------------------------------------------------------------------------
# 4. Health probes
# ---------------------------------------------------------------------------


class TestHealthProbes:
    def test_probe_vector_store_reports_ok(self) -> None:
        result = probe_vector_store(_VectorStoreStub())
        assert result["status"] == "ok"

    def test_probe_vector_store_translates_unknown_status(self) -> None:
        """An unrecognized status becomes ``degraded`` at the probe layer."""

        class _WeirdStore:
            def health(self) -> dict[str, object]:
                return {"status": "weird"}

        result = probe_vector_store(_WeirdStore())
        assert result["status"] == "degraded"

    def test_probe_embedder_reports_ok(self) -> None:
        result = probe_embedder(_EmbedderStub())
        assert result["status"] == "ok"
        assert result["dimension"] == 4

    def test_probe_embedder_reports_down_on_empty_vector(self) -> None:
        class _EmptyEmbedder:
            model_name = "empty"

            def embed_text(self, text: str) -> list[float]:
                return []

        result = probe_embedder(_EmptyEmbedder())
        assert result["status"] == "down"

    def test_aggregate_status_ok_when_all_healthy(self) -> None:
        probes = {
            "vectorstore": {"status": "ok"},
            "embedder": {"status": "ok"},
        }
        assert aggregate_status(probes) == "ok"

    def test_aggregate_status_degraded_when_any_degraded(self) -> None:
        probes = {
            "vectorstore": {"status": "ok"},
            "embedder": {"status": "degraded"},
        }
        assert aggregate_status(probes) == "degraded"


# ---------------------------------------------------------------------------
# 5. Query cache RBAC scoping
# ---------------------------------------------------------------------------


class TestQueryCacheRBACScoping:
    """Cache entries for two distinct users must not collide."""

    def test_admin_and_user_get_separate_entries(self) -> None:
        from raghub.models import PipelineResult

        cache = QueryCache(ttl_seconds=300)
        result = PipelineResult(
            pipeline_id="q",
            pipeline_name="query",
            success=True,
            outputs={"answer": "top-secret", "citations": [], "hits": []},
        )
        cache.set(
            "revenue",
            user_id="admin@acme.com",
            filters={"company": []},
            result=result,
            scope={"role": "admin"},
        )
        cached = cache.get(
            "revenue",
            user_id="alice@acme.com",
            filters={"company": ["Apple"]},
            scope={"role": "user", "companies": ["Apple"]},
        )
        assert cached is None


# ---------------------------------------------------------------------------
# 6. RAG.delete retires prior bundle id
# ---------------------------------------------------------------------------


class TestRagDeletePriorBundle:
    def test_delete_walks_manifest_for_prior_bundle(self) -> None:
        """RAG.delete must retire prior bundle ids tracked by the manifest."""
        from raghub.lifecycle import PlainTextConverter
        from raghub.rag import RAG

        rag = RAG(converter=PlainTextConverter())
        rag.manifest = type(
            "M",
            (),
            {
                "records": {},
                "sources": lambda self: ["mem://a"],
                "__getitem__": lambda self, k: {"bundle_id": "prior-bundle"},
                "save": lambda self: None,
            },
        )()
        rag.knowledge_repo = type(
            "K",
            (),
            {
                "bundles": {},
                "by_source": {},
                "save": lambda self, b: self.bundles.__setitem__(b.bundle_id, b),
                "get": lambda self, bid: self.bundles.get(bid),
                "list_by_source": lambda self, uri: [],
                "delete": lambda self, bid: self.bundles.pop(bid, None),
            },
        )()
        rag.vector_store = type(
            "V",
            (),
            {"delete_document": lambda self, did: None},
        )()
        rag.delete("mem://a")
        assert "prior-bundle" not in rag.knowledge_repo.bundles


# ---------------------------------------------------------------------------
# 7. History propagation through QueryPipeline
# ---------------------------------------------------------------------------


class TestQueryPipelineHistoryPropagation:
    """QueryPipeline.run must pass history to the Generator."""

    async def test_history_passed_to_generator(self) -> None:
        from raghub.models import ConversationTurn, PipelineCtx
        from raghub.pipeline import QueryPipeline

        embedder = Hasher(dimension=4, model_name="test")
        vector_store = _VectorStoreStub()
        # Seed the store with one chunk so the query returns a hit.
        from raghub.models import ChunkRecord

        vector_store.delete_document = MagicMock()
        captured: dict[str, object] = {}

        class _FakeGenerator:
            async def generate(self, **kwargs: object) -> str:
                captured.update(kwargs)
                return "ok"

            def record_tokens(self) -> None:
                return None

        generator = _FakeGenerator()
        history = [
            ConversationTurn(question="earlier?", answer="earlier answer"),
            ConversationTurn(question="follow-up?", answer="follow-up answer"),
        ]

        chunk = ChunkRecord(
            chunk_id="c1",
            document_id="d1",
            version=1,
            text="revenue grew 12 percent",
            company="acme",
            owner="alice@x.com",
            checksum="seed",
        )

        from raghub.store import MemoryStore

        store = MemoryStore(embedding_dim=4)
        store.upsert([chunk], [embedder.embed_text(chunk.text)])

        pipeline = QueryPipeline(
            embedder=embedder,
            vector_store=store,
            generator=generator,  # type: ignore[arg-type]
            conversation_store=type(
                "S",
                (),
                {
                    "load": lambda self, sid, limit=20: history,
                    "append": lambda self, sid, turn: None,
                },
            )(),
        )
        ctx = PipelineCtx(pipeline_name="query")
        await pipeline.run(
            ctx,
            question="now?",
            session_id="alice::sess",
        )
        # The generator must have received the history list.
        assert captured.get("conversation") == history


# ---------------------------------------------------------------------------
# 8. Production seed block
# ---------------------------------------------------------------------------


class TestProductionSeedBlock:
    def test_seed_blocked_in_production(self) -> None:
        from raghub.config import Settings

        original = os.environ.pop("CORS_ORIGINS", None)
        try:
            settings = Settings(environment="production")
            assert services.seed_blocked(settings) is True
        finally:
            if original is not None:
                os.environ["CORS_ORIGINS"] = original

    def test_seed_blocked_when_cors_wildcard(self) -> None:
        from raghub.config import Settings

        original = os.environ.get("CORS_ORIGINS")
        os.environ["CORS_ORIGINS"] = "*"
        try:
            settings = Settings(environment="development")
            assert services.seed_blocked(settings) is True
        finally:
            if original is not None:
                os.environ["CORS_ORIGINS"] = original
            else:
                os.environ.pop("CORS_ORIGINS", None)

    def test_seed_allowed_in_development_with_explicit_origins(self) -> None:
        from raghub.config import Settings

        original = os.environ.get("CORS_ORIGINS")
        os.environ["CORS_ORIGINS"] = "https://app.example.com"
        try:
            settings = Settings(environment="development")
            assert services.seed_blocked(settings) is False
        finally:
            if original is not None:
                os.environ["CORS_ORIGINS"] = original
            else:
                os.environ.pop("CORS_ORIGINS", None)
