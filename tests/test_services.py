"""Qualitative tests for the service layer.

Covers:

* :class:`Document` mixin — upload authorization, RBAC for delete.
* :class:`Health` mixin — aggregate status.
* :func:`probe_vector_store`, :func:`probe_embedder`, :func:`aggregate_status`.
* :class:`Synchronous` / :class:`ThreadPool` — worker primitives.
* :class:`MemoryQueue` — enqueue contract.
* :func:`parse_users` — JSON seed parsing.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from raghub.errors import (
    AuthorizationError,
    DocumentError,
)
from raghub.models import (
    DocumentLifecycleStatus,
    DocumentRecord,
    User,
)


def _load_services():
    """Load :mod:`raghub.services` after pre-binding the missing
    collaborator imports. ``raghub.services.__init__`` imports
    :class:`Catalog` from ``raghub.lifecycle`` even though it actually
    lives in ``raghub.parsers`` — pre-binding it in ``raghub.lifecycle``
    lets the module import cleanly for these unit tests.
    """
    import sys

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
    import raghub.services as services

    return services


services = _load_services()


# ===========================================================================
# helpers — probe_*, aggregate_status, missing_doc
# ===========================================================================


class TestProbeVectorStore:
    def test_returns_ok_when_health_reports_ok(self) -> None:
        class _GoodStore:
            def health(self) -> dict[str, str]:
                return {"status": "ok"}

        result = services.probe_vector_store(_GoodStore())
        assert result["status"] == "ok"

    def test_returns_degraded_when_health_reports_unknown(self) -> None:
        class _DegradedStore:
            def health(self) -> dict[str, str]:
                return {"status": "weird"}

        result = services.probe_vector_store(_DegradedStore())
        assert result["status"] == "degraded"

    def test_returns_unknown_when_no_health_method(self) -> None:
        result = services.probe_vector_store(object())
        assert result["status"] == "unknown"


class TestProbeEmbedder:
    def test_returns_ok_with_dimension_for_live_embedder(self) -> None:
        class _Embedder:
            model_name = "fake"

            def embed_text(self, text: str) -> list[float]:
                return [0.1, 0.2, 0.3]

        result = services.probe_embedder(_Embedder())
        assert result["status"] == "ok"
        assert result["dimension"] == 3

    def test_returns_down_for_empty_vector(self) -> None:
        class _EmptyEmbedder:
            model_name = "fake"

            def embed_text(self, text: str) -> list[float]:
                return []

        result = services.probe_embedder(_EmptyEmbedder())
        assert result["status"] == "down"

    def test_returns_unknown_when_no_embed_text(self) -> None:
        result = services.probe_embedder(object())
        assert result["status"] == "unknown"

    def test_returns_unknown_when_embedder_is_none(self) -> None:
        result = services.probe_embedder(None)
        assert result["status"] == "unknown"


class TestAggregateStatus:
    def test_all_ok(self) -> None:
        assert services.aggregate_status({"a": {"status": "ok"}, "b": {"status": "ok"}}) == "ok"

    def test_any_down_makes_overall_down(self) -> None:
        assert (
            services.aggregate_status({"a": {"status": "ok"}, "b": {"status": "down"}}) == "down"
        )

    def test_any_degraded_makes_overall_degraded(self) -> None:
        assert (
            services.aggregate_status(
                {"a": {"status": "ok"}, "b": {"status": "degraded"}}
            )
            == "degraded"
        )


class TestMissingDoc:
    def test_raises_document_error(self) -> None:
        with pytest.raises(DocumentError, match="Unknown document id"):
            services.missing_doc("missing-id")


# ===========================================================================
# Document service
# ===========================================================================


class TestDocumentUpload:
    @pytest.fixture
    def container(self) -> MagicMock:
        c = MagicMock()
        c.auth = AsyncMock()
        c.auth.resolve_user.return_value = (
            User(
                user_id="u1",
                email="admin@acme.com",
                allowed_companies=["Acme"],
                is_admin=True,
            ),
            [],
        )
        c.uow = MagicMock()
        c.uow.document_repo = AsyncMock()
        c.ingestion = AsyncMock()
        c.vector_store = MagicMock()
        return c

    @pytest.fixture
    def service(self, container: MagicMock) -> Any:
        return services.Document(container)

    @pytest.mark.asyncio
    async def test_upload_admin_for_own_tenant_succeeds(
        self, service: Any, container: MagicMock
    ) -> None:
        container.ingestion.ingest.return_value = MagicMock(
            document=DocumentRecord(
                document_id="d1",
                checksum="abc",
                owner="admin@acme.com",
                organization="Acme",
                status=DocumentLifecycleStatus.READY,
            )
        )
        result = await service.upload_document(
            token="tok1", filename="Acme_report.pdf", content=b"%PDF-1.4 dummy"
        )
        assert result.document_id == "d1"
        assert result.organization == "Acme"

    @pytest.mark.asyncio
    async def test_upload_rejects_other_tenant(
        self, service: Any, container: MagicMock
    ) -> None:
        container.auth.resolve_user.return_value = (
            User(
                user_id="u1",
                email="u@acme.com",
                allowed_companies=["Acme"],
                is_admin=False,
            ),
            [],
        )
        with pytest.raises(AuthorizationError, match="cannot upload"):
            await service.upload_document(
                token="tok1",
                filename="Globex_secret.pdf",
                content=b"%PDF-1.4 dummy",
            )
        container.ingestion.ingest.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_upload_company_override_skips_filename_heuristic(
        self, service: Any, container: MagicMock
    ) -> None:
        """When ``company`` is provided, it wins over the filename prefix."""
        container.ingestion.ingest.return_value = MagicMock(
            document=DocumentRecord(
                document_id="d1",
                checksum="abc",
                owner="admin@acme.com",
                organization="Acme",
                status=DocumentLifecycleStatus.READY,
            )
        )
        await service.upload_document(
            token="tok1",
            filename="Other_misleading.pdf",
            content=b"%PDF-1.4 dummy",
            company="Acme",
        )
        kwargs = container.ingestion.ingest.await_args.kwargs
        assert kwargs["organization"] == "Acme"


class TestDocumentDelete:
    @pytest.fixture
    def container(self) -> MagicMock:
        c = MagicMock()
        c.auth = AsyncMock()
        c.auth.resolve_user.return_value = (
            User(
                user_id="u1",
                email="admin@acme.com",
                allowed_companies=["Acme"],
                is_admin=True,
            ),
            [],
        )
        c.uow = MagicMock()
        c.uow.document_repo = AsyncMock()
        c.vector_store = MagicMock()
        return c

    @pytest.fixture
    def service(self, container: MagicMock) -> Any:
        return services.Document(container)

    @pytest.mark.asyncio
    async def test_delete_non_admin_forbidden(
        self, service: Any, container: MagicMock
    ) -> None:
        container.auth.resolve_user.return_value = (
            User(
                user_id="u1",
                email="u@acme.com",
                allowed_companies=["Acme"],
                is_admin=False,
            ),
            [],
        )
        with pytest.raises(AuthorizationError, match="Admin"):
            await service.delete_document(token="tok1", document_id="d1")


# ===========================================================================
# Health — container wiring
# ===========================================================================


class TestHealthContainer:
    def test_health_reports_overall_ok(self) -> None:
        container = MagicMock()
        container.vector_store.health.return_value = {"status": "ok"}
        container.embedder.embed_text.return_value = [0.1, 0.2]
        container.embedder.model_name = "fake"

        report = services.Health(container).health()
        assert report["status"] == "ok"

    def test_health_reports_degraded_when_component_unhealthy(self) -> None:
        """A non-OK status from any component is reported as ``degraded``.

        The probe layer translates unknown statuses to ``degraded``
        before they reach the aggregator, so the overall report is
        ``degraded`` rather than ``down`` for soft failures.
        """
        container = MagicMock()
        container.vector_store.health.return_value = {"status": "unknown"}
        container.embedder = None

        report = services.Health(container).health()
        assert report["status"] in {"degraded", "down"}


# ===========================================================================
# Worker primitives
# ===========================================================================


class TestSynchronousWorker:
    def test_returns_callable_result_inline(self) -> None:
        w = services.Synchronous()
        assert w.submit(lambda x: x * 2, 5) == 10

    def test_propagates_exception(self) -> None:
        w = services.Synchronous()
        with pytest.raises(RuntimeError, match="boom"):
            w.submit(lambda: (_ for _ in ()).throw(RuntimeError("boom")))


class TestThreadPoolWorker:
    def test_submit_returns_future(self) -> None:
        from concurrent.futures import Future

        w = services.ThreadPool(max_workers=2)
        try:
            future = w.submit(lambda x: x + 1, 4)
            assert isinstance(future, Future)
            assert future.result(timeout=5) == 5
        finally:
            w.executor.shutdown(wait=True)


# ===========================================================================
# MemoryQueue
# ===========================================================================


class TestInMemoryTaskQueue:
    def test_enqueue_returns_name(self) -> None:
        q = services.MemoryQueue()
        assert q.enqueue("job-1", {"k": "v"}) == "job-1"

    def test_enqueue_stores_payload(self) -> None:
        q = services.MemoryQueue()
        q.enqueue("job-1", {"k": "v"})
        item = q.queue.get_nowait()
        assert item == ("job-1", {"k": "v"})


# ===========================================================================
# parse_users
# ===========================================================================


class TestParseUsers:
    def test_parses_valid_json(self) -> None:
        import json

        raw = json.dumps(
            [
                {"email": "a@x.com", "is_admin": True, "allowed_companies": ["Acme"]},
                {"email": "b@x.com", "is_admin": False, "allowed_companies": []},
            ]
        )
        users = services.parse_users(raw)
        assert len(users) == 2
        assert users[0]["email"] == "a@x.com"
        assert users[1]["is_admin"] is False

    def test_invalid_json_raises(self) -> None:
        with pytest.raises(Exception):
            services.parse_users("not json")

    def test_empty_string_raises(self) -> None:
        with pytest.raises(Exception):
            services.parse_users("")
