"""Tests for ``raghub.services.documents`` (Documents service)."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from raghub.errors import AuthorizationError, IngestionError
from raghub.ingest import IngestionResult
from raghub.models import Classification, Document, DocumentLifecycleStatus
from raghub.services.documents import Documents, get_doc, list_records


def make_document(
    *,
    document_id: str = "doc-1",
    organization: str = "acme",
    owner: str = "alice@example.com",
) -> Document:
    """Build a minimal :class:`Document` for tests."""

    return Document(
        id=document_id,
        organization=organization,
        owner=owner,
        title="Quarterly Report",
        source_uri="mem://text",
        mime_type="text/plain",
        size_bytes=1024,
        checksum="deadbeef",
        classification=Classification.Internal,
        company="acme",
        department="finance",
        tenant_id=None,
        status=DocumentLifecycleStatus.Ready,
    )


def test_list_records_returns_documents_from_repository() -> None:
    """``list_records`` delegates to ``uow.document_repo.list_all()``."""

    docs = [make_document(document_id="a"), make_document(document_id="b")]

    class StubRepo:
        async def list_all(self) -> list[Document]:
            return docs

    uow = SimpleNamespace(document_repo=StubRepo())
    import asyncio

    records = asyncio.run(list_records(uow))
    assert records == docs


def test_get_doc_returns_document_when_found() -> None:
    """``get_doc`` returns the document when the repo finds it."""

    doc = make_document()

    class StubRepo:
        async def get(self, document_id: str) -> Document:
            return doc

    uow = SimpleNamespace(document_repo=StubRepo())
    import asyncio

    assert asyncio.run(get_doc(uow, "doc-1")) is doc


def test_get_doc_raises_ingestion_error_when_missing() -> None:
    """``get_doc`` raises :class:`IngestionError` when the doc is missing."""

    class StubRepo:
        async def get(self, document_id: str) -> None:
            return None

    uow = SimpleNamespace(document_repo=StubRepo())
    import asyncio

    with pytest.raises(IngestionError, match="Unknown document id: missing"):
        asyncio.run(get_doc(uow, "missing"))


def test_documents_log_delegates_to_emit_log() -> None:
    """``Documents.log`` forwards to ``emit_log`` with the given payload."""

    captured: list[tuple[str, dict[str, object]]] = []

    class TestLogger:
        def info(self, message: str, **kwargs: object) -> None:
            captured.append((message, kwargs))

    container = SimpleNamespace(logger=TestLogger())
    Documents(container).log("test.event", key="value")
    assert captured == [("test.event", {"extra": {"key": "value"}})]


@pytest.mark.asyncio
async def test_upload_document_rejects_user_without_company_access() -> None:
    """``Documents.upload_document`` raises AuthorizationError for wrong company."""

    user = SimpleNamespace(
        is_admin=False,
        allowed_companies=["other"],
        allowed_groups=[],
        email="alice@example.com",
    )

    class StubAuth:
        async def resolve_user(self, token: str) -> tuple[Any, list[Any]]:
            return user, []

    container = SimpleNamespace(auth=StubAuth(), ingestion=None)
    docs = Documents(container)
    with pytest.raises(AuthorizationError, match="cannot upload"):
        await docs.upload_document(token="t", filename="acme_report.txt", content=b"data")


@pytest.mark.asyncio
async def test_upload_document_uses_filename_company_when_unspecified() -> None:
    """When ``company`` is not provided, ``upload_document`` derives it from filename."""

    user = SimpleNamespace(
        is_admin=True,
        allowed_companies=[],
        allowed_groups=[],
        email="alice@example.com",
    )

    class StubAuth:
        async def resolve_user(self, token: str) -> tuple[Any, list[Any]]:
            return user, []

    captured_company: list[str] = []

    class StubIngestion:
        async def ingest(self, **kwargs: Any) -> IngestionResult:
            captured_company.append(kwargs["organization"])
            return IngestionResult(
                document=make_document(),
                chunks=[],
            )

    container = SimpleNamespace(auth=StubAuth(), ingestion=StubIngestion())
    docs = Documents(container)
    result = await docs.upload_document(token="t", filename="acme_report.txt", content=b"data")
    assert captured_company == ["acme"]
    assert result.id == "doc-1"


@pytest.mark.asyncio
async def test_list_documents_returns_all_for_admin() -> None:
    """Admins see every document via ``list_records``."""

    user = SimpleNamespace(is_admin=True, allowed_companies=[], allowed_groups=[])

    class StubAuth:
        async def resolve_user(self, token: str) -> tuple[Any, list[Any]]:
            return user, []

    docs = [make_document(document_id="a"), make_document(document_id="b")]

    class StubRepo:
        async def list_all(self) -> list[Document]:
            return docs

    container = SimpleNamespace(auth=StubAuth(), uow=SimpleNamespace(document_repo=StubRepo()))
    docs_svc = Documents(container)
    assert await docs_svc.list_documents("t") == docs


@pytest.mark.asyncio
async def test_list_documents_filters_by_company_for_non_admin() -> None:
    """Non-admin users see only documents for their allowed companies."""

    user = SimpleNamespace(
        is_admin=False,
        allowed_companies=["acme"],
        allowed_groups=[],
    )

    class StubAuth:
        async def resolve_user(self, token: str) -> tuple[Any, list[Any]]:
            return user, []

    acme_doc = make_document(document_id="a", organization="acme")
    other_doc = make_document(document_id="b", organization="globex")

    class StubRepo:
        async def list_by_organization(self, org: str) -> list[Document]:
            if org == "acme":
                return [acme_doc]
            return [other_doc]

    container = SimpleNamespace(auth=StubAuth(), uow=SimpleNamespace(document_repo=StubRepo()))
    docs_svc = Documents(container)
    assert await docs_svc.list_documents("t") == [acme_doc]


@pytest.mark.asyncio
async def test_document_status_rejects_unauthorized_company() -> None:
    """``document_status`` raises AuthorizationError when user cannot access."""

    user = SimpleNamespace(
        is_admin=False,
        allowed_companies=["other"],
        allowed_groups=[],
    )

    class StubAuth:
        async def resolve_user(self, token: str) -> tuple[Any, list[Any]]:
            return user, []

    doc = make_document(organization="acme")

    class StubRepo:
        async def get(self, document_id: str) -> Document:
            return doc

    container = SimpleNamespace(auth=StubAuth(), uow=SimpleNamespace(document_repo=StubRepo()))
    docs = Documents(container)
    with pytest.raises(AuthorizationError):
        await docs.document_status("t", "doc-1")


@pytest.mark.asyncio
async def test_document_status_returns_doc_when_authorized() -> None:
    """``document_status`` returns the doc for an authorised user."""

    user = SimpleNamespace(
        is_admin=False,
        allowed_companies=["acme"],
        allowed_groups=[],
    )

    class StubAuth:
        async def resolve_user(self, token: str) -> tuple[Any, list[Any]]:
            return user, []

    doc = make_document(organization="acme")

    class StubRepo:
        async def get(self, document_id: str) -> Document:
            return doc

    container = SimpleNamespace(auth=StubAuth(), uow=SimpleNamespace(document_repo=StubRepo()))
    docs = Documents(container)
    assert await docs.document_status("t", "doc-1") is doc


@pytest.mark.asyncio
async def test_delete_document_rejects_non_admin() -> None:
    """``delete_document`` raises AuthorizationError for non-admin users."""

    user = SimpleNamespace(is_admin=False, allowed_companies=[], allowed_groups=[])

    class StubAuth:
        async def resolve_user(self, token: str) -> tuple[Any, list[Any]]:
            return user, []

    container = SimpleNamespace(auth=StubAuth(), rag_facade=None)
    docs = Documents(container)
    with pytest.raises(AuthorizationError, match="Admin only"):
        await docs.delete_document("t", "doc-1")


@pytest.mark.asyncio
async def test_delete_document_delegates_to_rag_facade_when_present() -> None:
    """``delete_document`` routes to ``rag_facade.delete`` when available."""

    user = SimpleNamespace(is_admin=True, allowed_companies=[], allowed_groups=[])

    class StubAuth:
        async def resolve_user(self, token: str) -> tuple[Any, list[Any]]:
            return user, []

    deleted: list[str] = []

    class Rag:
        def delete(self, document_id: str) -> None:
            deleted.append(document_id)

    container = SimpleNamespace(auth=StubAuth(), rag_facade=Rag())
    docs = Documents(container)
    await docs.delete_document("t", "doc-1")
    assert deleted == ["doc-1"]
