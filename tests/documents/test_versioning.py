"""Tests for ``raghub.documents.versioning``."""
from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from raghub.documents.versioning import new_version
from raghub.models import (
    ChunkRecord,
    DocumentLifecycleStatus,
    DocumentRecord,
    DocumentVersion,
)


def _make_document(**overrides: object) -> DocumentRecord:
    base: dict[str, object] = dict(
        checksum="abc",
        owner="alice@acme.com",
        organization="acme",
        filename="report.pdf",
        mime_type="application/pdf",
        file_type="pdf",
        version=1,
        status=DocumentLifecycleStatus.READY,
    )
    base.update(overrides)
    return DocumentRecord(**base)  # type: ignore[arg-type]


def test_new_version_from_none_is_version_one() -> None:
    """Calling ``new_version(None)`` starts at version ``1``."""
    record = new_version(
        None, organization="acme", filename="report.pdf", owner="me", checksum="h"
    )
    assert record.version == 1
    assert record.status == DocumentLifecycleStatus.NEW
    assert record.organization == "acme"
    assert record.filename == "report.pdf"


def test_new_version_increments_previous() -> None:
    """A non-``None`` previous record yields ``previous.version + 1``."""
    previous = _make_document(version=3)
    record = new_version(previous)
    assert record.version == 4


def test_new_version_resets_status_to_new() -> None:
    """Every new version begins in the ``NEW`` lifecycle state."""
    previous = _make_document(status=DocumentLifecycleStatus.READY)
    record = new_version(previous)
    assert record.status == DocumentLifecycleStatus.NEW


def test_new_version_resets_status_even_when_overridden() -> None:
    """The status reset wins over a caller-supplied ``status`` override."""
    previous = _make_document(status=DocumentLifecycleStatus.READY)
    record = new_version(previous, status=DocumentLifecycleStatus.ARCHIVED)
    assert record.status == DocumentLifecycleStatus.NEW


def test_new_version_sets_updated_at_to_now() -> None:
    """``updated_at`` is always the current UTC time."""
    previous = _make_document(updated_at=datetime(2000, 1, 1, tzinfo=UTC))
    before = datetime.now(UTC)
    record = new_version(previous)
    after = datetime.now(UTC)
    assert before <= record.updated_at <= after


def test_new_version_preserves_document_id() -> None:
    """``document_id`` carries over from the previous version."""
    previous = _make_document()
    record = new_version(previous)
    assert record.document_id == previous.document_id


def test_new_version_allows_document_id_override() -> None:
    """A caller override for ``document_id`` wins over the previous value."""
    previous = _make_document()
    record = new_version(previous, document_id="new-id")
    assert record.document_id == "new-id"


def test_new_version_preserves_created_at() -> None:
    """``created_at`` carries over from the previous version."""
    original = datetime(2024, 1, 1, tzinfo=UTC)
    previous = _make_document(created_at=original)
    record = new_version(previous)
    assert record.created_at == original


def test_new_version_allows_created_at_override() -> None:
    """A caller-supplied ``created_at`` wins over the previous value."""
    original = datetime(2024, 1, 1, tzinfo=UTC)
    new = datetime(2025, 1, 1, tzinfo=UTC)
    previous = _make_document(created_at=original)
    record = new_version(previous, created_at=new)
    assert record.created_at == new


def test_new_version_clones_other_fields() -> None:
    """Non-overridden fields are cloned from the previous record."""
    previous = _make_document(
        organization="acme",
        filename="doc.pdf",
        mime_type="application/pdf",
        file_type="pdf",
        tags=["finance"],
    )
    record = new_version(previous)
    assert record.organization == "acme"
    assert record.filename == "doc.pdf"
    assert record.mime_type == "application/pdf"
    assert record.file_type == "pdf"
    assert record.tags == ["finance"]


def test_new_version_with_no_previous_requires_required_fields() -> None:
    """Starting from ``None`` still requires the minimal required fields."""
    with pytest.raises(Exception):
        new_version(None)  # missing owner / organization / filename / checksum


def test_new_version_returns_document_version_type() -> None:
    """The result is a ``DocumentVersion`` (``DocumentRecord`` alias)."""
    record = new_version(
        None, organization="acme", filename="x", owner="me", checksum="h"
    )
    assert isinstance(record, DocumentVersion)


def test_new_version_preserves_chunk_metadata() -> None:
    """Carried-over fields include nested structures like ``chunk_ids``."""
    chunk_ids = ["c1", "c2"]
    previous = _make_document(chunk_ids=chunk_ids, chunk_count=2)
    record = new_version(previous)
    assert record.chunk_ids == chunk_ids
    assert record.chunk_count == 2


def test_new_version_overrides_business_fields() -> None:
    """Caller-supplied owner / organization / filename take effect."""
    previous = _make_document(organization="acme", filename="old.pdf")
    record = new_version(
        previous, organization="globex", filename="new.pdf"
    )
    assert record.organization == "globex"
    assert record.filename == "new.pdf"
    # document_id is carried over (not overridden)
    assert record.document_id == previous.document_id


def test_new_version_uses_offset_for_updated_at() -> None:
    """``updated_at`` is updated even when the previous record is very recent."""
    previous = _make_document(updated_at=datetime.now(UTC) - timedelta(seconds=5))
    record = new_version(previous)
    assert record.updated_at >= previous.updated_at