from __future__ import annotations

from datetime import datetime

import pytest

from raghub.documents.lifecycle import DocumentLifecycleManager
from raghub.models import Classification, DocumentLifecycleStatus, DocumentVersion, Visibility


def _doc(status: DocumentLifecycleStatus = DocumentLifecycleStatus.NEW) -> DocumentVersion:
    return DocumentVersion(
        document_id="d1",
        version=1,
        checksum="abc",
        created_at=datetime(2025, 1, 1),
        updated_at=datetime(2025, 1, 1),
        owner="alice",
        organization="Acme",
        department="",
        tags=[],
        classification=Classification.INTERNAL,
        visibility=Visibility.ORGANIZATION,
        status=status,
        filename="doc.pdf",
        file_type="pdf",
        mime_type="application/pdf",
        chunk_count=0,
        chunk_ids=[],
        error=None,
    )


def test_lifecycle_legal_transition() -> None:
    mgr = DocumentLifecycleManager()
    doc = _doc(DocumentLifecycleStatus.PROCESSING)
    mgr.transition(doc, DocumentLifecycleStatus.CHUNKING)
    assert doc.status == DocumentLifecycleStatus.CHUNKING


def test_lifecycle_idempotent_transition() -> None:
    mgr = DocumentLifecycleManager()
    doc = _doc(DocumentLifecycleStatus.FAILED)
    mgr.transition(doc, DocumentLifecycleStatus.FAILED)
    assert doc.status == DocumentLifecycleStatus.FAILED


def test_lifecycle_illegal_transition() -> None:
    mgr = DocumentLifecycleManager()
    doc = _doc(DocumentLifecycleStatus.READY)
    with pytest.raises(ValueError, match="Illegal transition"):
        mgr.transition(doc, DocumentLifecycleStatus.NEW)
