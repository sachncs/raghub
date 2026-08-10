"""Coverage tests for :mod:`raghub.domain` repository ABCs and protocol contracts."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from raghub.models import (
    Chunk,
    Classification,
    Document,
    DocumentLifecycleStatus,
    Session,
    Turn,
)


def make_chunk(**overrides: Any) -> Chunk:
    """Build a Chunk fixture."""
    defaults: dict[str, Any] = {
        "id": "c1",
        "document_id": "d1",
        "version": 1,
        "text": "Revenue grew.",
        "classification": Classification.Internal,
        "company": "acme",
        "owner": "alice@example.com",
        "checksum": "0" * 64,
    }
    defaults.update(overrides)
    return Chunk(**defaults)


def make_document(**overrides: Any) -> Document:
    """Build a Document fixture."""
    now = datetime(2026, 1, 1, tzinfo=UTC)
    defaults: dict[str, Any] = {
        "id": "d1",
        "version": 1,
        "checksum": "c1",
        "created_at": now,
        "updated_at": now,
        "owner": "alice@example.com",
        "organization": "acme",
        "classification": Classification.Internal,
    }
    defaults.update(overrides)
    return Document(**defaults)


def make_session(**overrides: Any) -> Session:
    """Build a Session fixture."""
    now = datetime(2026, 1, 1, tzinfo=UTC)
    defaults: dict[str, Any] = {
        "user_id": "alice",
        "token": "t1",
        "created_at": now,
        "expires_at": now,
        "last_seen_at": now,
    }
    defaults.update(overrides)
    return Session(**defaults)


# ---------------------------------------------------------------------------
# Direct Pydantic model usage (the wrappers were deleted in Phase C)
# ---------------------------------------------------------------------------


def test_chunk_model_supports_attribute_assignment() -> None:
    """``Chunk`` supports direct attribute assignment."""

    chunk = make_chunk()
    chunk.text = "new text"
    assert chunk.text == "new text"


def test_chunk_model_copy_creates_independent_instance() -> None:
    """``Chunk.model_copy`` produces an independent copy for mutations."""

    chunk = make_chunk()
    copy = chunk.model_copy(update={"text": "updated"})
    assert copy.text == "updated"
    assert chunk.text == "Revenue grew."
    assert copy is not chunk


def test_document_model_supports_status_assignment() -> None:
    """``Document.status`` can be reassigned via Pydantic equality."""

    doc = make_document(status=DocumentLifecycleStatus.New)
    doc.status = DocumentLifecycleStatus.Ready
    assert doc.status == DocumentLifecycleStatus.Ready


def test_document_mark_failed_via_model_copy() -> None:
    """``Document`` failure pattern: ``model_copy`` with updated status + error."""

    doc = make_document()
    updated = doc.model_copy(update={"status": DocumentLifecycleStatus.Failed, "error": "boom"})
    assert updated.status == DocumentLifecycleStatus.Failed
    assert updated.error == "boom"


def test_session_model_supports_history_mutation() -> None:
    """``Session.history`` can be appended to directly."""

    session = make_session()
    session.history.append(Turn(question="q1", answer="a1"))
    assert len(session.history) == 1
    assert session.history[0].question == "q1"


def test_session_model_supports_attribute_assignment() -> None:
    """``Session`` supports direct attribute assignment."""

    session = make_session()
    session.user_id = "bob"
    assert session.user_id == "bob"


def test_session_history_clear() -> None:
    """``Session.history.clear()`` empties the conversation history."""

    session = make_session()
    session.history = [Turn(question="q1", answer="a1"), Turn(question="q2", answer="a2")]
    session.history.clear()
    assert session.history == []


def test_session_last_seen_at_is_writable() -> None:
    """``Session.last_seen_at`` can be updated to extend the session lifetime."""

    session = make_session()
    later = datetime(2026, 6, 1, tzinfo=UTC)
    session.last_seen_at = later
    assert session.last_seen_at == later


def test_document_owner_assignment_via_pydantic() -> None:
    """``Document.owner`` can be reassigned via direct attribute write."""

    doc = make_document(owner="alice@example.com")
    doc.owner = "bob@example.com"
    assert doc.owner == "bob@example.com"


def test_chunk_classification_assignment_via_pydantic() -> None:
    """``Chunk.classification`` can be reassigned via direct attribute write."""

    chunk = make_chunk(classification=Classification.Internal)
    chunk.classification = Classification.Confidential
    assert chunk.classification == Classification.Confidential
