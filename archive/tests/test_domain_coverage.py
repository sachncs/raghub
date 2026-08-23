"""Coverage tests for the raghub.models dataclasses.

These tests assert the freeze / copy contract of the dataclass
models. Each assertion uses :meth:`copy` (the ``dataclasses.replace``
thin wrapper inherited from :class:`Snap`) instead of direct attribute
assignment, because the dataclasses are frozen.
"""

from __future__ import annotations

from datetime import UTC, datetime
from hashlib import sha256
from typing import Any

from raghub.models import (
    Chunk,
    Classification,
    Document,
    DocumentLifecycleStatus,
    Session,
    Turn,
)


def sha(text: str) -> str:
    """Return the canonical sha256 hex digest for ``text``."""
    return sha256(text.encode("utf-8")).hexdigest()


def make_chunk(**overrides: Any) -> Chunk:
    """Build a Chunk fixture with a valid checksum."""
    defaults: dict[str, Any] = {
        "id": "c1",
        "document_id": "d1",
        "version": 1,
        "text": "Revenue grew.",
        "classification": Classification.Internal,
        "company": "acme",
        "owner": "alice@example.com",
        "checksum": sha("Revenue grew."),
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


def test_chunk_copy_returns_independent_instance() -> None:
    """``Chunk.copy`` produces an independent copy for mutations."""
    chunk = make_chunk()
    new_chunk = chunk.copy(text="updated")
    assert new_chunk.text == "updated"
    assert chunk.text == "Revenue grew."
    assert new_chunk is not chunk


def test_document_copy_status() -> None:
    """``Document.copy`` can update ``status``."""
    doc = make_document(status=DocumentLifecycleStatus.New)
    ready = doc.copy(status=DocumentLifecycleStatus.Ready)
    assert ready.status == DocumentLifecycleStatus.Ready
    assert doc.status == DocumentLifecycleStatus.New


def test_document_mark_failed_via_copy() -> None:
    """``Document`` failure pattern: ``copy`` with updated status + error."""
    doc = make_document()
    updated = doc.copy(status=DocumentLifecycleStatus.Failed, error="boom")
    assert updated.status == DocumentLifecycleStatus.Failed
    assert updated.error == "boom"


def test_session_history_appends_in_place() -> None:
    """``Session.history`` is a list and can be appended to directly."""
    session = make_session()
    session.history.append(Turn(question="q1", answer="a1"))
    assert len(session.history) == 1
    assert session.history[0].question == "q1"


def test_session_copy_replaces_user_id() -> None:
    """``Session.copy`` replaces ``user_id`` to a new value."""
    session = make_session()
    renamed = session.copy(user_id="bob")
    assert renamed.user_id == "bob"
    assert session.user_id == "alice"


def test_session_history_clear() -> None:
    """``Session.history.clear()`` empties the conversation history."""
    session = make_session()
    session.history.extend([Turn(question="q1", answer="a1"), Turn(question="q2", answer="a2")])
    session.history.clear()
    assert session.history == []


def test_session_last_seen_at_extends_session() -> None:
    """``Session.copy`` updates ``last_seen_at`` to extend the session lifetime."""
    session = make_session()
    later = datetime(2026, 6, 1, tzinfo=UTC)
    renewed = session.copy(last_seen_at=later)
    assert renewed.last_seen_at == later


def test_document_copy_owner() -> None:
    """``Document.copy`` updates the owner field."""
    doc = make_document(owner="alice@example.com")
    transferred = doc.copy(owner="bob@example.com")
    assert transferred.owner == "bob@example.com"
    assert doc.owner == "alice@example.com"


def test_chunk_copy_reclassifies() -> None:
    """``Chunk.copy`` updates the classification field."""
    chunk = make_chunk(classification=Classification.Internal)
    reclassified = chunk.copy(classification=Classification.Confidential)
    assert reclassified.classification == Classification.Confidential
    assert chunk.classification == Classification.Internal
