"""Coverage tests for :mod:`raghub.domain` reference wrappers and ABCs."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from raghub.domain import (
    ChunkRef,
    DocumentRef,
    SessionWrap,
)
from raghub.models import (
    Chunk,
    Classification,
    Document,
    DocumentLifecycleStatus,
    Session,
    Turn,
)


def _make_chunk(**overrides: Any) -> Chunk:
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


def _make_document(**overrides: Any) -> Document:
    """Build a Document fixture."""
    now = datetime.now(UTC)
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


def _make_session(**overrides: Any) -> Session:
    """Build a Session fixture."""
    now = datetime.now(UTC)
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
# ChunkRef
# ---------------------------------------------------------------------------


def test_chunk_ref_chunk_id_delegates() -> None:
    """``ChunkRef.chunk_id`` returns the wrapped chunk's id."""
    chunk = _make_chunk(id="c-42")
    ref = ChunkRef(chunk)
    assert ref.chunk_id == "c-42"


def test_chunk_ref_getattr_delegates() -> None:
    """Unknown attribute access forwards to the wrapped record."""
    ref = ChunkRef(_make_chunk(text="hello world"))
    assert ref.text == "hello world"


def test_chunk_ref_setattr_updates_record() -> None:
    """``__setattr__`` mutates the wrapped chunk when the attr is real."""
    chunk = _make_chunk()
    ref = ChunkRef(chunk)
    ref.text = "new text"
    assert chunk.text == "new text"


def test_chunk_ref_setattr_creates_internal_state() -> None:
    """``__setattr__`` stores ``_overrides`` keys on the ref itself."""
    ref = ChunkRef(_make_chunk())
    ref._custom = "x"  # type: ignore[attr-defined]
    assert ref._custom == "x"  # type: ignore[attr-defined]


def test_chunk_ref_update_returns_self() -> None:
    """``update`` mutates the record and returns the ref for chaining."""
    chunk = _make_chunk()
    ref = ChunkRef(chunk)
    result = ref.update(text="updated", version=2)
    assert result is ref
    assert chunk.text == "updated"
    assert chunk.version == 2


def test_chunk_ref_init_with_other_ref() -> None:
    """Constructing from another ref returns the underlying chunk."""
    ref = ChunkRef(_make_chunk(id="c-99"))
    ref2 = ChunkRef(ref)
    assert ref2.chunk_id == "c-99"


# ---------------------------------------------------------------------------
# DocumentRef
# ---------------------------------------------------------------------------


def test_document_ref_document_id_delegates() -> None:
    """``DocumentRef.document_id`` returns the wrapped document's id."""
    ref = DocumentRef(_make_document(id="d-42"))
    assert ref.document_id == "d-42"


def test_document_ref_status_getter() -> None:
    """``DocumentRef.status`` reads the wrapped document's status."""
    ref = DocumentRef(
        _make_document(status=DocumentLifecycleStatus.Ready)
    )
    assert ref.status == DocumentLifecycleStatus.Ready


def test_document_ref_status_setter() -> None:
    """``DocumentRef.status`` setter mutates the wrapped document."""
    doc = _make_document(status=DocumentLifecycleStatus.New)
    ref = DocumentRef(doc)
    ref.status = DocumentLifecycleStatus.Ready
    assert doc.status == DocumentLifecycleStatus.Ready


def test_document_ref_getattr_delegates() -> None:
    """Unknown attribute access forwards to the wrapped record."""
    ref = DocumentRef(_make_document(owner="bob@example.com"))
    assert ref.owner == "bob@example.com"


def test_document_ref_update_returns_self() -> None:
    """``update`` mutates the document and returns the ref."""
    doc = _make_document()
    ref = DocumentRef(doc)
    result = ref.update(owner="new@x.com")
    assert result is ref
    assert doc.owner == "new@x.com"


def test_document_ref_mark_failed() -> None:
    """``mark_failed`` sets status to ``FAILED`` and records the error."""
    doc = _make_document()
    ref = DocumentRef(doc)
    ref.mark_failed("boom")
    assert ref.status == DocumentLifecycleStatus.Failed
    assert doc.error == "boom"


def test_document_ref_init_with_other_ref() -> None:
    """Constructing from another ref returns the underlying document."""
    ref = DocumentRef(_make_document(id="d-99"))
    ref2 = DocumentRef(ref)
    assert ref2.document_id == "d-99"


# ---------------------------------------------------------------------------
# SessionWrap
# ---------------------------------------------------------------------------


def test_session_wrap_session_id_delegates() -> None:
    """``SessionWrap.session_id`` returns the wrapped session's id."""
    wrap = SessionWrap(_make_session(id="sess-1"))
    assert wrap.session_id == "sess-1"


def test_session_wrap_history_delegates() -> None:
    """``SessionWrap.history`` returns the wrapped session's history."""
    session = _make_session()
    session.history = [Turn(question="q1", answer="a1")]
    wrap = SessionWrap(session)
    assert len(wrap.history) == 1
    assert wrap.history[0].question == "q1"


def test_session_wrap_getattr_delegates() -> None:
    """Unknown attribute access forwards to the wrapped record."""
    wrap = SessionWrap(_make_session(user_id="alice"))
    assert wrap.user_id == "alice"


def test_session_wrap_setattr_updates_record() -> None:
    """``__setattr__`` mutates the wrapped session for real attrs."""
    session = _make_session()
    wrap = SessionWrap(session)
    wrap.user_id = "bob"
    assert session.user_id == "bob"


def test_session_wrap_add_turn() -> None:
    """``add_turn`` appends a turn and returns the ref."""
    session = _make_session()
    wrap = SessionWrap(session)
    result = wrap.add_turn("q1", "a1")
    assert result is wrap
    assert len(session.history) == 1
    assert session.history[0].question == "q1"


def test_session_wrap_clear() -> None:
    """``clear`` empties the history and returns the ref."""
    session = _make_session()
    session.history = [Turn(question="q1", answer="a1")]
    wrap = SessionWrap(session)
    result = wrap.clear()
    assert result is wrap
    assert session.history == []


def test_session_wrap_add_turn_with_metadata() -> None:
    """``add_turn`` forwards metadata kwargs to :class:`Turn`."""
    session = _make_session()
    wrap = SessionWrap(session)
    wrap.add_turn("q", "a", metadata={"src": "test"})
    assert session.history[0].metadata == {"src": "test"}


def test_session_wrap_history_returns_copy() -> None:
    """``history`` returns a shallow copy; mutating it does not affect the record."""
    session = _make_session()
    session.history = [Turn(question="q1", answer="a1")]
    wrap = SessionWrap(session)
    history = wrap.history
    history.append(Turn(question="q2", answer="a2"))
    assert len(session.history) == 1
