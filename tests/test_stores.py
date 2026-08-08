"""Stores module coverage tests.

Exercises :class:`Documents`, :class:`JsonSessions`, and
:class:`ImageStore`. The :class:`Sessions` SQLite-backed store is
lightly exercised because it requires the aiosqlite dep; deeper tests
live in :mod:`tests/test_sqlite_store` and the phase-4 integration
suite.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from raghub.models import Document, DocumentLifecycleStatus


def _make_document(
    document_id: str = "d1",
    version: int = 1,
    checksum: str = "abc123",
    company: str = "acme",
) -> Document:
    """Build a Document fixture with safe defaults."""

    from datetime import UTC, datetime

    return Document(
        id=document_id,
        version=version,
        source="mem://x",
        organization=company,
        owner="alice@example.com",
        checksum=checksum,
        created_at=datetime.now(UTC),
    )


# ---------------------------------------------------------------------------
# Documents
# ---------------------------------------------------------------------------


@pytest.fixture
def docs_registry(tmp_path: Path) -> DocumentsLike:
    """A fresh :class:`Documents` registry on a tmp file."""

    from raghub.stores import Documents

    return Documents(tmp_path / "documents.json")


class DocumentsLike:
    """Lightweight stand-in for type hints."""

    pass


def test_documents_init_empty(tmp_path: Path) -> None:
    """An empty Documents registry has no documents and no checksums."""

    from raghub.stores import Documents

    registry = Documents(tmp_path / "empty.json")
    assert registry.documents == {}
    assert registry.checksum_index == {}


def test_documents_load_with_invalid_json_resets(tmp_path: Path) -> None:
    """A malformed file resets Documents to empty state."""

    from raghub.stores import Documents

    path = tmp_path / "bad.json"
    path.write_text("not-json", encoding="utf-8")
    registry = Documents(path)
    assert registry.documents == {}


def test_documents_save_and_load_round_trip(tmp_path: Path) -> None:
    """A saved registry is loadable by a fresh instance."""

    from raghub.stores import Documents

    path = tmp_path / "rt.json"
    registry = Documents(path)
    document = _make_document(checksum="rtck1")
    registry.save_version(document)
    assert path.exists()

    other = Documents(path)
    assert other.get_latest(document.id).id == document.id


def test_documents_save_version_archive_predecessor(tmp_path: Path) -> None:
    """A higher version number archives the previous latest."""

    from raghub.stores import Documents

    path = tmp_path / "v.json"
    registry = Documents(path)
    registry.save_version(_make_document(version=1, checksum="a1"))
    registry.save_version(_make_document(version=2, checksum="a2"))
    latest = registry.get_latest("d1")
    assert latest.version == 2
    # The v1 document has been archived by the cascade.
    v1 = registry.get_version("d1", 1)
    assert v1.status == DocumentLifecycleStatus.Archived


def test_documents_save_version_replace_in_place(tmp_path: Path) -> None:
    """Saving the same version number replaces the existing entry."""

    from raghub.stores import Documents

    path = tmp_path / "rep.json"
    registry = Documents(path)
    registry.save_version(_make_document(version=1, checksum="a1"))
    other = _make_document(version=1, checksum="a2")
    registry.save_version(other)
    latest = registry.get_latest("d1")
    assert latest.checksum == "a2"
    # Only one version exists.
    versions = registry.documents["d1"]
    assert len(versions) == 1


def test_documents_get_latest_missing(tmp_path: Path) -> None:
    """get_latest returns None for unknown document ids."""

    from raghub.stores import Documents

    registry = Documents(tmp_path / "x.json")
    assert registry.get_latest("missing") is None


def test_documents_get_version(tmp_path: Path) -> None:
    """get_version finds a particular version (or None)."""

    from raghub.stores import Documents

    path = tmp_path / "gsv.json"
    registry = Documents(path)
    registry.save_version(_make_document(version=1, checksum="g1"))
    registry.save_version(_make_document(version=2, checksum="g2"))
    assert registry.get_version("d1", 2).checksum == "g2"
    assert registry.get_version("d1", 99) is None


def test_documents_by_checksum(tmp_path: Path) -> None:
    """by_checksum resolves a checksum to its Document (or None)."""

    from raghub.stores import Documents

    path = tmp_path / "gbc.json"
    registry = Documents(path)
    document = _make_document(checksum="ck1")
    registry.save_version(document)
    found = registry.by_checksum("ck1")
    assert found is not None
    assert found.id == document.id
    assert registry.by_checksum("missing") is None


def test_documents_list_accessible_filters_by_company(tmp_path: Path) -> None:
    """list_accessible returns non-archived documents for the tenant."""

    from raghub.stores import Documents

    path = tmp_path / "la.json"
    registry = Documents(path)
    registry.save_version(_make_document(document_id="a", company="acme"))
    registry.save_version(_make_document(document_id="b", company="globex"))
    accessible = registry.list_accessible(["acme"])
    ids = {doc.id for doc in accessible}
    assert ids == {"a"}


def test_documents_archive_sets_status(tmp_path: Path) -> None:
    """archive sets the latest to ARCHIVED status."""

    from raghub.stores import Documents

    path = tmp_path / "arch.json"
    registry = Documents(path)
    registry.save_version(_make_document())
    registry.archive("d1")
    latest = registry.get_latest("d1")
    assert latest.status == DocumentLifecycleStatus.Archived


def test_documents_archive_unknown_is_noop(tmp_path: Path) -> None:
    """archive on an unknown id is a silent no-op."""

    from raghub.stores import Documents

    registry = Documents(tmp_path / "arch2.json")
    registry.archive("missing")  # does not raise


def test_documents_dump_snapshot(tmp_path: Path) -> None:
    """dump returns an in-memory Snapshot reflecting the registry."""

    from raghub.stores import Documents

    path = tmp_path / "snap.json"
    registry = Documents(path)
    registry.save_version(_make_document())
    snapshot = registry.dump()
    assert "d1" in snapshot.documents


# ---------------------------------------------------------------------------
# JsonSessions
# ---------------------------------------------------------------------------


@pytest.fixture
def json_sessions(tmp_path: Path) -> JsonSessionsLike:
    from raghub.stores import JsonSessions

    return JsonSessions(tmp_path / "sessions.json", timeout_seconds=3600)


class JsonSessionsLike:
    """Lightweight stand-in for type hints."""

    pass


def test_json_sessions_init_empty(tmp_path: Path) -> None:
    """An empty JsonSessions has no sessions."""

    from raghub.stores import JsonSessions

    store = JsonSessions(tmp_path / "s.json", timeout_seconds=3600)
    assert store.sessions == {}


def test_json_sessions_create_and_get(tmp_path: Path) -> None:
    """create() persists a session and resolve() finds it."""

    from raghub.stores import JsonSessions

    store = JsonSessions(tmp_path / "s2.json", timeout_seconds=3600)
    session = store.create(user_id="u1")
    assert session.user_id == "u1"
    found = store.resolve(session.token)
    assert found is not None
    assert found.user_id == "u1"


def test_json_sessions_get_unknown_returns_none(tmp_path: Path) -> None:
    """resolve returns None for an unknown token."""

    from raghub.stores import JsonSessions

    store = JsonSessions(tmp_path / "s3.json", timeout_seconds=3600)
    assert store.resolve("nope") is None


def test_json_sessions_save_load_round_trip(tmp_path: Path) -> None:
    """A saved JsonSessions is loadable by a fresh instance."""

    from raghub.stores import JsonSessions

    path = tmp_path / "rt.json"
    store = JsonSessions(path, timeout_seconds=3600)
    session = store.create(user_id="u1")
    other = JsonSessions(path, timeout_seconds=3600)
    assert other.resolve(session.token) is not None


def test_json_sessions_load_corrupt_resets(tmp_path: Path) -> None:
    """A corrupt JSON file resets the store to empty."""

    from raghub.stores import JsonSessions

    path = tmp_path / "bad.json"
    path.write_text("[1,2,3]", encoding="utf-8")
    store = JsonSessions(path, timeout_seconds=3600)
    assert store.sessions == {}


def test_json_sessions_invalidate(tmp_path: Path) -> None:
    """invalidate removes a token."""

    from raghub.stores import JsonSessions

    store = JsonSessions(tmp_path / "d.json", timeout_seconds=3600)
    session = store.create(user_id="u1")
    store.invalidate(session.token)
    assert store.resolve(session.token) is None


def test_json_sessions_invalidate_unknown_noop(tmp_path: Path) -> None:
    """invalidate on an unknown token is silent."""

    from raghub.stores import JsonSessions

    store = JsonSessions(tmp_path / "d2.json", timeout_seconds=3600)
    store.invalidate("nope")  # does not raise


def test_json_sessions_expired_session_is_purged(tmp_path: Path) -> None:
    """A session past its expires_at is purged on the next resolve call."""

    from datetime import UTC, datetime, timedelta

    from raghub.stores import JsonSessions

    store = JsonSessions(tmp_path / "exp.json", timeout_seconds=3600)
    session = store.create(user_id="u1")
    # Force the expiry into the past.
    session.expires_at = datetime.now(UTC) - timedelta(seconds=1)
    store.save()
    assert store.resolve(session.token) is None


def test_json_sessions_resolve_slides_window(tmp_path: Path) -> None:
    """A successful resolve pushes the expiry window forward."""

    from raghub.stores import JsonSessions

    store = JsonSessions(tmp_path / "slide.json", timeout_seconds=3600)
    session = store.create(user_id="u1")
    original = session.expires_at
    store.resolve(session.token)
    # The in-memory entry now has a later expires_at.
    assert store.sessions[session.token].expires_at > original


# ---------------------------------------------------------------------------
# ImageStore
# ---------------------------------------------------------------------------


def test_image_store_save_returns_hash(tmp_path: Path) -> None:
    """ImageStore.save returns the SHA-256 content hash."""

    from raghub.stores import ImageStore

    store = ImageStore(base_path=tmp_path)
    h = store.save(b"png-bytes", extension=".png")
    assert len(h) == 64


def test_image_store_round_trip(tmp_path: Path) -> None:
    """ImageStore.get_bytes returns the original bytes."""

    from raghub.stores import ImageStore

    store = ImageStore(base_path=tmp_path)
    h = store.save(b"hello", extension=".png")
    assert store.get_bytes(h, extension=".png") == b"hello"


def test_image_store_get_path(tmp_path: Path) -> None:
    """ImageStore.get_path resolves to a Path that exists."""

    from raghub.stores import ImageStore

    store = ImageStore(base_path=tmp_path)
    h = store.save(b"data", extension=".jpg")
    path = store.get_path(h, extension=".jpg")
    assert path is not None
    assert path.exists()


def test_image_store_get_path_missing(tmp_path: Path) -> None:
    """ImageStore.get_path returns None for an unknown hash."""

    from raghub.stores import ImageStore

    store = ImageStore(base_path=tmp_path)
    assert store.get_path("0" * 64, extension=".png") is None


def test_image_store_get_bytes_missing(tmp_path: Path) -> None:
    """ImageStore.get_bytes returns None for an unknown hash."""

    from raghub.stores import ImageStore

    store = ImageStore(base_path=tmp_path)
    assert store.get_bytes("0" * 64, extension=".png") is None


def test_image_store_save_idempotent(tmp_path: Path) -> None:
    """Saving the same content twice does not duplicate the file."""

    from raghub.stores import ImageStore

    store = ImageStore(base_path=tmp_path)
    h1 = store.save(b"same", extension=".png")
    h2 = store.save(b"same", extension=".png")
    assert h1 == h2


def test_image_store_delete_removes(tmp_path: Path) -> None:
    """ImageStore.delete returns True for an existing file."""

    from raghub.stores import ImageStore

    store = ImageStore(base_path=tmp_path)
    h = store.save(b"data", extension=".png")
    assert store.delete(h, extension=".png") is True
    assert store.get_path(h, extension=".png") is None


def test_image_store_delete_missing_returns_false(tmp_path: Path) -> None:
    """ImageStore.delete returns False when nothing to remove."""

    from raghub.stores import ImageStore

    store = ImageStore(base_path=tmp_path)
    assert store.delete("0" * 64, extension=".png") is False
