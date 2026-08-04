"""SQLite-backed :class:`DocStore` and :class:`SessionStore` coverage."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import aiosqlite

from raghub.models import (
    Classification,
    Document,
    DocumentLifecycleStatus,
    Session,
    Turn,
)
from raghub.repos import DocStore, SessionStore
from raghub.stores import Database, Sessions


def _make_document(**overrides: Any) -> Document:
    """Build a Document fixture with sensible defaults."""
    now = datetime.now(UTC)
    defaults: dict[str, Any] = {
        "id": "doc-1",
        "version": 1,
        "checksum": "0" * 64,
        "created_at": now,
        "updated_at": now,
        "owner": "alice@example.com",
        "organization": "acme",
        "department": "finance",
        "tags": ["q3-2024"],
        "classification": Classification.INTERNAL,
        "status": DocumentLifecycleStatus.READY,
        "filename": "doc.txt",
        "file_type": "txt",
        "mime_type": "text/plain",
        "chunk_count": 0,
        "chunk_ids": [],
    }
    defaults.update(overrides)
    return Document(**defaults)


def _make_session(**overrides: Any) -> Session:
    """Build a Session fixture with sensible defaults."""
    now = datetime.now(UTC)
    defaults: dict[str, Any] = {
        "user_id": "alice",
        "token": "tok-1",
        "created_at": now,
        "expires_at": now + timedelta(seconds=3600),
        "last_seen_at": now,
    }
    defaults.update(overrides)
    return Session(**defaults)


# ---------------------------------------------------------------------------
# DocStore
# ---------------------------------------------------------------------------


async def test_doc_store_initialize_creates_schema() -> None:
    """``initialize`` creates the documents table."""
    mgr = Database(":memory:")
    await mgr.connect()
    try:
        store = DocStore(":memory:", db_manager=mgr)
        await store.initialize()
        cursor = await mgr.connection.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='documents'"
        )
        row = await cursor.fetchone()
        assert row is not None
        assert row[0] == "documents"
    finally:
        await mgr.close()


async def test_doc_store_save_inserts_and_replaces() -> None:
    """``save`` upserts a Document by primary key."""
    mgr = Database(":memory:")
    await mgr.connect()
    try:
        store = DocStore(":memory:", db_manager=mgr)
        await store.initialize()
        await store.save(_make_document(id="d1", version=1, checksum="c1"))
        fetched = await store.get("d1")
        assert fetched is not None
        assert fetched.id == "d1"
        assert fetched.owner == "alice@example.com"
    finally:
        await mgr.close()


async def test_doc_store_save_replaces_existing() -> None:
    """A second ``save`` with the same (id, version) replaces the row."""
    mgr = Database(":memory:")
    await mgr.connect()
    try:
        store = DocStore(":memory:", db_manager=mgr)
        await store.initialize()
        await store.save(_make_document(id="d1", version=1, owner="old@x", checksum="c1"))
        await store.save(_make_document(id="d1", version=1, owner="new@x", checksum="c1"))
        fetched = await store.get("d1")
        assert fetched.owner == "new@x"
    finally:
        await mgr.close()


async def test_doc_store_try_insert_returns_true() -> None:
    """``try_insert`` returns ``True`` on success."""
    mgr = Database(":memory:")
    await mgr.connect()
    try:
        store = DocStore(":memory:", db_manager=mgr)
        await store.initialize()
        assert await store.try_insert(_make_document(id="d1", checksum="c1")) is True
        assert (await store.get("d1")) is not None
    finally:
        await mgr.close()


async def test_doc_store_get_version_returns_specific() -> None:
    """``get_version`` returns the requested version when present."""
    mgr = Database(":memory:")
    await mgr.connect()
    try:
        store = DocStore(":memory:", db_manager=mgr)
        await store.initialize()
        await store.save(_make_document(id="d1", version=1, owner="v1@x", checksum="c1"))
        await store.save(_make_document(id="d1", version=2, owner="v2@x", checksum="c2"))
        doc = await store.get_version("d1", version=2)
        assert doc is not None
        assert doc.owner == "v2@x"
    finally:
        await mgr.close()


async def test_doc_store_get_returns_latest_version() -> None:
    """``get`` returns the latest version (no version arg)."""
    mgr = Database(":memory:")
    await mgr.connect()
    try:
        store = DocStore(":memory:", db_manager=mgr)
        await store.initialize()
        await store.save(_make_document(id="d1", version=1, owner="v1@x", checksum="c1"))
        await store.save(_make_document(id="d1", version=2, owner="v2@x", checksum="c2"))
        doc = await store.get("d1")
        assert doc is not None
        assert doc.version == 2
    finally:
        await mgr.close()


async def test_doc_store_get_returns_none_for_unknown() -> None:
    """``get`` returns ``None`` for an unknown document."""
    mgr = Database(":memory:")
    await mgr.connect()
    try:
        store = DocStore(":memory:", db_manager=mgr)
        await store.initialize()
        assert await store.get("missing") is None
    finally:
        await mgr.close()


async def test_doc_store_list_versions_returns_all() -> None:
    """``list_versions`` returns every version in ascending order."""
    mgr = Database(":memory:")
    await mgr.connect()
    try:
        store = DocStore(":memory:", db_manager=mgr)
        await store.initialize()
        await store.save(_make_document(id="d1", version=2, owner="v2@x", checksum="c2"))
        await store.save(_make_document(id="d1", version=1, owner="v1@x", checksum="c1"))
        versions = await store.list_versions("d1")
        assert [v.version for v in versions] == [1, 2]
    finally:
        await mgr.close()


async def test_doc_store_list_versions_empty_for_unknown() -> None:
    """``list_versions`` returns ``[]`` for an unknown document."""
    mgr = Database(":memory:")
    await mgr.connect()
    try:
        store = DocStore(":memory:", db_manager=mgr)
        await store.initialize()
        assert await store.list_versions("missing") == []
    finally:
        await mgr.close()


async def test_doc_store_get_by_checksum() -> None:
    """``get_by_checksum`` returns the latest row matching the checksum."""
    mgr = Database(":memory:")
    await mgr.connect()
    try:
        store = DocStore(":memory:", db_manager=mgr)
        await store.initialize()
        await store.save(_make_document(id="d1", version=1, checksum="c1"))
        await store.save(_make_document(id="d1", version=2, checksum="c1"))
        doc = await store.get_by_checksum("c1")
        assert doc is not None
        assert doc.version == 2
    finally:
        await mgr.close()


async def test_doc_store_get_by_checksum_unknown() -> None:
    """``get_by_checksum`` returns ``None`` for an unknown checksum."""
    mgr = Database(":memory:")
    await mgr.connect()
    try:
        store = DocStore(":memory:", db_manager=mgr)
        await store.initialize()
        assert await store.get_by_checksum("missing") is None
    finally:
        await mgr.close()


async def test_doc_store_delete_removes_all_versions() -> None:
    """``delete`` removes every version of a document."""
    mgr = Database(":memory:")
    await mgr.connect()
    try:
        store = DocStore(":memory:", db_manager=mgr)
        await store.initialize()
        await store.save(_make_document(id="d1", version=1, checksum="c1"))
        await store.save(_make_document(id="d1", version=2, checksum="c2"))
        await store.delete("d1")
        assert await store.get("d1") is None
        assert await store.list_versions("d1") == []
    finally:
        await mgr.close()


async def test_doc_store_delete_version_only_removes_one() -> None:
    """``delete_version`` only removes the specified version."""
    mgr = Database(":memory:")
    await mgr.connect()
    try:
        store = DocStore(":memory:", db_manager=mgr)
        await store.initialize()
        await store.save(_make_document(id="d1", version=1, owner="v1@x", checksum="c1"))
        await store.save(_make_document(id="d1", version=2, owner="v2@x", checksum="c2"))
        await store.delete_version("d1", 1)
        versions = await store.list_versions("d1")
        assert [v.version for v in versions] == [2]
    finally:
        await mgr.close()


async def test_doc_store_list_by_organization() -> None:
    """``list_by_organization`` returns the latest version of every doc."""
    mgr = Database(":memory:")
    await mgr.connect()
    try:
        store = DocStore(":memory:", db_manager=mgr)
        await store.initialize()
        await store.save(_make_document(id="d1", organization="acme", version=1, checksum="c1"))
        await store.save(_make_document(id="d2", organization="acme", version=1, checksum="c2"))
        await store.save(_make_document(id="d3", organization="other", version=1, checksum="c3"))
        docs = await store.list_by_organization("acme")
        assert {d.id for d in docs} == {"d1", "d2"}
    finally:
        await mgr.close()


async def test_doc_store_list_by_organization_returns_latest() -> None:
    """``list_by_organization`` returns the latest version, not historical ones."""
    mgr = Database(":memory:")
    await mgr.connect()
    try:
        store = DocStore(":memory:", db_manager=mgr)
        await store.initialize()
        await store.save(
            _make_document(id="d1", organization="acme", version=1, owner="v1@x", checksum="c1")
        )
        await store.save(
            _make_document(id="d1", organization="acme", version=2, owner="v2@x", checksum="c2")
        )
        docs = await store.list_by_organization("acme")
        assert len(docs) == 1
        assert docs[0].version == 2
        assert docs[0].owner == "v2@x"
    finally:
        await mgr.close()


async def test_doc_store_list_all() -> None:
    """``list_all`` returns the latest version of every document."""
    mgr = Database(":memory:")
    await mgr.connect()
    try:
        store = DocStore(":memory:", db_manager=mgr)
        await store.initialize()
        await store.save(_make_document(id="d1", organization="acme", version=1, checksum="c1"))
        await store.save(_make_document(id="d2", organization="other", version=1, checksum="c2"))
        docs = await store.list_all()
        assert {d.id for d in docs} == {"d1", "d2"}
    finally:
        await mgr.close()


async def test_doc_store_update_status() -> None:
    """``update_status`` changes the latest version's lifecycle status."""
    mgr = Database(":memory:")
    await mgr.connect()
    try:
        store = DocStore(":memory:", db_manager=mgr)
        await store.initialize()
        await store.save(_make_document(id="d1", version=1, checksum="c1"))
        await store.update_status("d1", DocumentLifecycleStatus.FAILED)
        doc = await store.get("d1")
        assert doc is not None
        assert doc.status == DocumentLifecycleStatus.FAILED
    finally:
        await mgr.close()


async def test_doc_store_as_record_round_trip() -> None:
    """``as_record`` reconstructs a Document from an aiosqlite row."""
    mgr = Database(":memory:")
    await mgr.connect()
    try:
        store = DocStore(":memory:", db_manager=mgr)
        await store.initialize()
        doc = _make_document(
            id="d1", tags=["a", "b"], chunks=["c1", "c2"], checksum="c1"
        )
        await store.save(doc)
        cursor = await mgr.connection.execute(
            "SELECT * FROM documents WHERE document_id = ?", ("d1",)
        )
        row = await cursor.fetchone()
        reconstructed = DocStore.as_record(row)
        assert reconstructed.id == "d1"
        assert reconstructed.tags == ["a", "b"]
        assert reconstructed.chunks == ["c1", "c2"]
        assert reconstructed.classification == Classification.INTERNAL
    finally:
        await mgr.close()


# ---------------------------------------------------------------------------
# SessionStore
# ---------------------------------------------------------------------------


async def test_session_store_initialize_creates_table(tmp_path: Path) -> None:
    """``initialize`` creates the sessions table."""
    store = SessionStore(tmp_path / "sess.db")
    await store.initialize()
    conn = await aiosqlite.connect(str(tmp_path / "sess.db"))
    try:
        cursor = await conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='sessions'"
        )
        row = await cursor.fetchone()
        assert row is not None
        assert row[0] == "sessions"
    finally:
        await conn.close()


async def test_session_store_create_and_load_round_trip(tmp_path: Path) -> None:
    """``create`` writes a row that ``Sessions`` can later load."""
    store = SessionStore(tmp_path / "sess.db")
    await store.initialize()
    record = _make_session(token="tok-x")
    await store.create(record)
    sessions = Sessions(tmp_path / "sess.db", 3600)
    await sessions.initialize()
    loaded = await sessions.get_by_token("tok-x")
    assert loaded is not None
    assert loaded.user_id == "alice"


async def test_session_store_create_from_record_preserves_history(tmp_path: Path) -> None:
    """``create_from_record`` persists the full history column."""
    store = SessionStore(tmp_path / "sess.db")
    await store.initialize()
    record = _make_session(token="tok-y")
    record.history = [Turn(question="q1", answer="a1"), Turn(question="q2", answer="a2")]
    await store.create_from_record(record)
    sessions = Sessions(tmp_path / "sess.db", 3600)
    await sessions.initialize()
    loaded = await sessions.get_by_token("tok-y")
    assert loaded is not None
    assert len(loaded.history) == 2
    assert loaded.history[0].question == "q1"


def test_session_store_constructs_with_default_timeout(tmp_path: Path) -> None:
    """``SessionStore`` defaults ``timeout_seconds`` to 3600."""
    store = SessionStore(tmp_path / "sess.db")
    assert store.inner.timeout == timedelta(seconds=3600)


def test_session_store_constructs_with_custom_timeout(tmp_path: Path) -> None:
    """A custom ``timeout_seconds`` is forwarded to :class:`Sessions`."""
    store = SessionStore(tmp_path / "sess.db", timeout_seconds=120)
    assert store.inner.timeout == timedelta(seconds=120)


async def test_session_store_inherits_db_manager(tmp_path: Path) -> None:
    """A supplied ``db_manager`` is exposed as an attribute."""
    mgr = Database(str(tmp_path / "shared.db"))
    await mgr.connect()
    try:
        store = SessionStore(tmp_path / "ignored.db", db_manager=mgr)
        assert store.db_manager is mgr
    finally:
        await mgr.close()
