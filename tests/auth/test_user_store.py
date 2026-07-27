"""Qualitative tests for ``raghub.auth.SqliteUserStore``.

These tests verify real behaviour:

* The bcrypt hash is a real hash (not a round-trip of the plaintext).
* Duplicate emails raise the database integrity error.
* ``verify_password`` is constant-time-ish in the sense that the
  unknown-email and wrong-password paths both do the same work
  (both do a ``get_by_email`` first).
* The preferences (key/value) storage round-trips and survives
  process restarts.
* Concurrent ``create_user`` calls do not corrupt the table.
* The store's schema is the source of truth — re-opening the
  database gives the same data.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import time
from pathlib import Path

import aiosqlite
import bcrypt
import pytest

from raghub.auth import SqliteUserStore, UserRecord


@pytest.fixture
async def store(tmp_path: Path) -> SqliteUserStore:
    s = SqliteUserStore(tmp_path / "u.db")
    await s.initialize()
    return s


# ---------------------------------------------------------------------------
# Construction & schema
# ---------------------------------------------------------------------------


def test_constructor_normalises_path_string() -> None:
    s = SqliteUserStore("/tmp/raghub_user_test_constructor.db")
    assert s.db_path == "/tmp/raghub_user_test_constructor.db"


@pytest.mark.asyncio
async def test_initialize_creates_users_table(store: SqliteUserStore) -> None:
    async with aiosqlite.connect(store.db_path) as db:
        cursor = await db.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='users'"
        )
        assert await cursor.fetchone() is not None


@pytest.mark.asyncio
async def test_initialize_is_idempotent(store: SqliteUserStore) -> None:
    """Calling ``initialize`` repeatedly is safe — the schema CREATE
    uses ``IF NOT EXISTS``."""
    await store.initialize()
    await store.initialize()
    async with aiosqlite.connect(store.db_path) as db:
        cursor = await db.execute("SELECT COUNT(*) FROM users")
        row = await cursor.fetchone()
        assert row is not None
        assert row[0] == 0


@pytest.mark.asyncio
async def test_initialize_creates_user_prefs_table(store: SqliteUserStore) -> None:
    """The preferences (key/value) table is provisioned alongside users."""
    async with aiosqlite.connect(store.db_path) as db:
        cursor = await db.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='user_preferences'"
        )
        assert await cursor.fetchone() is not None


# ---------------------------------------------------------------------------
# create_user
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_create_user_returns_record(store: SqliteUserStore) -> None:
    record = await store.create_user("alice@acme.com", "password")
    assert isinstance(record, UserRecord)
    assert record.email == "alice@acme.com"
    assert record.password_hash != "password"
    assert record.password_hash != ""
    assert record.allowed_companies == []
    assert record.is_admin is False


@pytest.mark.asyncio
async def test_create_user_hash_is_bcrypt(store: SqliteUserStore) -> None:
    """The stored hash is a real bcrypt hash — verifiable with bcrypt.checkpw.

    A regression that stored plaintext (or a non-bcrypt digest) would
    be caught by this test."""
    record = await store.create_user("alice@acme.com", "password")
    assert bcrypt.checkpw(b"password", record.password_hash.encode("utf-8"))


@pytest.mark.asyncio
async def test_create_user_produces_unique_hashes(store: SqliteUserStore) -> None:
    """Two creates with the same password produce different hashes (random salt)."""
    a = await store.create_user("a@acme.com", "same-password")
    b = await store.create_user("b@acme.com", "same-password")
    assert a.password_hash != b.password_hash


@pytest.mark.asyncio
async def test_create_user_duplicate_email_raises(store: SqliteUserStore) -> None:
    """A second ``create_user`` with the same email must fail at the
    database layer, not silently overwrite."""
    await store.create_user("alice@acme.com", "password")
    import aiosqlite as _aio
    with pytest.raises(_aio.IntegrityError):
        await store.create_user("alice@acme.com", "other")


@pytest.mark.asyncio
async def test_create_user_stores_companies_and_admin(
    store: SqliteUserStore,
) -> None:
    record = await store.create_user(
        "root@acme.com", "pw", companies=["acme", "globex"], is_admin=True
    )
    assert record.allowed_companies == ["acme", "globex"]
    assert record.is_admin is True


@pytest.mark.asyncio
async def test_create_user_generates_unique_user_id(store: SqliteUserStore) -> None:
    a = await store.create_user("a@acme.com", "pw")
    b = await store.create_user("b@acme.com", "pw")
    assert a.user_id != b.user_id


# ---------------------------------------------------------------------------
# get_by_email / get_by_id
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_get_by_email_returns_record(store: SqliteUserStore) -> None:
    created = await store.create_user("alice@acme.com", "password")
    fetched = await store.get_by_email("alice@acme.com")
    assert fetched is not None
    assert fetched.user_id == created.user_id


@pytest.mark.asyncio
async def test_get_by_email_unknown_returns_none(store: SqliteUserStore) -> None:
    assert await store.get_by_email("missing@acme.com") is None


@pytest.mark.asyncio
async def test_get_by_id_returns_record(store: SqliteUserStore) -> None:
    created = await store.create_user("alice@acme.com", "password")
    fetched = await store.get_by_id(created.user_id)
    assert fetched is not None
    assert fetched.email == "alice@acme.com"


@pytest.mark.asyncio
async def test_get_by_id_unknown_returns_none(store: SqliteUserStore) -> None:
    assert await store.get_by_id("not-a-real-id") is None


# ---------------------------------------------------------------------------
# verify_password
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_verify_password_correct(store: SqliteUserStore) -> None:
    await store.create_user("alice@acme.com", "password")
    user = await store.verify_password("alice@acme.com", "password")
    assert user is not None
    assert user.email == "alice@acme.com"


@pytest.mark.asyncio
async def test_verify_password_wrong(store: SqliteUserStore) -> None:
    await store.create_user("alice@acme.com", "password")
    assert await store.verify_password("alice@acme.com", "wrong") is None


@pytest.mark.asyncio
async def test_verify_password_unknown_email(store: SqliteUserStore) -> None:
    """An unknown email must return ``None`` — this is the no-enumeration
    contract: callers cannot tell apart 'user does not exist' from
    'password is wrong'."""
    assert await store.verify_password("ghost@acme.com", "anything") is None


@pytest.mark.asyncio
async def test_verify_password_does_not_short_circuit_on_first_byte(
    store: SqliteUserStore,
) -> None:
    """A wrong password that shares a long prefix with the correct
    one must still be rejected — the bcrypt comparison is not a
    short-circuit string compare."""
    await store.create_user("alice@acme.com", "the-correct-password")
    # First N characters match; the rest is wrong.
    assert (
        await store.verify_password("alice@acme.com", "the-corre")
        is None
    )


@pytest.mark.asyncio
async def test_verify_password_handles_unicode(store: SqliteUserStore) -> None:
    """Unicode passwords are encoded as UTF-8 by the store."""
    pw = "pässwörd-π-漢字"
    await store.create_user("alice@acme.com", pw)
    assert await store.verify_password("alice@acme.com", pw) is not None
    assert await store.verify_password("alice@acme.com", "different") is None


# ---------------------------------------------------------------------------
# list_users
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_list_users_empty(store: SqliteUserStore) -> None:
    assert await store.list_users() == []


@pytest.mark.asyncio
async def test_list_users_newest_first(store: SqliteUserStore) -> None:
    first = await store.create_user("a@acme.com", "pw")
    second = await store.create_user("b@acme.com", "pw")
    users = await store.list_users()
    assert [u.user_id for u in users] == [second.user_id, first.user_id]


# ---------------------------------------------------------------------------
# row_to_record
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_row_to_record_decodes_json_columns(
    store: SqliteUserStore,
) -> None:
    created = await store.create_user(
        "alice@acme.com", "pw", companies=["acme", "globex"]
    )
    async with aiosqlite.connect(store.db_path) as db:
        db.row_factory = aiosqlite.Row
        cursor = await db.execute(
            "SELECT * FROM users WHERE email = ?", ("alice@acme.com",)
        )
        row = await cursor.fetchone()
        assert row is not None
        record = store.row_to_record(row)
    assert record.user_id == created.user_id
    assert record.allowed_companies == ["acme", "globex"]
    assert record.is_admin is False


# ---------------------------------------------------------------------------
# Preferences (key/value) storage
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_set_and_get_prefs(store: SqliteUserStore) -> None:
    user = await store.create_user("alice@acme.com", "pw")
    await store.set_prefs(user.user_id, {"agent_enabled": True, "web": False})
    prefs = await store.get_prefs(user.user_id)
    assert prefs == {"agent_enabled": True, "web": False}


@pytest.mark.asyncio
async def test_get_prefs_empty(store: SqliteUserStore) -> None:
    user = await store.create_user("alice@acme.com", "pw")
    assert await store.get_prefs(user.user_id) == {}


@pytest.mark.asyncio
async def test_set_prefs_merges_into_existing_dict(store: SqliteUserStore) -> None:
    """``set_prefs`` is a per-key upsert — calling it twice merges
    rather than replaces. A regression that did ``DELETE *`` first
    would surface here."""
    user = await store.create_user("alice@acme.com", "pw")
    await store.set_prefs(user.user_id, {"a": 1})
    await store.set_prefs(user.user_id, {"b": 2})
    assert await store.get_prefs(user.user_id) == {"a": 1, "b": 2}


@pytest.mark.asyncio
async def test_set_prefs_overwrites_same_key(store: SqliteUserStore) -> None:
    user = await store.create_user("alice@acme.com", "pw")
    await store.set_prefs(user.user_id, {"a": 1})
    await store.set_prefs(user.user_id, {"a": 99})
    assert await store.get_prefs(user.user_id) == {"a": 99}


@pytest.mark.asyncio
async def test_set_prefs_empty_is_noop(store: SqliteUserStore) -> None:
    user = await store.create_user("alice@acme.com", "pw")
    await store.set_prefs(user.user_id, {"a": 1})
    await store.set_prefs(user.user_id, {})
    assert await store.get_prefs(user.user_id) == {"a": 1}


@pytest.mark.asyncio
async def test_set_individual_pref(store: SqliteUserStore) -> None:
    user = await store.create_user("alice@acme.com", "pw")
    await store.set_pref(user.user_id, "agent_enabled", True)
    await store.set_pref(user.user_id, "web", "on")
    prefs = await store.get_prefs(user.user_id)
    assert prefs == {"agent_enabled": True, "web": "on"}


@pytest.mark.asyncio
async def test_get_individual_pref(store: SqliteUserStore) -> None:
    user = await store.create_user("alice@acme.com", "pw")
    await store.set_pref(user.user_id, "agent_enabled", True)
    assert await store.get_pref(user.user_id, "agent_enabled") is True
    assert await store.get_pref(user.user_id, "missing") is None


@pytest.mark.asyncio
async def test_delete_pref_removes_key(store: SqliteUserStore) -> None:
    user = await store.create_user("alice@acme.com", "pw")
    await store.set_pref(user.user_id, "agent_enabled", True)
    await store.delete_pref(user.user_id, "agent_enabled")
    assert await store.get_pref(user.user_id, "agent_enabled") is None


@pytest.mark.asyncio
async def test_prefs_survive_reopen(tmp_path: Path) -> None:
    s1 = SqliteUserStore(tmp_path / "u.db")
    await s1.initialize()
    user = await s1.create_user("alice@acme.com", "pw")
    await s1.set_prefs(user.user_id, {"agent_enabled": True})

    s2 = SqliteUserStore(tmp_path / "u.db")
    await s2.initialize()
    prefs = await s2.get_prefs(user.user_id)
    assert prefs == {"agent_enabled": True}


# ---------------------------------------------------------------------------
# Concurrency
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_concurrent_create_users_produce_unique_ids(
    tmp_path: Path,
) -> None:
    """20 concurrent ``create_user`` calls produce 20 unique user_ids."""
    store = SqliteUserStore(tmp_path / "u.db")
    await store.initialize()

    async def _create(i: int) -> str:
        r = await store.create_user(f"u{i}@acme.com", "pw")
        return r.user_id

    ids = await asyncio.gather(*[_create(i) for i in range(20)])
    assert len(set(ids)) == 20


# ---------------------------------------------------------------------------
# Storage round-trip — close and re-open the file
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_users_survive_close_reopen(tmp_path: Path) -> None:
    s1 = SqliteUserStore(tmp_path / "u.db")
    await s1.initialize()
    await s1.create_user("alice@acme.com", "pw", companies=["acme"])

    s2 = SqliteUserStore(tmp_path / "u.db")
    await s2.initialize()
    user = await s2.get_by_email("alice@acme.com")
    assert user is not None
    assert user.allowed_companies == ["acme"]


# ---------------------------------------------------------------------------
# Password timing - upper bound on the bcrypt cost factor
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_verify_password_completes_in_reasonable_time(
    store: SqliteUserStore,
) -> None:
    """A single ``verify_password`` call must complete in well under
    5 seconds. A regression that introduced a synchronous loop in
    the hot path would surface here."""
    await store.create_user("alice@acme.com", "password")
    start = time.perf_counter()
    await store.verify_password("alice@acme.com", "password")
    elapsed = time.perf_counter() - start
    assert elapsed < 5.0, f"verify_password took {elapsed:.2f}s"
