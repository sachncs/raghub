"""Tests for ``raghub.auth.user_store.SqliteUserStore``."""
from __future__ import annotations

from pathlib import Path

import aiosqlite
import pytest

from raghub.auth.user_store import SqliteUserStore, UserRecord


@pytest.fixture
async def store(tmp_path: Path) -> SqliteUserStore:
    """Build a fresh, initialised store in a temp directory."""
    s = SqliteUserStore(tmp_path / "u.db")
    await s.initialize()
    return s


# ---------------------------------------------------------------------------
# Construction & schema
# ---------------------------------------------------------------------------


def test_constructor_accepts_path_string() -> None:
    """Path-like inputs are normalised to ``str``."""
    s = SqliteUserStore("/tmp/raghub_user_test.db")
    assert s.db_path == "/tmp/raghub_user_test.db"


@pytest.mark.asyncio
async def test_initialize_creates_users_table(store: SqliteUserStore) -> None:
    """``initialize`` provisions the ``users`` table."""
    async with aiosqlite.connect(store.db_path) as db:
        cursor = await db.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='users'"
        )
        assert await cursor.fetchone() is not None


@pytest.mark.asyncio
async def test_initialize_is_idempotent(store: SqliteUserStore) -> None:
    """Calling ``initialize`` repeatedly is safe."""
    await store.initialize()
    await store.initialize()
    async with aiosqlite.connect(store.db_path) as db:
        cursor = await db.execute(
            "SELECT COUNT(*) FROM users"
        )
        row = await cursor.fetchone()
        assert row is not None
        assert row[0] == 0


# ---------------------------------------------------------------------------
# create_user / get_by_email / get_by_id
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_create_user_persists_record(store: SqliteUserStore) -> None:
    """``create_user`` returns the persisted ``UserRecord``."""
    record = await store.create_user("alice@acme.com", "password")
    assert isinstance(record, UserRecord)
    assert record.email == "alice@acme.com"
    assert record.password_hash != "password"
    assert record.allowed_companies == []
    assert record.is_admin is False


@pytest.mark.asyncio
async def test_create_user_stores_companies_and_admin_flag(
    store: SqliteUserStore,
) -> None:
    """Constructor kwargs for companies / is_admin round-trip."""
    record = await store.create_user(
        "root@acme.com", "pw", companies=["acme", "globex"], is_admin=True
    )
    assert record.allowed_companies == ["acme", "globex"]
    assert record.is_admin is True


@pytest.mark.asyncio
async def test_get_by_email_returns_record(store: SqliteUserStore) -> None:
    """Lookup by email returns the stored record."""
    created = await store.create_user("alice@acme.com", "password")
    fetched = await store.get_by_email("alice@acme.com")
    assert fetched is not None
    assert fetched.user_id == created.user_id
    assert fetched.email == created.email


@pytest.mark.asyncio
async def test_get_by_email_returns_none_for_unknown(
    store: SqliteUserStore,
) -> None:
    """Unknown emails return ``None``."""
    assert await store.get_by_email("missing@acme.com") is None


@pytest.mark.asyncio
async def test_get_by_id_returns_record(store: SqliteUserStore) -> None:
    """Lookup by id returns the stored record."""
    created = await store.create_user("alice@acme.com", "password")
    fetched = await store.get_by_id(created.user_id)
    assert fetched is not None
    assert fetched.email == "alice@acme.com"


@pytest.mark.asyncio
async def test_get_by_id_returns_none_for_unknown(
    store: SqliteUserStore,
) -> None:
    """Unknown ids return ``None``."""
    assert await store.get_by_id("not-a-real-id") is None


# ---------------------------------------------------------------------------
# verify_password
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_verify_password_returns_record_on_match(
    store: SqliteUserStore,
) -> None:
    """Correct passwords authenticate the user."""
    await store.create_user("alice@acme.com", "password")
    user = await store.verify_password("alice@acme.com", "password")
    assert user is not None
    assert user.email == "alice@acme.com"


@pytest.mark.asyncio
async def test_verify_password_returns_none_for_wrong_password(
    store: SqliteUserStore,
) -> None:
    """Wrong passwords are rejected (return ``None``)."""
    await store.create_user("alice@acme.com", "password")
    assert await store.verify_password("alice@acme.com", "wrong") is None


@pytest.mark.asyncio
async def test_verify_password_returns_none_for_unknown_email(
    store: SqliteUserStore,
) -> None:
    """Unknown emails return ``None`` (no user-enumeration leak)."""
    assert await store.verify_password("ghost@acme.com", "anything") is None


# ---------------------------------------------------------------------------
# list_users
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_list_users_empty(store: SqliteUserStore) -> None:
    """An empty store returns an empty list."""
    assert await store.list_users() == []


@pytest.mark.asyncio
async def test_list_users_returns_all_records_newest_first(
    store: SqliteUserStore,
) -> None:
    """Records are returned newest-first by ``created_at``."""
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
    """JSON-encoded columns are deserialised on hydration."""
    created = await store.create_user(
        "alice@acme.com", "pw", companies=["acme", "globex"]
    )
    async with aiosqlite.connect(store.db_path) as db:
        db.row_factory = aiosqlite.Row
        cursor = await db.execute("SELECT * FROM users WHERE email = ?", ("alice@acme.com",))
        row = await cursor.fetchone()
        assert row is not None
        record = store.row_to_record(row)
    assert record.user_id == created.user_id
    assert record.allowed_companies == ["acme", "globex"]
    assert record.is_admin is False