"""Auth module coverage tests.

Exercises :class:`UserRecord`, :class:`SqliteUsers` CRUD, :class:`Authz`
RBAC checks, and :class:`AuthService.login/logout` against an in-memory
container. Real bcrypt + aiosqlite are used so the round-trip is
end-to-end.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from raghub.auth import AuthService, Authz, SqliteUsers, UserRecord
from raghub.errors import AuthenticationError, AuthorizationError, MissingDepError
from raghub.models import AuthLoginResponse, User


def _user(**overrides: object) -> User:
    """Build a User with sensible defaults; per-test overrides go in kwargs."""
    defaults: dict[str, object] = {
        "id": "u1",
        "email": "alice@example.com",
        "allowed_companies": ["acme"],
        "is_admin": False,
    }
    defaults.update(overrides)
    return User(**defaults)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# UserRecord pydantic model
# ---------------------------------------------------------------------------


def test_user_record_generates_uuid_and_timestamp() -> None:
    """UserRecord auto-fills user_id (UUID) and created_at (UTC now)."""
    record = UserRecord(email="a@x.com", password_hash="x")
    assert record.user_id
    assert len(record.user_id) >= 32
    assert record.created_at.tzinfo is not None
    assert record.allowed_companies == []
    assert record.allowed_groups == []
    assert record.is_admin is False


def test_user_record_accepts_all_fields() -> None:
    """UserRecord accepts the full field set."""
    record = UserRecord(
        user_id="uid-1",
        email="a@x.com",
        password_hash="hash",
        allowed_companies=["acme", "globex"],
        allowed_groups=["g1"],
        is_admin=True,
    )
    assert record.user_id == "uid-1"
    assert record.allowed_companies == ["acme", "globex"]
    assert record.is_admin is True


def test_user_record_round_trip_dump() -> None:
    """UserRecord.model_dump round-trips every field."""
    record = UserRecord(
        user_id="uid",
        email="a@x.com",
        password_hash="hash",
        allowed_companies=["c1"],
        allowed_groups=["g1"],
        is_admin=True,
    )
    dumped = record.model_dump()
    rebuilt = UserRecord(**dumped)
    assert rebuilt == record


# ---------------------------------------------------------------------------
# SqliteUsers CRUD (uses aiosqlite + bcrypt for real)
# ---------------------------------------------------------------------------


@pytest.fixture
async def users(tmp_path: Path) -> SqliteUsers:
    """Create a SqliteUsers backed by a temp file."""
    store = SqliteUsers(tmp_path / "users.db")
    await store.initialize()
    return store


@pytest.mark.asyncio
async def test_sqlite_users_create_and_get_by_email(users: SqliteUsers) -> None:
    """create_user persists; get_by_email finds it with hashed password."""
    record = await users.create_user(email="a@x.com", password="hunter2")
    assert record.email == "a@x.com"
    assert record.password_hash != "hunter2"
    assert record.password_hash.startswith("$2")

    found = await users.get_by_email("a@x.com")
    assert found is not None, "found should be set by test setup"
    assert found.user_id == record.user_id


@pytest.mark.asyncio
async def test_sqlite_users_create_returns_record_with_companies(users: SqliteUsers) -> None:
    """create_user honours the companies and is_admin kwargs."""
    record = await users.create_user(
        email="admin@example.com",
        password="long-secure-password",
        companies=["acme", "globex"],
        is_admin=True,
    )
    assert record.allowed_companies == ["acme", "globex"]
    assert record.is_admin is True


@pytest.mark.asyncio
async def test_sqlite_users_get_by_email_missing_returns_none(users: SqliteUsers) -> None:
    """get_by_email on a missing email returns None."""
    found = await users.get_by_email("absent@x.com")
    assert found is None


@pytest.mark.asyncio
async def test_sqlite_users_get_by_id_round_trip(users: SqliteUsers) -> None:
    """get_by_id finds the same record by its UUID."""
    created = await users.create_user(email="byid@x.com", password="pw")
    found = await users.get_by_id(created.user_id)
    assert found is not None, "found should be set by test setup"
    assert found.email == "byid@x.com"


@pytest.mark.asyncio
async def test_sqlite_users_get_by_id_missing(users: SqliteUsers) -> None:
    """get_by_id on a missing user_id returns None."""
    assert await users.get_by_id("missing-uid") is None


@pytest.mark.asyncio
async def test_sqlite_users_verify_password_success(users: SqliteUsers) -> None:
    """verify_password returns the record on a correct password."""
    await users.create_user(email="v@x.com", password="correct-password")
    verified = await users.verify_password("v@x.com", "correct-password")
    assert verified is not None, "verified should be set by test setup"
    assert verified.email == "v@x.com"


@pytest.mark.asyncio
async def test_sqlite_users_verify_password_wrong(users: SqliteUsers) -> None:
    """verify_password returns None on a wrong password."""
    await users.create_user(email="v@x.com", password="correct")
    assert await users.verify_password("v@x.com", "wrong") is None


@pytest.mark.asyncio
async def test_sqlite_users_verify_password_unknown_user(users: SqliteUsers) -> None:
    """verify_password returns None when the email isn't registered."""
    assert await users.verify_password("absent@x.com", "anything") is None


@pytest.mark.asyncio
async def test_sqlite_users_list_users_empty_then_populated(users: SqliteUsers) -> None:
    """list_users reflects rows; empty by default."""
    assert await users.list_users() == []
    a = await users.create_user(email="a@x.com", password="pw")
    b = await users.create_user(email="b@x.com", password="pw")
    listed = await users.list_users()
    assert len(listed) == 2
    ids = {u.user_id for u in listed}
    assert ids == {a.user_id, b.user_id}


@pytest.mark.asyncio
async def test_sqlite_users_double_email_raises(users: SqliteUsers) -> None:
    """create_user raises IntegrityError on a duplicate email."""
    import sqlite3

    await users.create_user(email="dup@x.com", password="pw")
    with pytest.raises(sqlite3.IntegrityError):
        await users.create_user(email="dup@x.com", password="pw")


@pytest.mark.asyncio
async def test_sqlite_users_prefs_round_trip(users: SqliteUsers) -> None:
    """set_pref/get_pref/get_prefs round-trip across scalars and dicts."""
    await users.create_user(email="pref@x.com", password="pw")
    store = await users.get_by_email("pref@x.com")
    assert store is not None, "store should be set by test setup"
    uid = store.user_id

    assert await users.get_pref(uid, "absent") is None
    assert await users.get_prefs(uid) == {}

    await users.set_pref(uid, "reranker", "colbert")
    assert await users.get_pref(uid, "reranker") == "colbert"

    await users.set_pref(uid, "tool_settings", {"max_steps": 8})
    assert await users.get_pref(uid, "tool_settings") == {"max_steps": 8}

    all_prefs = await users.get_prefs(uid)
    assert all_prefs == {"reranker": "colbert", "tool_settings": {"max_steps": 8}}


@pytest.mark.asyncio
async def test_sqlite_users_set_prefs_bulk(users: SqliteUsers) -> None:
    """set_prefs writes many keys in one transaction; empty dict is a no-op."""
    await users.create_user(email="bulk@x.com", password="pw")
    record = await users.get_by_email("bulk@x.com")
    assert record is not None, "record should be set by test setup"
    uid = record.user_id

    await users.set_pref(uid, "existing", "old")
    await users.set_prefs(uid, {"reranker": "cohere", "max_steps": 4})
    prefs = await users.get_prefs(uid)
    assert prefs["existing"] == "old"
    assert prefs["reranker"] == "cohere"
    assert prefs["max_steps"] == 4

    await users.set_prefs(uid, {})
    assert await users.get_prefs(uid) == prefs


@pytest.mark.asyncio
async def test_sqlite_users_delete_pref(users: SqliteUsers) -> None:
    """delete_pref removes the key; deleting again is a no-op."""
    await users.create_user(email="del@x.com", password="pw")
    record = await users.get_by_email("del@x.com")
    assert record is not None, "record should be set by test setup"
    uid = record.user_id

    await users.set_pref(uid, "k", "v")
    await users.delete_pref(uid, "k")
    assert await users.get_pref(uid, "k") is None
    await users.delete_pref(uid, "k")  # second delete is a no-op


# ---------------------------------------------------------------------------
# Authz RBAC
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_authz_check_access_admin_bypass() -> None:
    """Admin users pass check_access for any company."""
    authz = Authz(user_store=None)  # type: ignore[arg-type]
    admin = _user(is_admin=True)
    assert await authz.check_access(admin, "any-company") is True
    assert await authz.check_access(admin, "") is True


@pytest.mark.asyncio
async def test_authz_check_access_allow_list_match(users: SqliteUsers) -> None:
    """Standard user is granted for companies on their allow-list."""
    authz = Authz(user_store=users)
    user = _user(allowed_companies=["acme"])
    assert await authz.check_access(user, "acme") is True


@pytest.mark.asyncio
async def test_authz_check_access_allow_list_miss(users: SqliteUsers) -> None:
    """Standard user is denied for companies off their allow-list."""
    authz = Authz(user_store=users)
    user = _user(allowed_companies=["acme"])
    assert await authz.check_access(user, "globex") is False


@pytest.mark.asyncio
async def test_authz_check_access_logs_on_denial(caplog: pytest.LogCaptureFixture) -> None:
    """When a logger is provided, denial is logged at info level."""

    class Logger:
        def __init__(self) -> None:
            self.events: list[tuple[str, dict[str, object]]] = []

        def info(self, event: str, **kw: object) -> None:
            self.events.append((event, kw))

    log = Logger()
    authz = Authz(user_store=None, logger=log)  # type: ignore[arg-type]
    user = _user(allowed_companies=["acme"])
    assert await authz.check_access(user, "globex") is False
    assert log.events
    assert log.events[0][0] == "audit.rbac.denied"
    assert log.events[0][1]["user_id"] == "u1"


@pytest.mark.asyncio
async def test_authz_filter_companies_standard() -> None:
    """Standard user sees the configured allow-list."""
    authz = Authz(user_store=None)  # type: ignore[arg-type]
    user = _user(allowed_companies=["acme", "globex"])
    assert await authz.filter_companies(user) == ["acme", "globex"]


@pytest.mark.asyncio
async def test_authz_filter_companies_admin_is_empty() -> None:
    """Admin user sees empty list (sentinel for "everything")."""
    authz = Authz(user_store=None)  # type: ignore[arg-type]
    user = _user(is_admin=True)
    assert await authz.filter_companies(user) == []


@pytest.mark.asyncio
async def test_authz_require_admin_passes() -> None:
    """require_admin is silent for admins."""
    authz = Authz(user_store=None)  # type: ignore[arg-type]
    await authz.require_admin(_user(is_admin=True))


@pytest.mark.asyncio
async def test_authz_require_admin_raises() -> None:
    """require_admin raises AuthorizationError for non-admins."""
    authz = Authz(user_store=None)  # type: ignore[arg-type]
    with pytest.raises(AuthorizationError, match="Admin access required"):
        await authz.require_admin(_user(is_admin=False))


# ---------------------------------------------------------------------------
# AuthService login / logout / resolve_user
# ---------------------------------------------------------------------------


class StubContainer:
    """Minimal container exposing the surface AuthService requires."""

    def __init__(self, users: SqliteUsers, session_token: str | None = "tok") -> None:
        self.user_store = users
        self.store = self
        self._session_token = session_token
        self.created_sessions: list[str] = []
        self.deleted_sessions: list[str] = []
        self.bearer_log: list[tuple[str, str | None]] = []
        self._session_user_id = "u1"

    async def create_session(self, user_id: str) -> object:
        self.created_sessions.append(user_id)
        return StubSession(user_id=user_id, token=self._session_token or "tok")

    async def get_by_token(self, token: str) -> object | None:
        self.bearer_log.append((token, "lookup"))
        if token == "absent":
            return None
        return StubSession(user_id=self._session_user_id, token=token)

    async def delete_session(self, session_id: str) -> None:
        self.deleted_sessions.append(session_id)


def _build_get_by_token(user_id: str, valid_token: str) -> object:
    """Return a get_by_token coroutine bound to a specific user_id."""

    async def _get_by_token(token: str) -> object | None:
        if token == "absent":
            return None
        return StubSession(user_id=user_id, token=token)

    return _get_by_token


class StubSession:
    def __init__(self, user_id: str, token: str) -> None:
        self.user_id = user_id
        self.token = token
        self.session_id = "s1"
        self.history: list[object] = []


@pytest.mark.asyncio
async def test_auth_service_login_success(users: SqliteUsers) -> None:
    """A successful login returns an AuthLoginResponse."""
    created = await users.create_user(email="login@x.com", password="pw", companies=["acme"])
    container = StubContainer(users)
    svc = AuthService(container)
    response: AuthLoginResponse = await svc.login("login@x.com", "pw")
    assert response.user_email == "login@x.com"
    assert response.allowed_companies == ["acme"]
    assert response.session_token == "tok"
    assert container.created_sessions == [created.user_id]  # UserRecord.user_id passed


@pytest.mark.asyncio
async def test_auth_service_login_bad_credentials(users: SqliteUsers) -> None:
    """Bad credentials raise AuthenticationError; no session is created."""
    await users.create_user(email="login@x.com", password="pw")
    container = StubContainer(users)
    svc = AuthService(container)
    with pytest.raises(AuthenticationError, match="Invalid email or password"):
        await svc.login("login@x.com", "wrong-password")
    assert container.created_sessions == []


@pytest.mark.asyncio
async def test_auth_service_logout_invalidates_token(users: SqliteUsers) -> None:
    """logout() calls delete_session when the token resolves."""
    container = StubContainer(users, session_token="good")
    svc = AuthService(container)
    await svc.logout("good")
    assert container.deleted_sessions == ["s1"]


@pytest.mark.asyncio
async def test_auth_service_logout_unknown_token_is_silent(users: SqliteUsers) -> None:
    """logout() of an unknown token is a no-op (no raise)."""
    container = StubContainer(users)
    svc = AuthService(container)
    await svc.logout("absent")
    assert container.deleted_sessions == []


@pytest.mark.asyncio
async def test_auth_service_resolve_user_round_trip(users: SqliteUsers) -> None:
    """resolve_user returns (User, history) for a valid token."""
    created = await users.create_user(email="rt@x.com", password="pw", companies=["acme"])
    container = StubContainer(users, session_token="t")
    # The stub creates sessions with user_id="u1"; align the stub to the real UUID.
    container.get_by_token = _build_get_by_token(  # type: ignore[method-assign]
        created.user_id, "t"
    )
    svc = AuthService(container)
    user, history = await svc.resolve_user("t")
    assert user.email == "rt@x.com"
    assert isinstance(history, list)


@pytest.mark.asyncio
async def test_auth_service_resolve_user_unknown_token(users: SqliteUsers) -> None:
    """resolve_user raises AuthenticationError for an unknown token."""
    svc = AuthService(StubContainer(users))
    with pytest.raises(AuthenticationError, match="Invalid or expired session"):
        await svc.resolve_user("absent")


@pytest.mark.asyncio
async def test_auth_service_resolve_user_session_but_user_deleted(
    users: SqliteUsers,
) -> None:
    """resolve_user raises when the user behind the session is gone."""

    class Container:
        user_store = users
        store = None

        def __init__(self) -> None:
            self.store = self

        async def get_by_token(self, token: str) -> object:
            return StubSession(user_id="missing-uid", token=token)

    container = Container()
    svc = AuthService(container)  # type: ignore[arg-type]
    with pytest.raises(AuthenticationError, match="User not found"):
        await svc.resolve_user("t")


@pytest.mark.asyncio
async def test_auth_service_load_tool_settings_no_user_store() -> None:
    """When the container has no user_store, load_tool_settings returns {}."""

    class Empty:
        pass

    svc = AuthService(Empty())  # type: ignore[arg-type]
    assert await svc.load_tool_settings("any") == {}


@pytest.mark.asyncio
async def test_auth_service_load_tool_settings_from_user_store(
    users: SqliteUsers,
) -> None:
    """When user_store exposes get_pref, load_tool_settings returns its dict."""
    await users.create_user(email="ts@x.com", password="pw")
    record = await users.get_by_email("ts@x.com")
    assert record is not None, "record should be set by test setup"
    await users.set_pref(record.user_id, "tool_settings", {"max_steps": 7})
    container = StubContainer(users)
    svc = AuthService(container)
    assert await svc.load_tool_settings(record.user_id) == {"max_steps": 7}


# ---------------------------------------------------------------------------
# Module-level: MissingDepError contract
# ---------------------------------------------------------------------------


def test_missing_dep_error_contract() -> None:
    """MissingDepError is raised with the dep name and install hint."""
    err = MissingDepError("aiosqlite", "pip install raghub[auth]")
    assert err.package == "aiosqlite"
    assert err.hint == "pip install raghub[auth]"
    assert "pip install raghub[auth]" in str(err)
