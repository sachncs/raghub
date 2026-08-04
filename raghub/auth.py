"""Authentication, RBAC, and the user store.

The auth domain in one file because the components are tightly
coupled:

* :class:`UserRecord` / :class:`SqliteUsers` — the SQLite-backed
  user CRUD store with bcrypt password hashing.
* :class:`Authz` — admin-only authorisation checks
  used by API dependencies.
* :class:`AuthService` — login / logout / token resolution used by
  the API and CLI.
"""

from __future__ import annotations

import json
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

from raghub.errors import AuthenticationError, AuthorizationError, MissingDepError
from raghub.types import JSONValue

try:
    import aiosqlite
    import bcrypt
except ImportError:
    raise MissingDepError("aiosqlite", "pip install raghub[auth]") from None

from pydantic import BaseModel, Field

from raghub.models import AuthLoginResponse, Turn, User

__all__ = [
    "AuthService",
    "SqliteUsers",
    "UserRecord",
]


class UserRecord(BaseModel):
    """Pydantic model representing a single user.

    Attributes:
        user_id: Stable UUID; primary key.
        email: Login email. Unique.
        password_hash: bcrypt hash; never echoed in API responses.
        allowed_companies: Tenant allow-list; controls which companies
            the user can see in retrieval.
        allowed_groups: Group membership; reserved for future group-based
            authorisation.
        is_admin: ``True`` for admin users (bypass RBAC).
        created_at: UTC timestamp of account creation.

    """

    user_id: str = Field(default_factory=lambda: str(uuid4()))
    email: str
    password_hash: str
    allowed_companies: list[str] = Field(default_factory=list)
    allowed_groups: list[str] = Field(default_factory=list)
    is_admin: bool = False
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


class SqliteUsers:
    """Async CRUD wrapper around the ``users`` SQLite table.

    Each method opens a fresh :mod:`aiosqlite` connection. This is
    intentional: it keeps the surface area simple at the cost of a
    per-call connect.

    Attributes:
        db_path: Filesystem path of the SQLite database file. The file
            is created if it does not exist.

    """

    def __init__(self, db_path: str | Path) -> None:
        """Initialise the store.

        Args:
            db_path: SQLite database file path. Created on first
                :meth:`initialize` if it does not exist.

        """
        self.db_path = str(db_path)

    async def initialize(self) -> None:
        """Create the ``users`` table if it does not already exist.

        Also creates the ``user_preferences`` table (Phase 1.9) used
        for per-user tool/agent settings.
        """
        async with aiosqlite.connect(self.db_path) as db:
            await db.executescript("""
                CREATE TABLE IF NOT EXISTS users (
                    user_id TEXT PRIMARY KEY,
                    email TEXT UNIQUE NOT NULL,
                    password_hash TEXT NOT NULL,
                    allowed_companies TEXT DEFAULT '[]',
                    allowed_groups TEXT DEFAULT '[]',
                    is_admin INTEGER DEFAULT 0,
                    created_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS user_preferences (
                    user_id TEXT NOT NULL,
                    key TEXT NOT NULL,
                    value TEXT NOT NULL,
                    updated_at TEXT NOT NULL DEFAULT (datetime('now')),
                    PRIMARY KEY (user_id, key),
                    FOREIGN KEY (user_id) REFERENCES users(user_id) ON DELETE CASCADE
                );
                CREATE INDEX IF NOT EXISTS idx_user_prefs_user
                    ON user_preferences(user_id);
            """)
            await db.commit()

    async def create_user(
        self,
        email: str,
        password: str,
        companies: list[str] | None = None,
        is_admin: bool = False,
    ) -> UserRecord:
        """Create a new user with a bcrypt-hashed password.

        Args:
            email: The user's email address. Must be unique.
            password: The plaintext password; hashed via
                :func:`bcrypt.hashpw` with a fresh salt.
            companies: Optional initial tenant allow-list.
            is_admin: Whether to grant admin status.

        Returns:
            The persisted :class:`UserRecord`.

        Raises:
            aiosqlite.IntegrityError: If ``email`` already exists.

        """
        password_hash = bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")
        record = UserRecord(
            email=email,
            password_hash=password_hash,
            allowed_companies=companies or [],
            is_admin=is_admin,
        )
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute(
                """
                INSERT INTO users
                    (user_id, email, password_hash,
                     allowed_companies, allowed_groups,
                     is_admin, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    record.user_id,
                    record.email,
                    record.password_hash,
                    json.dumps(record.allowed_companies),
                    json.dumps(record.allowed_groups),
                    int(record.is_admin),
                    record.created_at.isoformat(),
                ),
            )
            await db.commit()
        return record

    async def get_by_email(self, email: str) -> UserRecord | None:
        """Look up a user by email.

        Args:
            email: The user's email address.

        Returns:
            The :class:`UserRecord`, or ``None`` if no such user exists.

        """
        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            cursor = await db.execute("SELECT * FROM users WHERE email = ?", (email,))
            row = await cursor.fetchone()
            if row is None:
                return None
            return self.as_record(row)

    async def get_by_id(self, user_id: str) -> UserRecord | None:
        """Look up a user by id.

        Args:
            user_id: The user's UUID.

        Returns:
            The :class:`UserRecord`, or ``None`` if no such user exists.

        """
        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            cursor = await db.execute("SELECT * FROM users WHERE user_id = ?", (user_id,))
            row = await cursor.fetchone()
            if row is None:
                return None
            return self.as_record(row)

    async def verify_password(self, email: str, password: str) -> UserRecord | None:
        """Verify ``password`` against the stored bcrypt hash.

        Args:
            email: The user's email.
            password: The plaintext password to verify.

        Returns:
            The :class:`UserRecord` on success; ``None`` if the user
            does not exist or the password does not match.

        """
        user = await self.get_by_email(email)
        if user is None:
            return None
        if bcrypt.checkpw(password.encode("utf-8"), user.password_hash.encode("utf-8")):
            return user
        return None

    async def list_users(self) -> list[UserRecord]:
        """List every user ordered by ``created_at`` descending.

        Returns:
            A list of :class:`UserRecord`. Empty when the table is empty.

        """
        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            cursor = await db.execute("SELECT * FROM users ORDER BY created_at DESC")
            rows = await cursor.fetchall()
            return [self.as_record(row) for row in rows]

    @staticmethod
    def as_record(row: aiosqlite.Row) -> UserRecord:
        """Hydrate a :class:`UserRecord` from a SQLite row.

        Args:
            row: An :class:`aiosqlite.Row` from the ``users`` table.

        Returns:
            A fully-typed :class:`UserRecord`.

        """
        data: dict[str, Any] = dict(row)
        data["allowed_companies"] = json.loads(data.get("allowed_companies", "[]"))
        data["allowed_groups"] = json.loads(data.get("allowed_groups", "[]"))
        data["is_admin"] = bool(data["is_admin"])
        data["created_at"] = datetime.fromisoformat(data["created_at"])
        return UserRecord.model_validate(data)

    async def get_prefs(self, user_id: str) -> dict[str, Any]:
        """Return every stored preference for ``user_id`` as a dict.

        Args:
            user_id: Owning user id.

        Returns:
            Mapping of preference key → decoded value.

        """
        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            cursor = await db.execute(
                "SELECT key, value FROM user_preferences WHERE user_id = ?",
                (user_id,),
            )
            rows = await cursor.fetchall()
        return {str(row["key"]): json.loads(row["value"]) for row in rows}

    async def get_pref(self, user_id: str, key: str) -> Any:
        """Return one preference value or ``None`` when absent."""
        async with aiosqlite.connect(self.db_path) as db:
            cursor = await db.execute(
                "SELECT value FROM user_preferences WHERE user_id = ? AND key = ?",
                (user_id, key),
            )
            row = await cursor.fetchone()
        if row is None:
            return None
        return json.loads(row[0])

    async def set_pref(
        self, user_id: str, key: str, value: JSONValue
    ) -> None:
        """Upsert a single preference.

        Args:
            user_id: Owning user id.
            key: Preference key. Namespaced by caller.
            value: JSON-serialisable value (str, int, float, bool, None,
                list[JSONValue], dict[str, JSONValue]).

        """
        encoded = json.dumps(value)
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute(
                """
                INSERT INTO user_preferences (user_id, key, value, updated_at)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(user_id, key) DO UPDATE SET
                    value = excluded.value,
                    updated_at = excluded.updated_at
                """,
                (user_id, key, encoded, datetime.now(UTC).isoformat()),
            )
            await db.commit()

    async def set_prefs(self, user_id: str, prefs: dict[str, Any]) -> None:
        """Upsert multiple preferences in a single transaction.

        Args:
            user_id: Owning user id.
            prefs: Key → value mapping; values must be JSON-serialisable.

        """
        if not prefs:
            return
        now = datetime.now(UTC).isoformat()
        async with aiosqlite.connect(self.db_path) as db:
            await db.executemany(
                """
                INSERT INTO user_preferences (user_id, key, value, updated_at)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(user_id, key) DO UPDATE SET
                    value = excluded.value,
                    updated_at = excluded.updated_at
                """,
                [(user_id, k, json.dumps(v), now) for k, v in prefs.items()],
            )
            await db.commit()

    async def delete_pref(self, user_id: str, key: str) -> None:
        """Delete one preference. No-op when the key is absent."""
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute(
                "DELETE FROM user_preferences WHERE user_id = ? AND key = ?",
                (user_id, key),
            )
            await db.commit()


class Authz:
    """Authorisation checks used by admin-only API dependencies.

    Attributes:
        user_store: User store held for future admin-elevation flows.
        logger: Optional loguru-compatible logger.

    """

    def __init__(self, user_store: SqliteUsers, logger: Any | None = None) -> None:
        """Initialise the service.

        Args:
            user_store: Backing user store.
            logger: Optional loguru-compatible logger.

        """
        self.user_store = user_store
        self.logger = logger

    async def check_access(self, user: User, required_company: str) -> bool:
        """Return whether ``user`` may access ``required_company``.

        Args:
            user: The principal performing the access.
            required_company: The company identifier of the resource.

        Returns:
            ``True`` if the user is an admin or if the company is in
            their tenant allow-list; ``False`` otherwise.

        """
        if user.is_admin:
            return True
        allowed = required_company in user.allowed_companies
        if not allowed and self.logger is not None:
            log = getattr(self.logger, "info", None)
            if callable(log):
                log(
                    "audit.rbac.denied",
                    user_id=user.id,
                    email=user.email,
                    required_company=required_company,
                    allowed_companies=list(user.allowed_companies),
                )
        return allowed

    @staticmethod
    async def filter_companies(user: User) -> list[str]:
        """Return the set of companies ``user`` may access.

        Admins see an empty list (a sentinel meaning "everything").

        Args:
            user: The principal being authorised.

        Returns:
            The companies the user may access, or an empty list for
            admin.

        """
        if user.is_admin:
            return []
        return list(user.allowed_companies)

    async def require_admin(self, user: User) -> None:
        """Raise :class:`AuthorizationError` unless ``user.is_admin``.

        Args:
            user: The principal being authorised.

        Raises:
            AuthorizationError: When ``user`` is not an admin.

        """
        if user.is_admin:
            return
        if self.logger is not None:
            log = getattr(self.logger, "warning", None)
            if callable(log):
                log("audit.rbac.admin_required", user_id=user.id, email=user.email)
        raise AuthorizationError("Admin access required")


class AuthService:
    """Login, logout, and principal-resolution operations.

    Attributes:
        container: The application container.

    """

    def __init__(self, container: Any) -> None:
        """Store the container reference.

        Args:
            container: The application container.

        """
        self.container = container

    def log(self, message: str, **payload: Any) -> None:
        """Emit a structured log event."""
        logger = getattr(self.container, "logger", None)
        log_method = getattr(logger, "info", None) if logger else None
        if callable(log_method):
            log_method(message, extra=payload)

    def emit_metric(self, name: str, started_at: float) -> None:
        """Record a latency metric."""
        metrics = getattr(self.container, "metrics", None)
        recorder = getattr(metrics, "record_latency", None) if metrics else None
        if callable(recorder):
            recorder(name, (time.perf_counter() - started_at) * 1000.0)

    async def login(self, email: str, password: str) -> AuthLoginResponse:
        """Verify credentials and create a session.

        Args:
            email: User email.
            password: Plaintext password.

        Returns:
            An :class:`AuthLoginResponse` carrying the session token,
            user email, and allowed companies.

        Raises:
            AuthenticationError: If the email/password combination is
                invalid.

        """
        started = time.perf_counter()
        user = await self.container.user_store.verify_password(email, password)
        if user is None:
            self.log("audit.login.failed", email=email, reason="invalid_credentials")
            raise AuthenticationError("Invalid email or password")
        session = await self.container.store.create_session(user.user_id)
        self.emit_metric("auth_login_latency_ms", started)
        self.log("audit.login.success", email=user.email)
        return AuthLoginResponse(
            session_token=session.token,
            user_email=user.email,
            allowed_companies=user.allowed_companies,
        )

    async def logout(self, token: str) -> None:
        """Invalidate the session associated with ``token``.

        Args:
            token: The bearer token presented by the client.

        """
        session = await self.container.store.get_by_token(token)
        if session is not None:
            await self.container.store.delete_session(session.session_id)

    async def resolve_user(self, token: str) -> tuple[User, list[Turn]]:
        """Resolve a bearer token to (principal, conversation history).

        Args:
            token: The bearer token.

        Returns:
            A tuple of :class:`User` and the session's
            conversation history.

        Raises:
            AuthenticationError: If the token does not correspond to a
                live session, or the underlying user has been deleted.

        """
        session = await self.container.store.get_by_token(token)
        if session is None:
            self.log("audit.token.invalid", reason="no_session")
            raise AuthenticationError("Invalid or expired session")
        record = await self.container.user_store.get_by_id(session.user_id)
        if record is None:
            self.log(
                "audit.token.invalid", user_id=session.user_id, reason="user_deleted"
            )  # session.user_id is FK to User (kept as user_id per FK convention)
            raise AuthenticationError("User not found")
        user = User(
            id=record.user_id,
            email=record.email,
            allowed_companies=record.allowed_companies,
            allowed_groups=record.allowed_groups,
            is_admin=record.is_admin,
            tool_settings=await self.load_tool_settings(record.user_id),
        )
        return user, list(session.history)

    async def load_tool_settings(self, user_id: str) -> dict[str, Any]:
        """Return the ``tool_settings`` prefs blob for ``user_id``.

        Args:
            user_id: The owning user's id.

        Returns:
            The stored ``tool_settings`` dict, or ``{}`` when the
            store lacks the method (e.g. a custom in-memory store
            used by tests).

        """
        store = getattr(self.container, "user_store", None)
        if store is None or not hasattr(store, "get_pref"):
            return {}
        value = await store.get_pref(user_id, "tool_settings")
        return value if isinstance(value, dict) else {}
