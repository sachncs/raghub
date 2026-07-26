"""SQLite-backed user store with bcrypt password hashing.

This module defines :class:`UserRecord` (the persisted user model) and
:class:`SqliteUserStore` (the async CRUD wrapper around a SQLite
database). Passwords are hashed with bcrypt; never stored as plaintext.

Schema:

    users (
        user_id TEXT PRIMARY KEY,
        email TEXT UNIQUE NOT NULL,
        password_hash TEXT NOT NULL,
        allowed_companies TEXT DEFAULT '[]',  -- JSON array
        allowed_groups TEXT DEFAULT '[]',     -- JSON array
        is_admin INTEGER DEFAULT 0,           -- 0 / 1
        created_at TEXT NOT NULL              -- ISO 8601
    )

The two JSON columns (``allowed_companies`` and ``allowed_groups``) are
serialised with :func:`json.dumps` and deserialised on read. Bools are
stored as integers per SQLite convention.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

import aiosqlite
import bcrypt
from pydantic import BaseModel, Field


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


class SqliteUserStore:
    """Async CRUD wrapper around the ``users`` SQLite table.

    Each method opens a fresh :mod:`aiosqlite` connection. This is
    intentional: it keeps the surface area simple at the cost of a
    per-call connect. For high-throughput paths, wrap the store in a
    connection pool (e.g. ``aiosqlite``'s ``Connection`` sharing pattern
    plus a semaphore) — outside the scope of this module.

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

        Safe to call multiple times; uses ``CREATE TABLE IF NOT EXISTS``.
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
        # Default bcrypt cost (12) is the modern recommendation; the
        # call is sync but cheap relative to network IO.
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
                INSERT INTO users (user_id, email, password_hash, allowed_companies, allowed_groups, is_admin, created_at)
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
            # ``Row`` enables attribute-style access; we cast to dict
            # before validation so Pydantic gets plain types.
            db.row_factory = aiosqlite.Row
            cursor = await db.execute("SELECT * FROM users WHERE email = ?", (email,))
            row = await cursor.fetchone()
            if row is None:
                return None
            return self.row_to_record(row)

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
            return self.row_to_record(row)

    async def verify_password(self, email: str, password: str) -> UserRecord | None:
        """Verify ``password`` against the stored bcrypt hash.

        Args:
            email: The user's email.
            password: The plaintext password to verify.

        Returns:
            The :class:`UserRecord` on success; ``None`` if the user
            does not exist or the password does not match. Callers
            should not distinguish the two failure modes to avoid
            leaking which emails are registered.
        """
        user = await self.get_by_email(email)
        if user is None:
            return None
        # ``bcrypt.checkpw`` is constant-time relative to the hash,
        # but its cost is dominated by the hashing work (intentionally
        # expensive on the success path).
        if bcrypt.checkpw(password.encode("utf-8"), user.password_hash.encode("utf-8")):
            return user
        return None

    async def list_users(self) -> list[UserRecord]:
        """List every user ordered by ``created_at`` descending.

        Returns:
            A list of :class:`UserRecord`. Empty when the table is
            empty.
        """
        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            cursor = await db.execute("SELECT * FROM users ORDER BY created_at DESC")
            rows = await cursor.fetchall()
            return [self.row_to_record(row) for row in rows]

    def row_to_record(self, row: aiosqlite.Row) -> UserRecord:
        """Hydrate a :class:`UserRecord` from a SQLite row.

        Args:
            row: An :class:`aiosqlite.Row` from the ``users`` table.

        Returns:
            A fully-typed :class:`UserRecord`.
        """
        data: dict[str, Any] = dict(row)
        # JSON columns need explicit decoding; defaults keep the call
        # safe for legacy rows written before the column existed.
        data["allowed_companies"] = json.loads(data.get("allowed_companies", "[]"))
        data["allowed_groups"] = json.loads(data.get("allowed_groups", "[]"))
        # SQLite stores booleans as 0/1 integers.
        data["is_admin"] = bool(data["is_admin"])
        data["created_at"] = datetime.fromisoformat(data["created_at"])
        return UserRecord.model_validate(data)

    # ------------------------------------------------------------------
    # Per-user preferences (Phase 1.10)
    # ------------------------------------------------------------------

    async def get_prefs(self, user_id: str) -> dict[str, Any]:
        """Return every stored preference for ``user_id`` as a dict.

        The values are stored as JSON text; this method decodes each
        value back into its native Python type (``dict``, ``list``,
        ``str``, ``int``, ``float``, ``bool``, ``None``).

        Args:
            user_id: Owning user id.

        Returns:
            Mapping of preference key → decoded value. Empty when the
            user has no preferences or does not exist.
        """
        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            cursor = await db.execute(
                "SELECT key, value FROM user_preferences WHERE user_id = ?",
                (user_id,),
            )
            rows = await cursor.fetchall()
        out: dict[str, Any] = {}
        for row in rows:
            try:
                out[str(row["key"])] = json.loads(row["value"])
            except (TypeError, ValueError):
                # Malformed JSON shouldn't crash a query — skip with
                # the raw text so callers can spot the corruption.
                out[str(row["key"])] = row["value"]
        return out

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
        try:
            return json.loads(row[0])
        except (TypeError, ValueError):
            return row[0]

    async def set_pref(self, user_id: str, key: str, value: Any) -> None:
        """Upsert a single preference.

        Args:
            user_id: Owning user id. A foreign-key violation (unknown
                user) raises :class:`aiosqlite.IntegrityError`.
            key: Preference key. Namespaced by caller (e.g.
                ``"tool_settings"``).
            value: Any JSON-serialisable value. ``None`` is stored as
                JSON ``null``.
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
