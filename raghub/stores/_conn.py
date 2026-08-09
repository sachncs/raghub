"""Connection and session-store primitives.

This module is a leaf of the import graph: it has no raghub
dependencies that point back at :mod:`raghub.repos` or
:mod:`raghub.stores`. Both packages depend on this leaf to break
the cycle.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path
from threading import RLock
from typing import Any
from uuid import uuid4

from raghub.constants import DEFAULT_SESSION_TIMEOUT_SECONDS
from raghub.errors import AuthenticationError, MissingDepError
from raghub.io import atomic_write_json, load_json
from raghub.models import Session, Turn

__all__ = [
    "Database",
    "JsonSessions",
    "Sessions",
    "__keyed",
    "serialize_overrides",
]


def __keyed(row: Any) -> bool:
    """Return ``True`` when ``row`` is an :class:`aiosqlite.Row`.

    Defers the import so the module loads without ``aiosqlite``.
    """
    try:
        import aiosqlite
    except ImportError:
        return False
    return isinstance(row, aiosqlite.Row)


def serialize_overrides(overrides: dict[str, Any] | None) -> str:
    """Serialize a session's overrides mapping for SQLite persistence."""
    return json.dumps(overrides or {})


class Database:
    """Manages a shared :class:`aiosqlite.Connection` with WAL mode."""

    def __init__(self, db_path: str | Path) -> None:
        """Store the db path; lazy-connect on first :meth:`connect`."""
        self.db_path = str(db_path)
        self.conn: Any | None = None

    async def connect(self) -> Any:
        """Open (or reuse) the underlying aiosqlite connection."""
        if self.conn is None:
            try:
                import aiosqlite
            except ImportError:
                raise MissingDepError(
                    "aiosqlite",
                    "pip install raghub[auth]",
                ) from None
            self.conn = await aiosqlite.connect(self.db_path, isolation_level=None)
            self.conn.row_factory = aiosqlite.Row
            await self.conn.execute("PRAGMA journal_mode=WAL")
            await self.conn.execute("PRAGMA synchronous=NORMAL")
            await self.conn.execute("PRAGMA foreign_keys=ON")
        return self.conn

    async def close(self) -> None:
        """Checkpoint the WAL and close the connection. Idempotent."""
        if self.conn is not None:
            conn = self.conn
            await conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
            await conn.close()
            self.conn = None

    @property
    def connection(self) -> Any:
        """Return the live connection or raise if not yet connected."""
        if self.conn is None:
            raise RuntimeError("Database not connected. Call connect() first.")
        return self.conn


class JsonSessions:
    """JSON-backed session store with sliding-window inactivity expiry.

    Every successful resolve resets the inactivity timer (expiry bumped
    to ``now + timeout``), implementing the classic "sliding session"
    behaviour: an active user is never logged out, idle sessions are
    pruned on access.

    Reach for the canonical :class:`Sessions` class and call
    ``Sessions.json(...)`` rather than constructing this directly; the
    factory keeps the backends interchangeable.
    """

    def __init__(self, path: str | Path, timeout_seconds: int) -> None:
        """Initialise the store and load existing state."""
        self.path = Path(path)
        self.timeout = timedelta(seconds=timeout_seconds)
        self.lock = RLock()
        self.sessions: dict[str, Session] = {}
        self.load()

    def load(self) -> None:
        """Hydrate in-memory state from disk."""
        if self.path.exists() and not self.path.read_text(encoding="utf-8").lstrip().startswith(
            "{"
        ):
            self.sessions = {}
            return
        try:
            payload = load_json(self.path, default={"sessions": {}})
        except (json.JSONDecodeError, ValueError, TypeError):
            self.sessions = {}
            return
        for token, raw in payload.get("sessions", {}).items():
            self.sessions[token] = Session.model_validate(raw)

    def save(self) -> None:
        """Atomically persist the in-memory sessions map to disk."""
        atomic_write_json(
            self.path,
            {
                "sessions": {
                    token: session.model_dump(mode="json")
                    for token, session in self.sessions.items()
                }
            },
        )

    def create(self, user_id: str) -> Session:
        """Create a fresh session for ``user_id``."""
        now = datetime.now(UTC)
        session = Session(
            id=str(uuid4()),
            user_id=user_id,
            token=str(uuid4()),
            created_at=now,
            expires_at=now + self.timeout,
            last_seen_at=now,
        )
        with self.lock:
            self.sessions[session.token] = session
            self.save()
        return session

    def resolve(self, token: str) -> Session | None:
        """Resolve ``token`` to a live session, sliding the expiry window."""
        with self.lock:
            session = self.sessions.get(token)
            if session is None:
                return None
            now = datetime.now(UTC)
            if now > session.expires_at:
                self.sessions.pop(token, None)
                self.save()
                return None
            session.last_seen_at = now
            session.expires_at = now + self.timeout
            self.save()
            return session

    def invalidate(self, token: str) -> None:
        """Remove ``token`` from the store. No-op if unknown."""
        with self.lock:
            self.sessions.pop(token, None)
            self.save()

    def append_turn(self, token: str, turn: Turn) -> None:
        """Append ``turn`` to the session's history.

        Raises:
            AuthenticationError: If the token is invalid or expired.

        """
        with self.lock:
            session = self.resolve(token)
            if session is None:
                raise AuthenticationError("Invalid session")
            session.history.append(turn)
            self.save()

    def load_turns(self, token: str) -> list[Turn]:
        """Return the full history for ``token``."""
        session = self.resolve(token)
        return list(session.history) if session else []

    def clear_turns(self, token: str) -> None:
        """Empty the session's conversation history.

        Raises:
            AuthenticationError: If the token is invalid or expired.

        """
        with self.lock:
            session = self.resolve(token)
            if session is None:
                raise AuthenticationError("Invalid session")
            session.history.clear()
            self.save()


class Sessions:
    """SQLite-backed session store with sliding-window inactivity expiry.

    Mirrors :class:`JsonSessions` but persists to a SQLite table, which
    is safer for multi-process deployments. Use the :meth:`json`
    factory to get the JSON-backed equivalent.
    """

    def __init__(
        self,
        db_path: str | Path,
        timeout_seconds: int = DEFAULT_SESSION_TIMEOUT_SECONDS,
        db: Database | None = None,
    ) -> None:
        """Initialise the store."""
        self.db_path = str(db_path)
        self.timeout = timedelta(seconds=timeout_seconds)
        self.db = db

    @classmethod
    def json(
        cls,
        path: Path,
        timeout_seconds: int = DEFAULT_SESSION_TIMEOUT_SECONDS,
    ) -> JsonSessions:
        """Construct a JSON-backed session store."""
        return JsonSessions(path, timeout_seconds)

    async def conn(self) -> Any:
        """Return a usable connection (shared or fresh)."""
        try:
            import aiosqlite
        except ImportError:
            raise MissingDepError(
                "aiosqlite",
                "pip install raghub[auth]",
            ) from None
        if self.db is not None:
            return self.db.connection
        conn = await aiosqlite.connect(self.db_path)
        conn.row_factory = aiosqlite.Row
        return conn

    async def maybe_commit_close(self, conn: Any) -> None:
        """Commit and close ``conn`` unless we share a manager."""
        if self.db is None:
            await conn.commit()
            await conn.close()

    async def initialize(self) -> None:
        """Create the ``sessions`` table and add the ``overrides`` column when absent."""
        conn = await self.conn()
        await conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS sessions (
                session_id TEXT PRIMARY KEY,
                user_id TEXT NOT NULL,
                token TEXT NOT NULL,
                created_at TEXT NOT NULL,
                expires_at TEXT NOT NULL,
                last_seen_at TEXT NOT NULL,
                history TEXT DEFAULT '[]'
            );
            """
        )
        cursor = await conn.execute("PRAGMA table_info(sessions)")
        columns = await cursor.fetchall()
        if not any(column[1] == "overrides" for column in columns):
            await conn.execute("ALTER TABLE sessions ADD COLUMN overrides TEXT DEFAULT '{}'")
        if self.db is None:
            await conn.commit()
            await conn.close()

    async def create_session_record(self, session: Session) -> None:
        """Insert a full :class:`Session` including its history."""
        session.verify()
        conn = await self.conn()
        await conn.execute(
            """
            INSERT INTO sessions (
                session_id, user_id, token,
                created_at, expires_at, last_seen_at,
                history, overrides
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                session.id,
                session.user_id,
                session.token,
                session.created_at.isoformat(),
                session.expires_at.isoformat(),
                session.last_seen_at.isoformat(),
                json.dumps([t.model_dump(mode="json") for t in session.history]),
                serialize_overrides(session.overrides),
            ),
        )
        await self.maybe_commit_close(conn)

    async def create_session(self, user_id: str) -> Session:
        """Create and persist a new session."""
        now = datetime.now(UTC)
        session = Session(
            id=str(uuid4()),
            user_id=user_id,
            token=str(uuid4()),
            created_at=now,
            expires_at=now + self.timeout,
            last_seen_at=now,
        )
        conn = await self.conn()
        await conn.execute(
            """
            INSERT INTO sessions
                (session_id, user_id, token, created_at,
                 expires_at, last_seen_at, history, overrides)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                session.id,
                session.user_id,
                session.token,
                session.created_at.isoformat(),
                session.expires_at.isoformat(),
                session.last_seen_at.isoformat(),
                json.dumps([]),
                serialize_overrides(session.overrides),
            ),
        )
        await self.maybe_commit_close(conn)
        return session

    async def get_session(self, session_id: str) -> Session | None:
        """Look up a session by primary key."""
        conn = await self.conn()
        cursor = await conn.execute("SELECT * FROM sessions WHERE session_id = ?", (session_id,))
        row = await cursor.fetchone()
        await self.maybe_commit_close(conn)
        if row is None:
            return None
        return self.as_session(row)

    async def get_by_token(self, token: str) -> Session | None:
        """Look up a session by bearer token, with sliding expiry."""
        conn = await self.conn()
        cursor = await conn.execute("SELECT * FROM sessions WHERE token = ?", (token,))
        row = await cursor.fetchone()
        if row is None:
            await self.maybe_commit_close(conn)
            return None
        session = self.as_session(row)
        now = datetime.now(UTC)
        if now > session.expires_at:
            await conn.execute(
                "DELETE FROM sessions WHERE token = ?",
                (token,),
            )
            await self.maybe_commit_close(conn)
            return None
        session.last_seen_at = now
        session.expires_at = now + self.timeout
        await conn.execute(
            """
            UPDATE sessions
            SET last_seen_at = ?, expires_at = ?, history = ?, overrides = ?
            WHERE token = ?
            """,
            (
                session.last_seen_at.isoformat(),
                session.expires_at.isoformat(),
                json.dumps([t.model_dump(mode="json") for t in session.history]),
                serialize_overrides(session.overrides),
                session.token,
            ),
        )
        await self.maybe_commit_close(conn)
        return session

    async def update_session(self, session: Session) -> None:
        """Overwrite a session row with the supplied record."""
        session.verify()
        conn = await self.conn()
        await conn.execute(
            """
            UPDATE sessions
            SET user_id = ?, token = ?, created_at = ?, expires_at = ?,
                last_seen_at = ?, history = ?, overrides = ?
            WHERE session_id = ?
            """,
            (
                session.user_id,
                session.token,
                session.created_at.isoformat(),
                session.expires_at.isoformat(),
                session.last_seen_at.isoformat(),
                json.dumps([t.model_dump(mode="json") for t in session.history]),
                serialize_overrides(session.overrides),
                session.id,
            ),
        )
        await self.maybe_commit_close(conn)

    async def delete_session(self, session_id: str) -> None:
        """Delete a session by primary key. No-op if unknown."""
        conn = await self.conn()
        await conn.execute("DELETE FROM sessions WHERE session_id = ?", (session_id,))
        await self.maybe_commit_close(conn)

    async def get_overrides(self, session_id: str) -> dict[str, Any]:
        """Return the session's ``overrides`` mapping, or ``{}``."""
        conn = await self.conn()
        cursor = await conn.execute(
            "SELECT overrides FROM sessions WHERE session_id = ?", (session_id,)
        )
        row = await cursor.fetchone()
        await self.maybe_commit_close(conn)
        if row is None:
            return {}
        raw = row[0] if not __keyed(row) else row["overrides"]
        if not raw:
            return {}
        decoded = json.loads(raw)
        return decoded if isinstance(decoded, dict) else {}

    async def set_overrides(self, session_id: str, overrides: dict[str, Any]) -> None:
        """Replace the session's ``overrides`` mapping. No-op if unknown."""
        conn = await self.conn()
        cursor = await conn.execute(
            "SELECT 1 FROM sessions WHERE session_id = ?", (session_id,)
        )
        row = await cursor.fetchone()
        if row is None:
            await self.maybe_commit_close(conn)
            return
        await conn.execute(
            "UPDATE sessions SET overrides = ? WHERE session_id = ?",
            (serialize_overrides(overrides), session_id),
        )
        await self.maybe_commit_close(conn)

    async def append_history(self, session_id: str, turn: Turn) -> None:
        """Append a turn to the session's history. No-op if unknown."""
        conn = await self.conn()
        cursor = await conn.execute(
            "SELECT history FROM sessions WHERE session_id = ?", (session_id,)
        )
        row = await cursor.fetchone()
        if row is None:
            await self.maybe_commit_close(conn)
            return
        history = json.loads(row["history"])
        history.append(turn.model_dump(mode="json"))
        await conn.execute(
            "UPDATE sessions SET history = ? WHERE session_id = ?",
            (json.dumps(history), session_id),
        )
        await self.maybe_commit_close(conn)

    async def get_history(self, session_id: str) -> list[Turn]:
        """Return the full history of a session."""
        conn = await self.conn()
        cursor = await conn.execute(
            "SELECT history FROM sessions WHERE session_id = ?", (session_id,)
        )
        row = await cursor.fetchone()
        await self.maybe_commit_close(conn)
        if row is None:
            return []
        history = json.loads(row["history"])
        return [Turn.model_validate(t) for t in history]

    # ------------------------------------------------------------------
    # SessionStore protocol aliases
    # ------------------------------------------------------------------

    async def create(self, user_id: str) -> Session:
        """Protocol-conformant alias for :meth:`create_session`."""
        return await self.create_session(user_id)

    async def resolve(self, token: str) -> Session | None:
        """Protocol-conformant alias for :meth:`get_by_token`."""
        return await self.get_by_token(token)

    async def invalidate(self, token: str) -> None:
        """Protocol-conformant alias — deletes the session for ``token``."""
        session = await self.get_by_token(token)
        if session is not None:
            await self.delete_session(session.id)

    @staticmethod
    def as_session(row: Any) -> Session:
        """Hydrate a :class:`Session` from a SQLite row."""
        row_keys = row.keys() if hasattr(row, "keys") else row
        history_raw = row["history"] if "history" in row_keys else "[]"
        overrides_raw = row["overrides"] if "overrides" in row_keys else "{}"
        history = json.loads(history_raw) if history_raw else []
        overrides = json.loads(overrides_raw) if overrides_raw else {}
        return Session(
            id=row["session_id"],
            user_id=row["user_id"],
            token=row["token"],
            created_at=datetime.fromisoformat(row["created_at"]),
            expires_at=datetime.fromisoformat(row["expires_at"]),
            last_seen_at=datetime.fromisoformat(row["last_seen_at"]),
            history=[Turn.model_validate(t) for t in history],
            overrides=overrides if isinstance(overrides, dict) else {},
        )
