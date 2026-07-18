"""SQLite-backed session store with sliding-window inactivity expiry.

This is the production-grade equivalent of
:class:`raghub.storage.session_store.JsonSessionStore`. It mirrors the
JSON store's behaviour (sliding-window expiry, lazy eviction) but
persists to a SQLite table, which is safer for multi-process
deployments because each process holds its own connection rather than
coordinating on a single JSON file.

The store can optionally share a :class:`DatabaseManager` (and therefore
a single underlying :class:`aiosqlite.Connection`) with the rest of the
application; when no manager is supplied it opens and closes its own
connection per call. The shared-manager path is the default in
production via :class:`raghub.repositories.UnitOfWork`.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path
from uuid import uuid4

import aiosqlite

from raghub.models import ConversationTurn, SessionRecord
from raghub.storage.database import DatabaseManager


class SqliteSessionStore:
    """Async CRUD for the ``sessions`` table.

    Attributes:
        db_path: SQLite database file path.
        timeout: Sliding inactivity window as a :class:`timedelta`.
        db_manager: Optional shared :class:`DatabaseManager`. When
            ``None`` the store opens its own connections per call.
    """

    def __init__(
        self,
        db_path: str | Path,
        timeout_seconds: int = 3600,
        db_manager: DatabaseManager | None = None,
    ) -> None:
        """Initialise the store.

        Args:
            db_path: SQLite database file path.
            timeout_seconds: Inactivity expiry window in seconds.
            db_manager: Optional shared connection manager.
        """
        self.db_path = str(db_path)
        self.timeout = timedelta(seconds=timeout_seconds)
        self.db_manager = db_manager

    async def conn(self) -> aiosqlite.Connection:
        """Return a usable connection.

        When a shared :class:`DatabaseManager` is in use we borrow its
        connection; otherwise we open a fresh connection with the row
        factory set.

        Returns:
            An :class:`aiosqlite.Connection`.
        """
        if self.db_manager is not None:
            return self.db_manager.connection
        conn = await aiosqlite.connect(self.db_path)
        # ``Row`` enables attribute-style access; we cast to dict
        # before model validation to keep the code path simple.
        conn.row_factory = aiosqlite.Row
        return conn

    async def maybe_commit_close(self, conn: aiosqlite.Connection) -> None:
        """Commit and close ``conn`` unless we share a manager.

        Args:
            conn: The connection to release.
        """
        if self.db_manager is None:
            await conn.commit()
            await conn.close()

    async def initialize(self) -> None:
        """Create the ``sessions`` table if it does not exist."""
        conn = await self.conn()
        await conn.executescript("""
            CREATE TABLE IF NOT EXISTS sessions (
                session_id TEXT PRIMARY KEY,
                user_id TEXT NOT NULL,
                token TEXT NOT NULL,
                created_at TEXT NOT NULL,
                expires_at TEXT NOT NULL,
                last_seen_at TEXT NOT NULL,
                history TEXT DEFAULT '[]'
            );
        """)
        if self.db_manager is None:
            await conn.commit()
            await conn.close()

    async def create_session(self, user_id: str) -> SessionRecord:
        """Create and persist a new session.

        Args:
            user_id: The owning user's id.

        Returns:
            The freshly created :class:`SessionRecord`.
        """
        now = datetime.now(UTC)
        session = SessionRecord(
            session_id=str(uuid4()),
            user_id=user_id,
            token=str(uuid4()),
            created_at=now,
            expires_at=now + self.timeout,
            last_seen_at=now,
        )
        conn = await self.conn()
        await conn.execute(
            """
            INSERT INTO sessions (session_id, user_id, token, created_at, expires_at, last_seen_at, history)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                session.session_id,
                session.user_id,
                session.token,
                session.created_at.isoformat(),
                session.expires_at.isoformat(),
                session.last_seen_at.isoformat(),
                json.dumps([]),
            ),
        )
        await self.maybe_commit_close(conn)
        return session

    async def get_session(self, session_id: str) -> SessionRecord | None:
        """Look up a session by primary key.

        Args:
            session_id: The session id (not the bearer token).

        Returns:
            The :class:`SessionRecord`, or ``None`` if not found.
        """
        conn = await self.conn()
        cursor = await conn.execute("SELECT * FROM sessions WHERE session_id = ?", (session_id,))
        row = await cursor.fetchone()
        await self.maybe_commit_close(conn)
        if row is None:
            return None
        return self.row_to_session(row)

    async def get_by_token(self, token: str) -> SessionRecord | None:
        """Look up a session by bearer token, with sliding expiry.

        Args:
            token: The bearer token presented by the client.

        Returns:
            The live :class:`SessionRecord` (with the expiry window
            refreshed), or ``None`` if the token is unknown or expired.
            Expired sessions are deleted as a side effect.
        """
        conn = await self.conn()
        cursor = await conn.execute("SELECT * FROM sessions WHERE token = ?", (token,))
        row = await cursor.fetchone()
        if row is None:
            await self.maybe_commit_close(conn)
            return None
        session = self.row_to_session(row)
        now = datetime.now(UTC)
        if now > session.expires_at:
            # Lazy eviction: delete and report missing. The deletion
            # keeps the table from accumulating dead rows.
            await conn.execute("DELETE FROM sessions WHERE session_id = ?", (session.session_id,))
            await self.maybe_commit_close(conn)
            return None
        # Sliding expiry: refresh both timestamps so the next call
        # starts from ``now`` rather than the prior visit.
        session.last_seen_at = now
        session.expires_at = now + self.timeout
        await conn.execute(
            """
            UPDATE sessions
            SET last_seen_at = ?, expires_at = ?, history = ?
            WHERE session_id = ?
            """,
            (
                session.last_seen_at.isoformat(),
                session.expires_at.isoformat(),
                json.dumps([t.model_dump(mode="json") for t in session.history]),
                session.session_id,
            ),
        )
        await self.maybe_commit_close(conn)
        return session

    async def update_session(self, session: SessionRecord) -> None:
        """Overwrite a session row with the supplied record.

        Args:
            session: The :class:`SessionRecord` to persist.
        """
        conn = await self.conn()
        await conn.execute(
            """
            UPDATE sessions
            SET user_id = ?, token = ?, created_at = ?, expires_at = ?,
                last_seen_at = ?, history = ?
            WHERE session_id = ?
            """,
            (
                session.user_id,
                session.token,
                session.created_at.isoformat(),
                session.expires_at.isoformat(),
                session.last_seen_at.isoformat(),
                json.dumps([t.model_dump(mode="json") for t in session.history]),
                session.session_id,
            ),
        )
        await self.maybe_commit_close(conn)

    async def delete_session(self, session_id: str) -> None:
        """Delete a session by primary key.

        Args:
            session_id: The session id. No-op if unknown.
        """
        conn = await self.conn()
        await conn.execute("DELETE FROM sessions WHERE session_id = ?", (session_id,))
        await self.maybe_commit_close(conn)

    async def append_history(self, session_id: str, turn: ConversationTurn) -> None:
        """Append a turn to a session's history.

        Args:
            session_id: The session id.
            turn: The :class:`ConversationTurn` to append.

        Note:
            No-op if the session does not exist (we never raise here
            because the typical caller is the conversation manager,
            which would rather lose a turn than crash the request).
        """
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

    async def get_history(self, session_id: str) -> list[ConversationTurn]:
        """Return the full history of a session.

        Args:
            session_id: The session id.

        Returns:
            A list of :class:`ConversationTurn`. Empty when the session
            is unknown.
        """
        conn = await self.conn()
        cursor = await conn.execute(
            "SELECT history FROM sessions WHERE session_id = ?", (session_id,)
        )
        row = await cursor.fetchone()
        await self.maybe_commit_close(conn)
        if row is None:
            return []
        history = json.loads(row["history"])
        return [ConversationTurn.model_validate(t) for t in history]

    # ------------------------------------------------------------------
    # SessionStore protocol conformance
    # ------------------------------------------------------------------

    async def create(self, user_id: str) -> SessionRecord:
        """Protocol-conformant alias for :meth:`create_session`."""
        return await self.create_session(user_id)

    async def resolve(self, token: str) -> SessionRecord | None:
        """Protocol-conformant alias for :meth:`get_by_token`."""
        return await self.get_by_token(token)

    async def invalidate(self, token: str) -> None:
        """Protocol-conformant alias — deletes the session for ``token``."""
        session = await self.get_by_token(token)
        if session is not None:
            await self.delete_session(session.session_id)

    def row_to_session(self, row: aiosqlite.Row) -> SessionRecord:
        """Hydrate a :class:`SessionRecord` from a SQLite row.

        Args:
            row: An :class:`aiosqlite.Row`.

        Returns:
            The fully-typed :class:`SessionRecord`.
        """
        return SessionRecord(
            session_id=row["session_id"],
            user_id=row["user_id"],
            token=row["token"],
            created_at=datetime.fromisoformat(row["created_at"]),
            expires_at=datetime.fromisoformat(row["expires_at"]),
            last_seen_at=datetime.fromisoformat(row["last_seen_at"]),
            history=[ConversationTurn.model_validate(t) for t in json.loads(row["history"])],
        )
