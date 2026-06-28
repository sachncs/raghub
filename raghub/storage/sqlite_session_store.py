from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from uuid import uuid4

import aiosqlite

from raghub.models import ConversationTurn, SessionRecord
from raghub.storage.database import DatabaseManager


class SqliteSessionStore:
    def __init__(self, db_path: str | Path, timeout_seconds: int = 3600,
                 db_manager: DatabaseManager | None = None) -> None:
        self.db_path = str(db_path)
        self.timeout = timedelta(seconds=timeout_seconds)
        self.db_manager = db_manager

    async def conn(self) -> aiosqlite.Connection:
        if self.db_manager is not None:
            return self.db_manager.connection
        conn = await aiosqlite.connect(self.db_path)
        conn.row_factory = aiosqlite.Row
        return conn

    async def maybe_commit_close(self, conn: aiosqlite.Connection) -> None:
        if self.db_manager is None:
            await conn.commit()
            await conn.close()

    async def initialize(self) -> None:
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
        now = datetime.now(timezone.utc)
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
        conn = await self.conn()
        cursor = await conn.execute(
            "SELECT * FROM sessions WHERE session_id = ?", (session_id,)
        )
        row = await cursor.fetchone()
        await self.maybe_commit_close(conn)
        if row is None:
            return None
        return self.row_to_session(row)

    async def get_by_token(self, token: str) -> SessionRecord | None:
        conn = await self.conn()
        cursor = await conn.execute(
            "SELECT * FROM sessions WHERE token = ?", (token,)
        )
        row = await cursor.fetchone()
        if row is None:
            await self.maybe_commit_close(conn)
            return None
        session = self.row_to_session(row)
        now = datetime.now(timezone.utc)
        if now > session.expires_at:
            await conn.execute(
                "DELETE FROM sessions WHERE session_id = ?", (session.session_id,)
            )
            await self.maybe_commit_close(conn)
            return None
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
        conn = await self.conn()
        await conn.execute(
            "DELETE FROM sessions WHERE session_id = ?", (session_id,)
        )
        await self.maybe_commit_close(conn)

    async def append_history(self, session_id: str, turn: ConversationTurn) -> None:
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

    def row_to_session(self, row: aiosqlite.Row) -> SessionRecord:
        return SessionRecord(
            session_id=row["session_id"],
            user_id=row["user_id"],
            token=row["token"],
            created_at=datetime.fromisoformat(row["created_at"]),
            expires_at=datetime.fromisoformat(row["expires_at"]),
            last_seen_at=datetime.fromisoformat(row["last_seen_at"]),
            history=[ConversationTurn.model_validate(t) for t in json.loads(row["history"])],
        )
