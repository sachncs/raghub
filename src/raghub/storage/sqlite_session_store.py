from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from uuid import uuid4

import aiosqlite

from raghub.models import ConversationTurn, SessionRecord


class SqliteSessionStore:
    def __init__(self, db_path: str | Path, timeout_seconds: int = 3600) -> None:
        self.db_path = str(db_path)
        self.timeout = timedelta(seconds=timeout_seconds)

    async def initialize(self) -> None:
        async with aiosqlite.connect(self.db_path) as db:
            await db.executescript("""
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
            await db.commit()

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
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute(
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
            await db.commit()
        return session

    async def get_session(self, session_id: str) -> SessionRecord | None:
        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            cursor = await db.execute(
                "SELECT * FROM sessions WHERE session_id = ?", (session_id,)
            )
            row = await cursor.fetchone()
            if row is None:
                return None
            return self._row_to_session(row)

    async def get_by_token(self, token: str) -> SessionRecord | None:
        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            cursor = await db.execute(
                "SELECT * FROM sessions WHERE token = ?", (token,)
            )
            row = await cursor.fetchone()
            if row is None:
                return None
            session = self._row_to_session(row)
            now = datetime.now(timezone.utc)
            if now > session.expires_at:
                await self.delete_session(session.session_id)
                return None
            session.last_seen_at = now
            session.expires_at = now + self.timeout
            await self.update_session(session)
            return session

    async def update_session(self, session: SessionRecord) -> None:
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute(
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
            await db.commit()

    async def delete_session(self, session_id: str) -> None:
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute(
                "DELETE FROM sessions WHERE session_id = ?", (session_id,)
            )
            await db.commit()

    async def append_history(self, session_id: str, turn: ConversationTurn) -> None:
        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            cursor = await db.execute(
                "SELECT history FROM sessions WHERE session_id = ?", (session_id,)
            )
            row = await cursor.fetchone()
            if row is None:
                return
            history = json.loads(row["history"])
            history.append(turn.model_dump(mode="json"))
            await db.execute(
                "UPDATE sessions SET history = ? WHERE session_id = ?",
                (json.dumps(history), session_id),
            )
            await db.commit()

    async def get_history(self, session_id: str) -> list[ConversationTurn]:
        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            cursor = await db.execute(
                "SELECT history FROM sessions WHERE session_id = ?", (session_id,)
            )
            row = await cursor.fetchone()
            if row is None:
                return []
            history = json.loads(row["history"])
            return [ConversationTurn.model_validate(t) for t in history]

    def _row_to_session(self, row: aiosqlite.Row) -> SessionRecord:
        return SessionRecord(
            session_id=row["session_id"],
            user_id=row["user_id"],
            token=row["token"],
            created_at=datetime.fromisoformat(row["created_at"]),
            expires_at=datetime.fromisoformat(row["expires_at"]),
            last_seen_at=datetime.fromisoformat(row["last_seen_at"]),
            history=[ConversationTurn.model_validate(t) for t in json.loads(row["history"])],
        )
