from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

import aiosqlite
import bcrypt
from pydantic import BaseModel, Field


class UserRecord(BaseModel):
    user_id: str = Field(default_factory=lambda: str(uuid4()))
    email: str
    password_hash: str
    allowed_companies: list[str] = Field(default_factory=list)
    allowed_groups: list[str] = Field(default_factory=list)
    is_admin: bool = False
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class SqliteUserStore:
    def __init__(self, db_path: str | Path) -> None:
        self.db_path = str(db_path)

    async def initialize(self) -> None:
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
            """)
            await db.commit()

    async def create_user(self, email: str, password: str,
                          companies: list[str] | None = None,
                          is_admin: bool = False) -> UserRecord:
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
        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            cursor = await db.execute("SELECT * FROM users WHERE email = ?", (email,))
            row = await cursor.fetchone()
            if row is None:
                return None
            return self.row_to_record(row)

    async def get_by_id(self, user_id: str) -> UserRecord | None:
        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            cursor = await db.execute("SELECT * FROM users WHERE user_id = ?", (user_id,))
            row = await cursor.fetchone()
            if row is None:
                return None
            return self.row_to_record(row)

    async def verify_password(self, email: str, password: str) -> UserRecord | None:
        user = await self.get_by_email(email)
        if user is None:
            return None
        if bcrypt.checkpw(password.encode("utf-8"), user.password_hash.encode("utf-8")):
            return user
        return None

    async def list_users(self) -> list[UserRecord]:
        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            cursor = await db.execute("SELECT * FROM users ORDER BY created_at DESC")
            rows = await cursor.fetchall()
            return [self.row_to_record(row) for row in rows]

    def row_to_record(self, row: aiosqlite.Row) -> UserRecord:
        data: dict[str, Any] = dict(row)
        data["allowed_companies"] = json.loads(data.get("allowed_companies", "[]"))
        data["allowed_groups"] = json.loads(data.get("allowed_groups", "[]"))
        data["is_admin"] = bool(data["is_admin"])
        data["created_at"] = datetime.fromisoformat(data["created_at"])
        return UserRecord.model_validate(data)
