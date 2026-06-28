"""Shared database connection manager with WAL mode."""

from __future__ import annotations

from pathlib import Path

import aiosqlite


class DatabaseManager:
    """Manages a shared aiosqlite connection with WAL mode."""

    def __init__(self, db_path: str | Path) -> None:
        self.db_path = str(db_path)
        self.conn: aiosqlite.Connection | None = None

    async def connect(self) -> aiosqlite.Connection:
        """Open (or reuse) a connection with WAL mode enabled."""
        if self.conn is None:
            self.conn = await aiosqlite.connect(self.db_path)
            self.conn.row_factory = aiosqlite.Row
            await self.conn.execute("PRAGMA journal_mode=WAL")
            await self.conn.execute("PRAGMA synchronous=NORMAL")
            await self.conn.execute("PRAGMA foreign_keys=ON")
        return self.conn

    async def close(self) -> None:
        if self.conn is not None:
            await self.conn.close()
            self.conn = None

    @property
    def connection(self) -> aiosqlite.Connection:
        if self.conn is None:
            raise RuntimeError("Database not connected. Call connect() first.")
        return self.conn
