"""Shared :class:`aiosqlite` connection manager with WAL mode enabled.

The :class:`DatabaseManager` centralises a single aiosqlite connection
that multiple stores can borrow. This avoids the overhead of opening
a new connection per call (file-lock acquisition, page-cache warm-up)
and lets SQLite's WAL mode provide non-blocking reads alongside a
single writer.

Configuration applied on connect:

* ``isolation_level=None`` — autocommit. Stores no longer need to
  remember to call ``commit`` after every write; a plain ``INSERT``
  persists immediately. Explicit ``BEGIN``/``COMMIT`` still works for
  multi-statement :class:`UnitOfWork` transactions.
* ``PRAGMA journal_mode=WAL`` — concurrent readers with one writer.
* ``PRAGMA synchronous=NORMAL`` — durability with one fewer fsync
  per commit; acceptable for application-level WAL files.
* ``PRAGMA foreign_keys=ON`` — SQLite needs this per-connection; the
  default in :mod:`aiosqlite` is OFF.

Use a single instance per database file per process. Sharing the
connection across threads is safe with aiosqlite because the underlying
driver serialises access internally.
"""

from __future__ import annotations

import contextlib
from pathlib import Path

import aiosqlite


class DatabaseManager:
    """Manages a shared :class:`aiosqlite.Connection` with WAL mode."""

    def __init__(self, db_path: str | Path) -> None:
        self.db_path = str(db_path)
        self.conn: aiosqlite.Connection | None = None

    async def connect(self) -> aiosqlite.Connection:
        if self.conn is None:
            # isolation_level=None turns on autocommit so plain
            # INSERT/UPDATE/DELETE persist without an explicit commit;
            # BEGIN/COMMIT still drive UnitOfWork transactions.
            self.conn = await aiosqlite.connect(self.db_path, isolation_level=None)
            self.conn.row_factory = aiosqlite.Row
            await self.conn.execute("PRAGMA journal_mode=WAL")
            await self.conn.execute("PRAGMA synchronous=NORMAL")
            await self.conn.execute("PRAGMA foreign_keys=ON")
        return self.conn

    async def close(self) -> None:
        if self.conn is not None:
            conn = self.conn
            try:
                # ponytail: best-effort WAL checkpoint before close pushes any
                # straggling -wal frames back into the main file so a subsequent
                # cold start sees a clean database. Silently ignored on
                # in-memory or already-checkpointed DBs.
                with contextlib.suppress(Exception):
                    await conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
                await conn.close()
            finally:
                self.conn = None

    @property
    def connection(self) -> aiosqlite.Connection:
        if self.conn is None:
            raise RuntimeError("Database not connected. Call connect() first.")
        return self.conn
