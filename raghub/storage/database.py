"""Shared :class:`aiosqlite` connection manager with WAL mode enabled.

The :class:`DatabaseManager` centralises a single aiosqlite connection
that multiple stores can borrow. This avoids the overhead of opening
a new connection per call (file-lock acquisition, page-cache warm-up)
and lets SQLite's WAL mode provide non-blocking reads alongside a
single writer.

Configuration applied on connect:

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

from pathlib import Path

import aiosqlite


class DatabaseManager:
    """Manages a shared :class:`aiosqlite.Connection` with WAL mode."""

    def __init__(self, db_path: str | Path) -> None:
        """Initialise the manager.

        Args:
            db_path: Path to the SQLite database file. Created on first
                :meth:`connect` if it does not exist.
        """
        self.db_path = str(db_path)
        self.conn: aiosqlite.Connection | None = None

    async def connect(self) -> aiosqlite.Connection:
        """Open (or reuse) the connection, applying required PRAGMAs.

        Returns:
            The live :class:`aiosqlite.Connection`.
        """
        if self.conn is None:
            self.conn = await aiosqlite.connect(self.db_path)
            self.conn.row_factory = aiosqlite.Row
            # WAL mode + normal sync + foreign keys: the standard
            # combo for application-level SQLite use.
            await self.conn.execute("PRAGMA journal_mode=WAL")
            await self.conn.execute("PRAGMA synchronous=NORMAL")
            await self.conn.execute("PRAGMA foreign_keys=ON")
        return self.conn

    async def close(self) -> None:
        """Close the underlying connection.

        No-op when the manager has not been connected. After calling
        this, :meth:`connect` will re-open the connection.
        """
        if self.conn is not None:
            await self.conn.close()
            self.conn = None

    @property
    def connection(self) -> aiosqlite.Connection:
        """Return the active connection.

        Returns:
            The :class:`aiosqlite.Connection`.

        Raises:
            RuntimeError: If :meth:`connect` has not been called.
        """
        if self.conn is None:
            raise RuntimeError("Database not connected. Call connect() first.")
        return self.conn