"""Qualitative tests for the :class:`Database` wrapper.

These tests cover the real contract:

* The connection is opened lazily on first ``connect()`` and is
  reusable for subsequent calls.
* ``close()`` is idempotent — calling it twice is a no-op.
* ``close()`` without a prior ``connect()`` is also a no-op.
* Writes persist without an explicit commit (autocommit mode).
* Re-opening a database after close preserves the data.
* The WAL file is checkpointed back to the main database on close.
* Accessing ``connection`` before ``connect()`` raises a clear
  ``RuntimeError`` — not a vague ``AttributeError``.
"""

from __future__ import annotations

from pathlib import Path

import aiosqlite
import pytest

from raghub.stores import Database


class TestDatabaseManagerContract:
    def test_connection_property_before_connect_raises(self) -> None:
        mgr = Database(":memory:")
        with pytest.raises(RuntimeError, match="Database not connected"):
            _ = mgr.connection

    async def test_connect_and_close(self) -> None:
        mgr = Database(":memory:")
        conn = await mgr.connect()
        assert conn is mgr.connection
        await mgr.close()
        assert mgr.conn is None

    async def test_double_connect_reuses_connection(self) -> None:
        """Two ``connect()`` calls return the same connection — opening
        a new connection every time would leak file handles."""
        mgr = Database(":memory:")
        c1 = await mgr.connect()
        c2 = await mgr.connect()
        assert c1 is c2
        await mgr.close()

    async def test_close_is_idempotent(self) -> None:
        """Calling ``close()`` repeatedly is a no-op (defensive
        shutdown from caller code paths)."""
        mgr = Database(":memory:")
        await mgr.connect()
        await mgr.close()
        await mgr.close()
        assert mgr.conn is None

    async def test_close_without_connect_is_noop(self) -> None:
        """A manager that never connected can be closed without error."""
        mgr = Database(":memory:")
        await mgr.close()
        assert mgr.conn is None


class TestDatabaseManagerDurability:
    async def test_reopen_after_close(self, tmp_path: Path) -> None:
        """The on-disk file survives a close + re-open cycle."""
        db_path = tmp_path / "reopen.db"
        mgr = Database(db_path)
        await mgr.connect()
        await mgr.conn.execute("CREATE TABLE t (x INTEGER)")
        await mgr.conn.execute("INSERT INTO t VALUES (1)")
        await mgr.close()

        mgr2 = Database(db_path)
        try:
            await mgr2.connect()
            cursor = await mgr2.conn.execute("SELECT x FROM t")
            row = await cursor.fetchone()
            assert row is not None
            assert row[0] == 1
        finally:
            await mgr2.close()

    async def test_writes_persist_without_explicit_commit(self, tmp_path: Path) -> None:
        """Autocommit mode: callers do not need to remember to commit."""
        db_path = tmp_path / "autocommit.db"
        mgr = Database(db_path)
        await mgr.connect()
        await mgr.conn.execute("CREATE TABLE t (x INTEGER)")
        await mgr.conn.execute("INSERT INTO t VALUES (42)")
        await mgr.close()

        mgr2 = Database(db_path)
        try:
            await mgr2.connect()
            cursor = await mgr2.conn.execute("SELECT x FROM t")
            row = await cursor.fetchone()
            assert row is not None and row[0] == 42
        finally:
            await mgr2.close()

    async def test_wal_checkpointed_on_close(self, tmp_path: Path) -> None:
        """After ``close()`` the WAL sidecar is either gone or empty —
        the main database is the source of truth on next open."""
        db_path = tmp_path / "wal.db"
        mgr = Database(db_path)
        await mgr.connect()
        await mgr.conn.execute("CREATE TABLE t (x INTEGER)")
        await mgr.conn.execute("INSERT INTO t VALUES (1)")
        await mgr.close()

        wal_file = db_path.with_suffix(db_path.suffix + "-wal")
        assert not wal_file.exists() or wal_file.stat().st_size == 0

    async def test_many_writes_then_close_preserves_all(self, tmp_path: Path) -> None:
        """1 000 inserts survive a close + reopen — autocommit + WAL
        checkpointing must be lossless."""
        db_path = tmp_path / "bulk.db"
        mgr = Database(db_path)
        await mgr.connect()
        await mgr.conn.execute("CREATE TABLE t (x INTEGER)")
        for i in range(1000):
            await mgr.conn.execute("INSERT INTO t VALUES (?)", (i,))
        await mgr.close()

        async with aiosqlite.connect(str(db_path)) as conn:
            cursor = await conn.execute("SELECT COUNT(*) FROM t")
            row = await cursor.fetchone()
            assert row is not None
            assert row[0] == 1000

    async def test_concurrent_connects_to_same_path(self, tmp_path: Path) -> None:
        """Two managers opened against the same file see the same data."""
        db_path = tmp_path / "shared.db"
        mgr1 = Database(db_path)
        await mgr1.connect()
        await mgr1.conn.execute("CREATE TABLE t (x INTEGER)")
        await mgr1.conn.execute("INSERT INTO t VALUES (1)")
        await mgr1.close()

        mgr2 = Database(db_path)
        try:
            await mgr2.connect()
            cursor = await mgr2.conn.execute("SELECT x FROM t")
            row = await cursor.fetchone()
            assert row is not None and row[0] == 1
        finally:
            await mgr2.close()
