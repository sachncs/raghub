from __future__ import annotations

from pathlib import Path

import pytest

from raghub.storage.database import DatabaseManager


class TestDatabaseManager:
    def test_connection_property_before_connect_raises(self) -> None:
        mgr = DatabaseManager(":memory:")
        with pytest.raises(RuntimeError, match="Database not connected"):
            _ = mgr.connection

    async def test_connect_and_close(self) -> None:
        mgr = DatabaseManager(":memory:")
        conn = await mgr.connect()
        assert conn is mgr.connection
        await mgr.close()
        assert mgr.conn is None

    async def test_double_connect_reuses_connection(self) -> None:
        mgr = DatabaseManager(":memory:")
        c1 = await mgr.connect()
        c2 = await mgr.connect()
        assert c1 is c2
        await mgr.close()

    async def test_close_is_idempotent(self) -> None:
        mgr = DatabaseManager(":memory:")
        await mgr.connect()
        await mgr.close()
        await mgr.close()
        assert mgr.conn is None

    async def test_close_without_connect_is_noop(self) -> None:
        mgr = DatabaseManager(":memory:")
        await mgr.close()
        assert mgr.conn is None

    async def test_reopen_after_close(self, tmp_path: Path) -> None:
        db_path = tmp_path / "reopen.db"
        mgr = DatabaseManager(db_path)
        await mgr.connect()
        await mgr.conn.execute("CREATE TABLE t (x INTEGER)")
        await mgr.conn.execute("INSERT INTO t VALUES (1)")
        await mgr.close()

        mgr2 = DatabaseManager(db_path)
        try:
            await mgr2.connect()
            cursor = await mgr2.conn.execute("SELECT x FROM t")
            row = await cursor.fetchone()
            assert row is not None
            assert row[0] == 1
        finally:
            await mgr2.close()

    async def test_writes_persist_without_explicit_commit(self, tmp_path: Path) -> None:
        # With autocommit (isolation_level=None) callers no longer need
        # to remember to commit after every statement; durability is
        # verified by closing and re-opening the DB.
        db_path = tmp_path / "autocommit.db"
        mgr = DatabaseManager(db_path)
        await mgr.connect()
        await mgr.conn.execute("CREATE TABLE t (x INTEGER)")
        await mgr.conn.execute("INSERT INTO t VALUES (42)")
        await mgr.close()

        mgr2 = DatabaseManager(db_path)
        try:
            await mgr2.connect()
            cursor = await mgr2.conn.execute("SELECT x FROM t")
            row = await cursor.fetchone()
            assert row is not None and row[0] == 42
        finally:
            await mgr2.close()

    async def test_close_checkpoints_wal(self, tmp_path: Path) -> None:
        # Touching the manager with WAL writes leaves a -wal sidecar.
        # After close() the sidecar should be merged back so the
        # -wal file no longer exists (or is empty).
        db_path = tmp_path / "wal.db"
        mgr = DatabaseManager(db_path)
        await mgr.connect()
        await mgr.conn.execute("CREATE TABLE t (x INTEGER)")
        await mgr.conn.execute("INSERT INTO t VALUES (1)")
        await mgr.close()

        wal_file = db_path.with_suffix(db_path.suffix + "-wal")
        # Either the file is gone (fully checkpointed) or present but
        # empty — both are acceptable outcomes.
        assert not wal_file.exists() or wal_file.stat().st_size == 0
