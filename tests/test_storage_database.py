from __future__ import annotations

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
