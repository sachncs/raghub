"""Tests for raghub.tenants.isolation — Tier 2 v0.9.0.

Each test is gated on optional infrastructure (Postgres + pgvector)
so the suite runs without external services.
"""

from __future__ import annotations

import os
from typing import Any

import pytest

from raghub.config import Settings
from raghub.tenants import (
    CompositeTenantResolver,
    HeaderTenantResolver,
    JwtClaimTenantResolver,
    validate_tenant,
)
from raghub.tenants.isolation import (
    DatabasePerTenant,
    Isolation,
    RowLevel,
    SchemaPerTenant,
    TenantContext,
    TenantRegistry,
    current,
    require_tenant,
    reset,
    set_current,
)

# ---------------------------------------------------------------------------
# validate_tenant
# ---------------------------------------------------------------------------


class TestValidateTenantId:
    @pytest.mark.parametrize(
        "tenant_id",
        ["abc", "acme", "tenant-1", "tenant_2", "abc-def-ghi"],
    )
    def test_valid_tenant_ids_accepted(self, tenant_id: str) -> None:
        validate_tenant(tenant_id)

    @pytest.mark.parametrize(
        "tenant_id",
        [
            "",
            "ab",  # too short
            "1abc",  # starts with digit
            "abc.def",  # dot
            "abc def",  # space
            "ABC",  # uppercase
            "a" * 65,  # too long
            "abc!",  # punctuation
        ],
    )
    def test_invalid_tenant_ids_rejected(self, tenant_id: str) -> None:
        with pytest.raises(ValueError, match="invalid tenant id"):
            validate_tenant(tenant_id)


# ---------------------------------------------------------------------------
# TenantContext / contextvars
# ---------------------------------------------------------------------------


class TestTenantContext:
    def test_current_returns_none_by_default(self) -> None:
        token = set_current(None)
        try:
            assert current() is None
        finally:
            reset(token)

    def test_set_and_reset_tenant(self) -> None:
        ctx = TenantContext(tenant_id="acme")
        token = set_current(ctx)
        try:
            assert current() is ctx
        finally:
            reset(token)
        assert current() is None


class TestRequireTenant:
    def test_require_tenant_raises_when_no_context(self) -> None:
        from raghub.errors import AuthorizationError

        token = set_current(None)
        try:
            with pytest.raises(AuthorizationError, match="missing tenant context"):
                require_tenant()
        finally:
            reset(token)

    def test_require_tenant_returns_context_when_bound(self) -> None:
        ctx = TenantContext(tenant_id="acme")
        token = set_current(ctx)
        try:
            assert require_tenant() is ctx
        finally:
            reset(token)


# ---------------------------------------------------------------------------
# RowLevel.filter_query — Tier 2 Item 8
# ---------------------------------------------------------------------------


class TestRowLevelFilterQuery:
    def test_filter_query_returns_empty_when_no_context(self) -> None:
        token = set_current(None)
        try:
            clause, params = RowLevel().filter_query()
            assert clause == ""
            assert params == {}
        finally:
            reset(token)

    def test_filter_query_returns_clause_when_context_bound(self) -> None:
        ctx = TenantContext(tenant_id="acme")
        token = set_current(ctx)
        try:
            clause, params = RowLevel().filter_query()
            assert clause == "tenant_id = :tenant_id"
            assert params == {"tenant_id": "acme"}
        finally:
            reset(token)

    def test_filter_query_custom_column(self) -> None:
        ctx = TenantContext(tenant_id="acme")
        token = set_current(ctx)
        try:
            clause, params = RowLevel().filter_query(column="owner")
            assert clause == "owner = :tenant_id"
        finally:
            reset(token)

    def test_filter_query_custom_operator(self) -> None:
        ctx = TenantContext(tenant_id="acme")
        token = set_current(ctx)
        try:
            clause, params = RowLevel().filter_query(operator="!=")
            assert clause == "tenant_id != :tenant_id"
        finally:
            reset(token)


class TestRowLevelApplyToKwargs:
    def test_apply_to_kwargs_passes_through_when_no_context(self) -> None:
        token = set_current(None)
        try:
            kwargs = {"foo": "bar"}
            assert RowLevel().apply_to_kwargs(kwargs) is kwargs
        finally:
            reset(token)

    def test_apply_to_kwargs_injects_tenant_id(self) -> None:
        ctx = TenantContext(tenant_id="acme")
        token = set_current(ctx)
        try:
            kwargs = {"query_vector": [0.1, 0.2]}
            result = RowLevel().apply_to_kwargs(kwargs)
            assert result["tenant_id"] == "acme"
            # original is not mutated
            assert "tenant_id" not in kwargs
        finally:
            reset(token)

    def test_apply_to_kwargs_does_not_overwrite_explicit_tenant_id(self) -> None:
        ctx = TenantContext(tenant_id="acme")
        token = set_current(ctx)
        try:
            kwargs = {"tenant_id": "explicit"}
            result = RowLevel().apply_to_kwargs(kwargs)
            assert result["tenant_id"] == "explicit"
        finally:
            reset(token)


# ---------------------------------------------------------------------------
# TenantRegistry
# ---------------------------------------------------------------------------


class TestTenantRegistry:
    def test_get_unknown_raises(self) -> None:
        registry = TenantRegistry()
        with pytest.raises(KeyError, match="unknown tenant id"):
            registry.get("missing")

    def test_upsert_then_get(self) -> None:
        registry = TenantRegistry()
        registry.upsert("acme", dsn="postgres://x", vector_dim=512)
        record = registry.get("acme")
        assert record["dsn"] == "postgres://x"
        assert record["vector_dim"] == 512

    def test_remove(self) -> None:
        registry = TenantRegistry()
        registry.upsert("acme", dsn="postgres://x")
        registry.remove("acme")
        with pytest.raises(KeyError):
            registry.get("acme")


# ---------------------------------------------------------------------------
# DatabasePerTenant.connection_for — Tier 2 Item 14
# ---------------------------------------------------------------------------


class TestDatabasePerTenantRouting:
    @pytest.mark.asyncio
    async def test_connection_for_unknown_tenant_raises(self) -> None:
        registry = TenantRegistry()
        dbt = DatabasePerTenant(registry)
        token = set_current(TenantContext(tenant_id="missing"))
        try:
            with pytest.raises(KeyError, match="unknown tenant id"):
                await dbt.connection_for("missing")
        finally:
            reset(token)


# ---------------------------------------------------------------------------
# Settings.tenants wiring
# ---------------------------------------------------------------------------


class TestTenantsSettings:
    def test_default_tenants_isolation_is_row_level(self) -> None:
        settings = Settings()
        assert settings.tenants.isolation == Isolation.RowLevel

    def test_default_tenants_resolver_is_none(self) -> None:
        settings = Settings()
        assert settings.tenants.resolver == "none"


# ---------------------------------------------------------------------------
# Resolver instantiation
# ---------------------------------------------------------------------------


def test_composite_resolver_prefers_jwt_over_header() -> None:
    """CompositeTenantResolver prefers JWT claim; falls back to header."""

    class FakeRequest:
        headers = {"X-Tenant-ID": "from-header"}
        claims = {"tenant_id": "from-jwt"}

    resolver = CompositeTenantResolver()
    assert resolver.resolve(FakeRequest()) == "from-jwt"


def test_composite_resolver_falls_back_to_header() -> None:
    class FakeRequest:
        headers = {"X-Tenant-ID": "from-header"}
        claims = {}

    resolver = CompositeTenantResolver()
    assert resolver.resolve(FakeRequest()) == "from-header"


def test_jwt_resolver_returns_claim() -> None:
    class FakeRequest:
        claims = {"tenant_id": "alice"}
        headers = {}

    resolver = JwtClaimTenantResolver()
    assert resolver.resolve(FakeRequest()) == "alice"


def test_header_resolver_returns_header() -> None:
    class FakeRequest:
        headers = {"X-Tenant-ID": "bob"}
        claims = {}

    resolver = HeaderTenantResolver()
    assert resolver.resolve(FakeRequest()) == "bob"


# ---------------------------------------------------------------------------
# Gated: SchemaPerTenant (requires asyncpg + Postgres)
# ---------------------------------------------------------------------------


PG_DSN_ENV = "RAG_TEST_PGVECTOR_DSN"


@pytest.mark.skipif(
    not os.environ.get(PG_DSN_ENV),
    reason=(
        f"{PG_DSN_ENV} environment variable is not set; live Postgres test "
        "skipped. Set RAG_TEST_PGVECTOR_DSN to a Postgres DSN to enable."
    ),
)
class TestSchemaPerTenantLive:
    @pytest.mark.asyncio
    async def test_ensure_schema_creates_schema(self) -> None:
        dsn = os.environ[PG_DSN_ENV]
        # Unique tenant id per test run to avoid colliding with prior runs
        tenant_id = f"test_tenant_{os.getpid()}"
        sp = SchemaPerTenant(dsn)
        token = set_current(TenantContext(tenant_id=tenant_id))
        try:
            await sp.ensure_schema(tenant_id)
        finally:
            reset(token)


@pytest.mark.skipif(
    not os.environ.get(PG_DSN_ENV),
    reason=(
        f"{PG_DSN_ENV} environment variable is not set; live Postgres test "
        "skipped. Set RAG_TEST_PGVECTOR_DSN to a Postgres DSN to enable."
    ),
)
class TestDatabasePerTenantLive:
    @pytest.mark.asyncio
    async def test_connection_for_returns_asyncpg_connection(self) -> None:
        dsn = os.environ[PG_DSN_ENV]
        registry = TenantRegistry()
        registry.upsert("acme", dsn=dsn, vector_dim=384)
        dbt = DatabasePerTenant(registry)
        token = set_current(TenantContext(tenant_id="acme"))
        try:
            conn = await dbt.connection_for("acme")
            assert conn is not None
            await conn.close()
        finally:
            reset(token)


# ---------------------------------------------------------------------------
# Tier 2 Item 15: migrate_row_to_schema round-trip
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    not os.environ.get(PG_DSN_ENV),
    reason=(
        f"{PG_DSN_ENV} environment variable is not set; live Postgres test "
        "skipped. Set RAG_TEST_PGVECTOR_DSN to a Postgres DSN to enable."
    ),
)
class TestMigrateRowToSchema:
    """Item 15: migrate_row_to_schema round-trip test."""

    @pytest.mark.asyncio
    async def test_migrate_row_to_schema_round_trip(self) -> None:
        import asyncpg

        from raghub.tenants.isolation import migrate_row_to_schema

        dsn = os.environ[PG_DSN_ENV]
        tenant_id = f"migrate_test_{os.getpid()}"
        src_conn = await asyncpg.connect(dsn)
        dst_conn = await asyncpg.connect(dsn)
        try:
            await src_conn.execute("CREATE SCHEMA IF NOT EXISTS public")
            await src_conn.execute(
                "CREATE TABLE IF NOT EXISTS public.raghub_chunks ("
                "  chunk_id TEXT PRIMARY KEY,"
                "  content TEXT,"
                "  tenant_id TEXT"
                ")"
            )
            await dst_conn.execute("CREATE SCHEMA IF NOT EXISTS tenant_" + tenant_id)
            await dst_conn.execute(
                "CREATE TABLE IF NOT EXISTS tenant_" + tenant_id + ".raghub_chunks ("
                "  chunk_id TEXT PRIMARY KEY,"
                "  content TEXT,"
                "  tenant_id TEXT"
                ")"
            )
            for i in range(10):
                await src_conn.execute(
                    "INSERT INTO public.raghub_chunks (chunk_id, content, tenant_id) "
                    "VALUES ($1, $2, $3)",
                    f"chunk_{i}",
                    f"content_{i}",
                    tenant_id,
                )
            rows_migrated = migrate_row_to_schema(src_conn, dst_conn, tenant_id)
            assert rows_migrated == 10
            count = await dst_conn.fetchval(
                "SELECT COUNT(*) FROM tenant_" + tenant_id + ".raghub_chunks"
            )
            assert count == 10
        finally:
            await src_conn.close()
            await dst_conn.close()


# ---------------------------------------------------------------------------
# Non-Postgres: RowLevel isolation end-to-end via in-process stores
# ---------------------------------------------------------------------------


class TestRowLevelEndToEnd:
    """Round-trip RowLevel isolation against the in-process :class:`MemoryStore`.

    Without a Postgres database the wiring test is still meaningful:
    the strategy must filter chunks at search time. The setup
    ingests one chunk per tenant; switching the tenant context
    must change which chunks surface.
    """

    def _build_chunk(self, chunk_id: str, tenant_id: str, text: str) -> Any:
        from datetime import UTC, datetime
        from hashlib import sha256

        from raghub.models import Chunk, Classification

        return Chunk(
            id=chunk_id,
            document_id=f"doc-{chunk_id}",
            version=1,
            company="acme",
            owner="alice@example.com",
            classification=Classification.Internal,
            checksum=sha256(text.encode("utf-8")).hexdigest(),
            text=text,
            created_at=datetime(2026, 1, 1, tzinfo=UTC),
            tenant_id=tenant_id,
        )

    def test_row_level_isolation_across_search_calls(self) -> None:
        """A chunk ingested for tenant A is invisible when tenant B is active."""
        from raghub.store import MemoryStore

        store = MemoryStore(embedding_dim=2)
        store.insert(
            [
                self._build_chunk("a1", tenant_id="acme", text="acme secret"),
                self._build_chunk("b1", tenant_id="bobco", text="bobco secret"),
            ],
            [[0.1, 0.2], [0.3, 0.4]],
        )
        store.rebuild_index()

        token_a = set_current(TenantContext(tenant_id="acme"))
        try:
            hits_a = store.search(vector=[0.1, 0.2], top_k=10)
        finally:
            reset(token_a)
        ids_a = {h["chunk_id"] for h in hits_a}
        assert ids_a == {"a1"}, f"tenant acme must see only its own chunk; got {ids_a}"

        token_b = set_current(TenantContext(tenant_id="bobco"))
        try:
            hits_b = store.search(vector=[0.3, 0.4], top_k=10)
        finally:
            reset(token_b)
        ids_b = {h["chunk_id"] for h in hits_b}
        assert ids_b == {"b1"}, f"tenant bobco must see only its own chunk; got {ids_b}"

        # Without a tenant context, both chunks are reachable (the
        # admin-style read path). This is the contract that admin
        # users and the anonymous fallback depend on.
        token_none = set_current(None)
        try:
            hits_none = store.search(vector=[0.1, 0.2], top_k=10)
        finally:
            reset(token_none)
        assert {h["chunk_id"] for h in hits_none} == {"a1", "b1"}
