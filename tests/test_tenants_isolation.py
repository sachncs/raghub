"""Tests for raghub.tenants.isolation — Tier 2 v0.9.0.

Each test is gated on optional infrastructure (Postgres + pgvector)
so the suite runs without external services.
"""

from __future__ import annotations

import os

import pytest

from raghub.config import Settings, TenantsConfig
from raghub.tenants import (
    CompositeTenantResolver,
    HeaderTenantResolver,
    JwtClaimTenantResolver,
    validate_tenant_id,
)
from raghub.tenants.isolation import (
    DatabasePerTenant,
    IsolationStrategy,
    RowLevel,
    SchemaPerTenant,
    TenantContext,
    TenantRegistry,
    get_current_tenant,
    require_tenant,
    reset_current_tenant,
    set_current_tenant,
)


# ---------------------------------------------------------------------------
# validate_tenant_id
# ---------------------------------------------------------------------------


class TestValidateTenantId:
    @pytest.mark.parametrize(
        "tenant_id",
        ["abc", "acme", "tenant-1", "tenant_2", "abc-def-ghi"],
    )
    def test_valid_tenant_ids_accepted(self, tenant_id: str) -> None:
        validate_tenant_id(tenant_id)

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
            validate_tenant_id(tenant_id)


# ---------------------------------------------------------------------------
# TenantContext / contextvars
# ---------------------------------------------------------------------------


class TestTenantContext:
    def test_get_current_tenant_returns_none_by_default(self) -> None:
        token = set_current_tenant(None)
        try:
            assert get_current_tenant() is None
        finally:
            reset_current_tenant(token)

    def test_set_and_reset_tenant(self) -> None:
        ctx = TenantContext(tenant_id="acme")
        token = set_current_tenant(ctx)
        try:
            assert get_current_tenant() is ctx
        finally:
            reset_current_tenant(token)
        assert get_current_tenant() is None


class TestRequireTenant:
    def test_require_tenant_raises_when_no_context(self) -> None:
        from raghub.errors import AuthorizationError

        token = set_current_tenant(None)
        try:
            with pytest.raises(AuthorizationError, match="missing tenant context"):
                require_tenant()
        finally:
            reset_current_tenant(token)

    def test_require_tenant_returns_context_when_bound(self) -> None:
        ctx = TenantContext(tenant_id="acme")
        token = set_current_tenant(ctx)
        try:
            assert require_tenant() is ctx
        finally:
            reset_current_tenant(token)


# ---------------------------------------------------------------------------
# RowLevel.filter_query — Tier 2 Item 8
# ---------------------------------------------------------------------------


class TestRowLevelFilterQuery:
    def test_filter_query_returns_empty_when_no_context(self) -> None:
        token = set_current_tenant(None)
        try:
            clause, params = RowLevel().filter_query()
            assert clause == ""
            assert params == {}
        finally:
            reset_current_tenant(token)

    def test_filter_query_returns_clause_when_context_bound(self) -> None:
        ctx = TenantContext(tenant_id="acme")
        token = set_current_tenant(ctx)
        try:
            clause, params = RowLevel().filter_query()
            assert clause == "tenant_id = :tenant_id"
            assert params == {"tenant_id": "acme"}
        finally:
            reset_current_tenant(token)

    def test_filter_query_custom_column(self) -> None:
        ctx = TenantContext(tenant_id="acme")
        token = set_current_tenant(ctx)
        try:
            clause, params = RowLevel().filter_query(column="owner")
            assert clause == "owner = :tenant_id"
        finally:
            reset_current_tenant(token)

    def test_filter_query_custom_operator(self) -> None:
        ctx = TenantContext(tenant_id="acme")
        token = set_current_tenant(ctx)
        try:
            clause, params = RowLevel().filter_query(operator="!=")
            assert clause == "tenant_id != :tenant_id"
        finally:
            reset_current_tenant(token)


class TestRowLevelApplyToKwargs:
    def test_apply_to_kwargs_passes_through_when_no_context(self) -> None:
        token = set_current_tenant(None)
        try:
            kwargs = {"foo": "bar"}
            assert RowLevel().apply_to_kwargs(kwargs) is kwargs
        finally:
            reset_current_tenant(token)

    def test_apply_to_kwargs_injects_tenant_id(self) -> None:
        ctx = TenantContext(tenant_id="acme")
        token = set_current_tenant(ctx)
        try:
            kwargs = {"query_vector": [0.1, 0.2]}
            result = RowLevel().apply_to_kwargs(kwargs)
            assert result["tenant_id"] == "acme"
            # original is not mutated
            assert "tenant_id" not in kwargs
        finally:
            reset_current_tenant(token)

    def test_apply_to_kwargs_does_not_overwrite_explicit_tenant_id(self) -> None:
        ctx = TenantContext(tenant_id="acme")
        token = set_current_tenant(ctx)
        try:
            kwargs = {"tenant_id": "explicit"}
            result = RowLevel().apply_to_kwargs(kwargs)
            assert result["tenant_id"] == "explicit"
        finally:
            reset_current_tenant(token)


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
        token = set_current_tenant(TenantContext(tenant_id="missing"))
        try:
            with pytest.raises(KeyError, match="unknown tenant id"):
                await dbt.connection_for("missing")
        finally:
            reset_current_tenant(token)


# ---------------------------------------------------------------------------
# Settings.tenants wiring
# ---------------------------------------------------------------------------


class TestTenantsSettings:
    def test_default_tenants_isolation_is_row_level(self) -> None:
        settings = Settings()
        assert settings.tenants.isolation == IsolationStrategy.ROW_LEVEL

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
    reason=f"{PG_DSN_ENV} not set; skipping live Postgres test",
)
class TestSchemaPerTenantLive:
    @pytest.mark.asyncio
    async def test_ensure_schema_creates_schema(self) -> None:
        dsn = os.environ[PG_DSN_ENV]
        # Unique tenant id per test run to avoid colliding with prior runs
        tenant_id = f"test_tenant_{os.getpid()}"
        sp = SchemaPerTenant(dsn)
        token = set_current_tenant(TenantContext(tenant_id=tenant_id))
        try:
            await sp.ensure_schema(tenant_id)
        finally:
            reset_current_tenant(token)


@pytest.mark.skipif(
    not os.environ.get(PG_DSN_ENV),
    reason=f"{PG_DSN_ENV} not set; skipping live Postgres test",
)
class TestDatabasePerTenantLive:
    @pytest.mark.asyncio
    async def test_connection_for_returns_asyncpg_connection(self) -> None:
        dsn = os.environ[PG_DSN_ENV]
        registry = TenantRegistry()
        registry.upsert("acme", dsn=dsn, vector_dim=384)
        dbt = DatabasePerTenant(registry)
        token = set_current_tenant(TenantContext(tenant_id="acme"))
        try:
            conn = await dbt.connection_for("acme")
            assert conn is not None
            await conn.close()
        finally:
            reset_current_tenant(token)
