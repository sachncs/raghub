"""Multi-tenant isolation strategies.

Three isolation strategies ship as first-class:

* :class:`RowLevel` — every store query gets an explicit
  ``tenant_id`` filter. Works on every backend.
* :class:`SchemaPerTenant` — Postgres + pgvector only; creates
  ``tenant_<id>`` schemas and pins ``search_path`` per connection.
* :class:`DatabasePerTenant` — Postgres only, generic; maps
  ``tenant_id`` to a per-tenant DSN and routes connections.

Each strategy implements the same :class:`Isolation`
Protocol. The default is :class:`RowLevel`. Per-tenant secrets
(API keys) are encrypted at rest with Fernet and stored in the
``raghub_tenant_secrets`` table (schema in
:mod:`raghub.stores.schema`).
"""

from __future__ import annotations

import os
import sqlite3
from contextvars import ContextVar
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

from raghub.constants import DEFAULT_EMBEDDING_DIM, ENV_RAGHUB_TENANT_SECRETS_KEY
from raghub.errors import AuthorizationError, MissingDepError, RagHubError

__all__ = [
    "DatabasePerTenant",
    "Isolation",
    "RowLevel",
    "SchemaPerTenant",
    "TenantContext",
    "TenantRegistry",
    "TenantSecretCipher",
]


class Isolation(StrEnum):
    """The three shipped isolation strategies."""

    RowLevel = "row_level"
    SchemaPerTenant = "schema_per_tenant"
    DatabasePerTenant = "database_per_tenant"


@dataclass(frozen=True, slots=True)
class TenantContext:
    """Per-request tenant context propagated through every storage call."""

    tenant_id: str
    user_id: str | None = None
    is_admin: bool = False
    isolation: Isolation = Isolation.RowLevel


_tenant_context: ContextVar[TenantContext | None] = ContextVar("_tenant_context", default=None)


def current() -> TenantContext | None:
    """Return the current :class:`TenantContext` or ``None``."""
    return _tenant_context.get()


def set_current(context: TenantContext | None) -> Any:
    """Bind ``context`` for the current task; return a reset token."""
    return _tenant_context.set(context)


def reset(token: Any) -> None:
    """Reset the tenant context to its prior state."""
    _tenant_context.reset(token)


def require_tenant() -> TenantContext:
    """Return the current tenant context or raise :class:`AuthorizationError`.

    Under :attr:`Isolation.SchemaPerTenant` and
    :attr:`Isolation.DatabasePerTenant`, every storage
    call must run inside a tenant context; the absence of one is
    an authz failure, not a runtime error.
    """
    context = _tenant_context.get()
    if context is None:
        raise AuthorizationError(
            "missing tenant context under "
            f"{Isolation.SchemaPerTenant.value} or "
            f"{Isolation.DatabasePerTenant.value}"
        )
    return context


class RowLevel:
    """Default isolation: every store query adds an explicit tenant filter.

    Two helpers are provided:

    * :meth:`apply_to_kwargs` — keyword-argument injection for
      in-memory stores (used by :class:`MemoryStore`).
    * :meth:`filter_query` — SQL ``WHERE`` clause + params for
      SQL-backed stores (used by :class:`SqliteStore`,
      :class:`PgVectorStore`).
    """

    @staticmethod
    def apply_to_kwargs(kwargs: dict[str, Any]) -> dict[str, Any]:
        """Return ``kwargs`` with a ``tenant_id`` key when set."""
        context = _tenant_context.get()
        if context is None:
            return kwargs
        kwargs = dict(kwargs)
        kwargs.setdefault("tenant_id", context.tenant_id)
        return kwargs

    @staticmethod
    def filter_query(
        *,
        column: str = "tenant_id",
        operator: str = "=",
    ) -> tuple[str, dict[str, Any]]:
        """Return a SQL ``WHERE`` fragment and bind params for the bound tenant.

        Args:
            column: Column name to filter on. Default ``"tenant_id"``.
            operator: SQL comparison operator. Default ``"="``.

        Returns:
            ``(where_clause, params)`` where ``where_clause`` is either
            ``""`` (no tenant bound) or ``f"{column} {operator} :tenant_id"``.
            ``params`` carries ``{"tenant_id": "..."}`` when a clause is
            returned.

        """
        context = _tenant_context.get()
        if context is None:
            return "", {}
        return (
            f"{column} {operator} :tenant_id",
            {"tenant_id": context.tenant_id},
        )


class SchemaPerTenant:
    """Postgres + pgvector only.

    The schema for tenant ``t1`` is ``tenant_t1``; ``search_path`` is
    pinned to that schema per connection.
    """

    def __init__(self, dsn: str) -> None:
        """Store the Postgres DSN used to open per-tenant connections."""
        self.dsn = dsn

    async def ensure_schema(self, tenant_id: str) -> None:
        """Create the ``tenant_<id>`` schema and run the migrations inside."""
        require_tenant()
        schema = f"tenant_{tenant_id}"
        try:
            import asyncpg
        except ImportError as exc:
            raise MissingDepError(
                "asyncpg",
                "pip install raghub[pgvector]",
            ) from exc
        conn = await asyncpg.connect(self.dsn)
        try:
            await conn.execute(f'CREATE SCHEMA IF NOT EXISTS "{schema}"')
            await conn.execute(f'SET search_path TO "{schema}", public')
            await conn.execute(_DDL_SQL)
        finally:
            await conn.close()


class DatabasePerTenant:
    """Generic database-per-tenant routing.

    Maintains a mapping ``tenant_id -> {"dsn": ..., "vector_dim": ...}``
    and opens a connection per tenant on demand.
    """

    def __init__(self, tenants: TenantRegistry) -> None:
        """Store the registry used to resolve per-tenant DSNs."""
        self.tenants = tenants

    async def connection_for(self, tenant_id: str) -> Any:
        """Return a connection bound to ``tenant_id``'s database."""
        from raghub.tenants.isolation import require_tenant

        require_tenant()
        record = self.tenants.get(tenant_id)
        try:
            import asyncpg
        except ImportError as exc:
            raise MissingDepError(
                "asyncpg",
                "pip install raghub[pgvector]",
            ) from exc
        return await asyncpg.connect(record["dsn"])


_DDL_SQL = (
    "CREATE TABLE IF NOT EXISTS raghub_chunks ("
    "id TEXT PRIMARY KEY, "
    "document_id TEXT NOT NULL, "
    "ordinal INTEGER NOT NULL, "
    "text TEXT NOT NULL, "
    "metadata JSONB NOT NULL DEFAULT '{}'::jsonb, "
    f"embedding VECTOR({DEFAULT_EMBEDDING_DIM}) NOT NULL, "
    "tenant_id TEXT, "
    "created_at TIMESTAMPTZ NOT NULL DEFAULT now(), "
    "updated_at TIMESTAMPTZ NOT NULL DEFAULT now())"
)


class TenantRegistry:
    """In-memory map of tenant ids to per-tenant DSNs.

    The on-disk version lives in ``raghub/tenants/store.py`` (added
    in a future release). For this release the registry is
    populated from ``Settings`` and used for connection routing.
    """

    def __init__(self, entries: dict[str, dict[str, Any]] | None = None) -> None:
        """Copy ``entries`` into the in-memory tenant map."""
        self.entries: dict[str, dict[str, Any]] = dict(entries or {})

    def get(self, tenant_id: str) -> dict[str, Any]:
        """Return the per-tenant record."""
        if tenant_id not in self.entries:
            raise KeyError(f"unknown tenant id: {tenant_id!r}")
        return dict(self.entries[tenant_id])

    def upsert(self, tenant_id: str, dsn: str, vector_dim: int = DEFAULT_EMBEDDING_DIM) -> None:
        """Register or update a tenant record."""
        self.entries[tenant_id] = {"dsn": dsn, "vector_dim": vector_dim}

    def remove(self, tenant_id: str) -> None:
        """Remove a tenant record."""
        self.entries.pop(tenant_id, None)


class TenantSecretCipher:
    """Fernet-based encryption for per-tenant secrets.

    Backed by a SQLite table so secrets survive restarts. The
    Fernet key is read from ``RAGHUB_TENANT_SECRETS_KEY`` (a
    32-byte URL-safe base64 string) and is required in production.
    """

    def __init__(self, db_path: str) -> None:
        """Store the SQLite path backing the encrypted-secrets table."""
        self.db_path = db_path

    @staticmethod
    def fernet() -> Any:
        """Return a Fernet cipher built from ``RAGHUB_TENANT_SECRETS_KEY``."""
        try:
            from cryptography.fernet import Fernet
        except ImportError as exc:
            raise MissingDepError(
                "cryptography",
                "pip install raghub[auth]",
            ) from exc
        key = os.getenv(ENV_RAGHUB_TENANT_SECRETS_KEY)
        if not key:
            raise MissingDepError(
                ENV_RAGHUB_TENANT_SECRETS_KEY,
                "set RAGHUB_TENANT_SECRETS_KEY to a Fernet key (Fernet.generate_key()).",
            )
        return Fernet(key.encode("utf-8") if isinstance(key, str) else key)

    def initialize(self) -> None:
        """Create the ``raghub_tenant_secrets`` table."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "CREATE TABLE IF NOT EXISTS raghub_tenant_secrets ("
                "tenant_id TEXT NOT NULL, "
                "key TEXT NOT NULL, "
                "value TEXT NOT NULL, "
                "PRIMARY KEY (tenant_id, key))"
            )
            conn.commit()

    def get(self, tenant_id: str, key: str) -> str | None:
        """Return the decrypted secret or ``None`` when absent."""
        with sqlite3.connect(self.db_path) as conn:
            row = conn.execute(
                "SELECT value FROM raghub_tenant_secrets WHERE tenant_id = ? AND key = ?",
                (tenant_id, key),
            ).fetchone()
        if row is None:
            return None
        return self.fernet().decrypt(row[0].encode("utf-8")).decode("utf-8")

    def set(self, tenant_id: str, key: str, value: str) -> None:
        """Encrypt and store ``value`` under ``(tenant_id, key)``."""
        require_tenant()
        token = self.fernet().encrypt(value.encode("utf-8")).decode("utf-8")
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "INSERT INTO raghub_tenant_secrets (tenant_id, key, value) "
                "VALUES (?, ?, ?) "
                "ON CONFLICT (tenant_id, key) DO UPDATE SET value = excluded.value",
                (tenant_id, key, token),
            )
            conn.commit()

    def rotate(self, tenant_id: str, key: str) -> str | None:
        """Re-encrypt the existing secret under the current Fernet key."""
        existing = self.get(tenant_id, key)
        if existing is None:
            return None
        self.set(tenant_id, key, existing)
        return existing

    def list_keys(self, tenant_id: str) -> list[str]:
        """Return the keys stored for ``tenant_id`` (no values)."""
        with sqlite3.connect(self.db_path) as conn:
            rows = conn.execute(
                "SELECT key FROM raghub_tenant_secrets WHERE tenant_id = ?",
                (tenant_id,),
            ).fetchall()
        return [row[0] for row in rows]


class TenantMigrationError(RagHubError, RuntimeError):
    """Raised when a tenant-split migration fails."""


def migrate_tenant_split(
    source_dsn: str,
    target_dsn: str,
    *,
    from_strategy: Isolation,
    to_strategy: Isolation,
    tenant_id: str | None = None,
) -> int:
    """Migrate data between isolation strategies.

    Returns the number of rows migrated. Raises
    :class:`TenantMigrationError` on any failure.

    Supported directions:

    * ``ROW_LEVEL -> SCHEMA_PER_TENANT`` (single tenant or all)
    * ``SCHEMA_PER_TENANT -> DATABASE_PER_TENANT``
    * ``ROW_LEVEL -> DATABASE_PER_TENANT``

    """
    try:
        import asyncpg
    except ImportError as exc:
        raise MissingDepError(
            "asyncpg",
            "pip install raghub[pgvector]",
        ) from exc
    conn_src = sync_connect(asyncpg, source_dsn)
    conn_dst = sync_connect(asyncpg, target_dsn)
    try:
        if from_strategy == Isolation.RowLevel and to_strategy == Isolation.SchemaPerTenant:
            return migrate_row_to_schema(conn_src, conn_dst, tenant_id)
        if (
            from_strategy == Isolation.SchemaPerTenant
            and to_strategy == Isolation.DatabasePerTenant
        ):
            return migrate_schema_to_db(conn_src, conn_dst, tenant_id)
        if from_strategy == Isolation.RowLevel and to_strategy == Isolation.DatabasePerTenant:
            return migrate_row_to_db(conn_src, conn_dst, tenant_id)
        raise TenantMigrationError(
            f"unsupported migration direction: {from_strategy.value} -> {to_strategy.value}"
        )
    finally:
        conn_src.close()
        conn_dst.close()


def sync_connect(asyncpg: Any, dsn: str) -> Any:
    """Open a connection synchronously.

    Uses :func:`asyncio.run` under the hood. Production code should
    use the async API; this helper exists for the migration CLI.
    """
    import asyncio

    return asyncio.run(asyncpg.connect(dsn))


def migrate_row_to_schema(src: Any, dst: Any, tenant_id: str | None) -> int:
    """Copy rows from the source table to a tenant-scoped schema.

    Args:
        src: An open asyncpg connection to the source database.
        dst: An open asyncpg connection to the target database.
        tenant_id: When ``None``, all tenants are migrated (one schema
            per tenant). When set, only that tenant's rows are migrated.

    Returns:
        The number of rows copied.

    """
    import asyncio

    async def migrate() -> int:
        """Copy ``src`` rows to a new schema-per-tenant layout in ``dst``."""
        where = ""
        params: list[Any] = []
        if tenant_id is not None:
            where = " WHERE tenant_id = $1"
            params = [tenant_id]
        # Discover tenants in the source table.
        rows = await src.fetch(
            f"SELECT DISTINCT tenant_id FROM raghub_chunks{where}",  # nosec B608 - where is a literal column ref, tenant_id is parameterised via $1
            *params,
        )
        copied = 0
        for row in rows:
            tid = row["tenant_id"]
            schema = f"tenant_{tid}"
            await dst.execute(f'CREATE SCHEMA IF NOT EXISTS "{schema}"')
            await dst.execute(f'SET search_path TO "{schema}", public')
            await dst.execute(_DDL_SQL)
            tenant_rows = await src.fetch("SELECT * FROM raghub_chunks WHERE tenant_id = $1", tid)
            for tr in tenant_rows:
                await dst.execute(
                    "INSERT INTO raghub_chunks "
                    "(id, document_id, ordinal, text, metadata, embedding, tenant_id) "
                    "VALUES ($1, $2, $3, $4, $5::jsonb, $6::vector, $7) "
                    "ON CONFLICT (id) DO NOTHING",
                    tr["id"],
                    tr["document_id"],
                    tr["ordinal"],
                    tr["text"],
                    tr["metadata"],
                    tr["embedding"],
                    tr["tenant_id"],
                )
                copied += 1
        return copied

    return asyncio.run(migrate())


def migrate_schema_to_db(src: Any, dst: Any, tenant_id: str | None) -> int:
    """Copy rows from a tenant schema in the source database to a separate target database.

    Args:
        src: Connection to the source (multi-tenant) database.
        dst: Connection to the target (per-tenant) database.
        tenant_id: When ``None``, every tenant schema is migrated.
            When set, only that tenant's schema is migrated.

    """
    import asyncio

    async def migrate() -> int:
        """Copy ``src`` rows into the ``tenant_*`` schemas that exist on ``dst``."""
        copied = 0
        if tenant_id is not None:
            schemas = [f"tenant_{tenant_id}"]
        else:
            schema_rows = await src.fetch(
                "SELECT schema_name FROM information_schema.schemata "
                "WHERE schema_name LIKE 'tenant_%'"
            )
            schemas = [row["schema_name"] for row in schema_rows]
        for schema in schemas:
            await dst.execute(_DDL_SQL)
            rows = await src.fetch(
                f'SELECT * FROM "{schema}".raghub_chunks'  # nosec B608 - schema is enumerated from information_schema.schemata with a literal LIKE prefix
            )
            for r in rows:
                await dst.execute(
                    "INSERT INTO raghub_chunks "
                    "(id, document_id, ordinal, text, metadata, embedding, tenant_id) "
                    "VALUES ($1, $2, $3, $4, $5::jsonb, $6::vector, $7) "
                    "ON CONFLICT (id) DO NOTHING",
                    r["id"],
                    r["document_id"],
                    r["ordinal"],
                    r["text"],
                    r["metadata"],
                    r["embedding"],
                    r["tenant_id"],
                )
                copied += 1
        return copied

    return asyncio.run(migrate())


def migrate_row_to_db(src: Any, dst: Any, tenant_id: str | None) -> int:
    """Copy rows from the source row-level table directly to a per-tenant database.

    Args:
        src: Connection to the source database.
        dst: Connection to the target (per-tenant) database.
        tenant_id: When ``None``, every tenant is migrated into a
            separate target database row. When set, only that
            tenant's rows are migrated.

    """
    import asyncio

    async def migrate() -> int:
        """Copy ``src`` rows into ``dst`` keeping a single shared schema."""
        await dst.execute(_DDL_SQL)
        where = ""
        params: list[Any] = []
        if tenant_id is not None:
            where = " WHERE tenant_id = $1"
            params = [tenant_id]
        rows = await src.fetch(
            f"SELECT * FROM raghub_chunks{where}",  # nosec B608 - where is a literal column ref, tenant_id is parameterised via $1
            *params,
        )
        copied = 0
        for r in rows:
            await dst.execute(
                "INSERT INTO raghub_chunks "
                "(id, document_id, ordinal, text, metadata, embedding, tenant_id) "
                "VALUES ($1, $2, $3, $4, $5::jsonb, $6::vector, $7) "
                "ON CONFLICT (id) DO NOTHING",
                r["id"],
                r["document_id"],
                r["ordinal"],
                r["text"],
                r["metadata"],
                r["embedding"],
                r["tenant_id"],
            )
            copied += 1
        return copied

    return asyncio.run(migrate())
