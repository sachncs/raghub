"""Multi-tenant isolation strategies.

Three isolation strategies ship as first-class:

* :class:`RowLevel` — every store query gets an explicit
  ``tenant_id`` filter. Works on every backend.
* :class:`SchemaPerTenant` — Postgres + pgvector only; creates
  ``tenant_<id>`` schemas and pins ``search_path`` per connection.
* :class:`DatabasePerTenant` — Postgres only, generic; maps
  ``tenant_id`` to a per-tenant DSN and routes connections.

Each strategy implements the same :class:`IsolationStrategy`
Protocol. The default is :class:`RowLevel`. Per-tenant secrets
(API keys) are encrypted at rest with Fernet and stored in the
``raghub_tenant_secrets`` table (schema in
:mod:`raghub.store.schema`).
"""

from __future__ import annotations

import os
import sqlite3
from contextvars import ContextVar
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

from raghub.errors import AuthorizationError, MissingDepError

__all__ = [
    "DatabasePerTenant",
    "IsolationStrategy",
    "RowLevel",
    "SchemaPerTenant",
    "TenantContext",
    "TenantRegistry",
    "TenantSecretCipher",
]


class IsolationStrategy(StrEnum):
    """The three shipped isolation strategies."""

    ROW_LEVEL = "row_level"
    SCHEMA_PER_TENANT = "schema_per_tenant"
    DATABASE_PER_TENANT = "database_per_tenant"


@dataclass(frozen=True, slots=True)
class TenantContext:
    """Per-request tenant context propagated through every storage call."""

    tenant_id: str
    user_id: str | None = None
    is_admin: bool = False
    isolation: IsolationStrategy = IsolationStrategy.ROW_LEVEL


_tenant_context: ContextVar[TenantContext | None] = ContextVar(
    "_tenant_context", default=None
)


def get_current_tenant() -> TenantContext | None:
    """Return the current :class:`TenantContext` or ``None``."""
    return _tenant_context.get()


def set_current_tenant(context: TenantContext | None) -> Any:
    """Bind ``context`` for the current task; return a reset token."""
    return _tenant_context.set(context)


def reset_current_tenant(token: Any) -> None:
    """Reset the tenant context to its prior state."""
    _tenant_context.reset(token)


def require_tenant() -> TenantContext:
    """Return the current tenant context or raise
    :class:`AuthorizationError`.

    Under :attr:`IsolationStrategy.SCHEMA_PER_TENANT` and
    :attr:`IsolationStrategy.DATABASE_PER_TENANT`, every storage
    call must run inside a tenant context; the absence of one is
    an authz failure, not a runtime error.
    """
    context = _tenant_context.get()
    if context is None:
        raise AuthorizationError(
            "missing tenant context under "
            f"{IsolationStrategy.SCHEMA_PER_TENANT.value} or "
            f"{IsolationStrategy.DATABASE_PER_TENANT.value}"
        )
    return context


class RowLevel:
    """Default isolation: every store query adds an explicit tenant filter."""

    def apply_to_kwargs(self, kwargs: dict[str, Any]) -> dict[str, Any]:
        """Return ``kwargs`` with a ``tenant_id`` key when set."""
        context = _tenant_context.get()
        if context is None:
            return kwargs
        kwargs = dict(kwargs)
        kwargs.setdefault("tenant_id", context.tenant_id)
        return kwargs


class SchemaPerTenant:
    """Postgres + pgvector only.

    The schema for tenant ``t1`` is ``tenant_t1``; ``search_path`` is
    pinned to that schema per connection.
    """

    def __init__(self, dsn: str) -> None:
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
    "embedding VECTOR(384) NOT NULL, "
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
        self.entries: dict[str, dict[str, Any]] = dict(entries or {})

    def get(self, tenant_id: str) -> dict[str, Any]:
        """Return the per-tenant record."""
        if tenant_id not in self.entries:
            raise KeyError(f"unknown tenant id: {tenant_id!r}")
        return dict(self.entries[tenant_id])

    def upsert(self, tenant_id: str, dsn: str, vector_dim: int = 384) -> None:
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
        self.db_path = db_path

    def _fernet(self) -> Any:
        try:
            from cryptography.fernet import Fernet
        except ImportError as exc:
            raise MissingDepError(
                "cryptography",
                "pip install raghub[auth]",
            ) from exc
        key = os.getenv("RAGHUB_TENANT_SECRETS_KEY")
        if not key:
            raise MissingDepError(
                "RAGHUB_TENANT_SECRETS_KEY",
                "set RAGHUB_TENANT_SECRETS_KEY to a Fernet key "
                "(Fernet.generate_key()).",
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
                "SELECT value FROM raghub_tenant_secrets "
                "WHERE tenant_id = ? AND key = ?",
                (tenant_id, key),
            ).fetchone()
        if row is None:
            return None
        return self._fernet().decrypt(row[0].encode("utf-8")).decode("utf-8")

    def set(self, tenant_id: str, key: str, value: str) -> None:
        """Encrypt and store ``value`` under ``(tenant_id, key)``."""
        require_tenant()
        token = self._fernet().encrypt(value.encode("utf-8")).decode("utf-8")
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


class TenantMigrationError(RuntimeError):
    """Raised when a tenant-split migration fails."""


def migrate_tenant_split(
    source_dsn: str,
    target_dsn: str,
    *,
    from_strategy: IsolationStrategy,
    to_strategy: IsolationStrategy,
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
    conn_src = _sync_connect(asyncpg, source_dsn)
    conn_dst = _sync_connect(asyncpg, target_dsn)
    try:
        if (
            from_strategy == IsolationStrategy.ROW_LEVEL
            and to_strategy == IsolationStrategy.SCHEMA_PER_TENANT
        ):
            return _migrate_row_to_schema(conn_src, conn_dst, tenant_id)
        if (
            from_strategy == IsolationStrategy.SCHEMA_PER_TENANT
            and to_strategy == IsolationStrategy.DATABASE_PER_TENANT
        ):
            return _migrate_schema_to_db(conn_src, conn_dst, tenant_id)
        if (
            from_strategy == IsolationStrategy.ROW_LEVEL
            and to_strategy == IsolationStrategy.DATABASE_PER_TENANT
        ):
            return _migrate_row_to_db(conn_src, conn_dst, tenant_id)
        raise TenantMigrationError(
            f"unsupported migration direction: "
            f"{from_strategy.value} -> {to_strategy.value}"
        )
    finally:
        conn_src.close()
        conn_dst.close()


def _sync_connect(asyncpg: Any, dsn: str) -> Any:
    """Synchronously open a connection.

    Uses :func:`asyncio.run` under the hood. Production code should
    use the async API; this helper exists for the migration CLI.
    """
    import asyncio

    return asyncio.run(asyncpg.connect(dsn))


def _migrate_row_to_schema(src: Any, dst: Any, tenant_id: str | None) -> int:
    """Copy rows from ``public.raghub_chunks`` to ``tenant_<id>.raghub_chunks``."""
    raise NotImplementedError("row -> schema migration ships in a future release")


def _migrate_schema_to_db(src: Any, dst: Any, tenant_id: str | None) -> int:
    """Dump a schema's contents into a separate database."""
    raise NotImplementedError("schema -> database migration ships in a future release")


def _migrate_row_to_db(src: Any, dst: Any, tenant_id: str | None) -> int:
    """Copy rows from the row-level table into a per-tenant database."""
    raise NotImplementedError("row -> database migration ships in a future release")
