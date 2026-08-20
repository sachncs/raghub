# Isolation tiers

RAGHub supports three multi-tenant isolation tiers, ranging from
application-level row filtering to full database-per-tenant separation.

See [ADR 0014](../adr/0014-multi-tenant-isolation.md) for the design rationale.

## Configuration

Set `RAGHUB_ISOLATION_TIER` in your environment:

| Value | Description |
|-------|-------------|
| `row_level` | **Default.** All tenants share one database; queries are filtered by `tenant_id`. |
| `schema_per_tenant` | Each tenant gets its own PostgreSQL schema within one database. |
| `database_per_tenant` | Each tenant gets its own PostgreSQL database; connection routing via `TenantRegistry`. |

## Tier 1 — Row-level (default)

Every table that stores tenant-scoped data includes a `tenant_id TEXT`
column. A PostgreSQL Row-Level Security (RLS) policy enforces:

```sql
CREATE POLICY tenant_isolation ON chunks
  USING (tenant_id = current_setting('app.current_tenant'));
```

**Pros:** Zero operational overhead; single database to back up.

**Cons:** All tenants share indexes and connection pool; a query bug
could leak data if RLS is misconfigured.

## Tier 2 — Schema-per-tenant

Each tenant receives a dedicated PostgreSQL schema. Connection routing
sets `search_path` at request time:

```sql
SET search_path TO tenant_acme, public;
```

**Pros:** Stronger isolation; per-tenant DDL without affecting others.

**Cons:** More complex migrations; schema count can grow large; backup
must iterate schemas.

## Tier 3 — Database-per-tenant

Each tenant has a separate PostgreSQL database. The `TenantRegistry`
maps tenant IDs to connection strings, and the `TenantResolver` chain
resolves the current tenant from the request:

```python
from raghub.isolation import CompositeTenantResolver, JwtClaimTenantResolver, HeaderTenantResolver

resolver = CompositeTenantResolver([
    JwtClaimTenantResolver(claim="org_id"),
    HeaderTenantResolver(header="X-Tenant-Id"),
])
```

**Pros:** Complete data separation; independent scaling and backup.

**Cons:** Highest operational cost; connection pool per database; cross-tenant
queries are impossible by design.

## Per-tenant secrets

Each tier supports per-tenant encryption keys managed by
`TenantSecretCipher`. Secrets are stored in a `tenant_secrets` table
(or per-database for tier 3) and resolved at request time.

## Migration between tiers

Use the CLI to promote a tenant:

```bash
raghub migrate-tenant --from row_level --to schema_per_tenant --tenant acme
```

This copies existing rows into a new schema and verifies RLS policies.
