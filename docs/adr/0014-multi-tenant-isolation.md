# ADR 0014 — Multi-tenant isolation

## Status

Accepted (v0.7.6)

## Context

Revex must serve multiple organisations with data isolation guarantees.
Early releases stored all tenants in a single SQLite database with a
`tenant_id` column, which is insufficient for regulated workloads where
cross-tenant access is unacceptable even at the database-engine level.

The v0.7.6 threat model formalised three isolation tiers and shipped
the first set of row-level security policies for PostgreSQL.

## Decision

We adopt a three-tier isolation model:

| Tier | Mechanism | Default |
|------|-----------|---------|
| **Row-level** | `WHERE tenant_id = ?` predicates enforced in the ORM | Yes |
| **Schema-per-tenant** | Separate PostgreSQL schemas within one database | Optional |
| **Database-per-tenant** | Separate PostgreSQL databases; connection routing via `TenantRegistry` | Optional |

Row-level isolation is always active. Schema- and database-per-tenant
tiers are opt-in via `REVEX_ISOLATION_TIER` (`row_level` | `schema_per_tenant` |
`database_per_tenant`).

Per-tenant secrets are managed by `TenantSecretCipher` and resolved
at request time by the `TenantResolver` chain (JWT claim → HTTP header
→ no-tenant fallback).

## Consequences

- **Operational cost**: Schema- and database-per-tenant tiers require
  additional Postgres setup and monitoring. Backups must be per-tenant.
- **RLS policies**: Row-level policies are required for every table
  that stores tenant-scoped data. New tables must be audited.
- **Migration path**: Tenants can be promoted from row-level to
  schema-per-tenant without downtime by running the
  `revex migrate-tenant` CLI.
- **Testing overhead**: CI must exercise all three tiers to prevent
  isolation regressions.

## Alternatives considered

- **Single-tenant**: Rejected — does not scale to SaaS deployments.
- **Namespace-per-tenant (e.g., prefix on table names)**: Rejected —
  provides no real isolation; a single misconfigured query can leak data.
- **Logical replication with separate Postgres instances per tenant**:
  Considered for extreme compliance scenarios but deferred to a future
  release due to operational complexity.
