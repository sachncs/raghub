# Tier 2 — Make v0.7.6 isolation tiers real (Items 8-15)

The v0.7.6 plan shipped `IsolationStrategy`, `RowLevel`,
`SchemaPerTenant`, `DatabasePerTenant`, `migrate_tenant_split` —
but the latter four are stubs or raise `NotImplementedError`. Tier 2
makes them real and wires row-level filtering into every vector store.

---

## Item 8 — `RowLevel.filter_query` helper

- **File(s)**: `raghub/tenants/isolation.py`, `tests/test_tenants_isolation.py` (new file)
- **Change**: Add `RowLevel.filter_query(self, kwargs: dict) -> tuple[str, dict]` returning `("tenant_id = :tenant_id", {"tenant_id": "..."})` when `TenantContext` is bound; `("", {})` when not.
- **Test**: `tests/test_tenants_isolation.py::test_row_level_filter_query_when_tenant_bound`, `::test_row_level_filter_query_when_no_context`.
- **Acceptance criteria**:
  - R3 — single-word class name; one-word discriminator not needed (method, not class).
  - R8 — behaviour observable through tests.
  - T1, T3 — pass.
- **Success criteria**:
  - `RowLevel().filter_query({})` returns `("", {})`.
  - With `set_current_tenant(TenantContext(tenant_id="alice"))`, `filter_query({})` returns `("tenant_id = :tenant_id", {"tenant_id": "alice"})`.

---

## Item 9 — `MemoryStore.search` honours `tenant_id`

- **File(s)**: `raghub/store.py`, `tests/test_store_memory.py`
- **Change**: Add `tenant_id: str | None = None` kwarg to `MemoryStore.search`; when set, apply `RowLevel.filter_query` semantics (post-filter chunks whose `tenant_id` doesn't match).
- **Test**: `tests/test_store_memory.py::test_search_filters_by_tenant_id`.
- **Acceptance criteria**:
  - R8 — `verify()` invoked; per-chunk `tenant_id` checked.
  - R9 — assertion names the value being filtered.
  - T1, T3 — pass.
- **Success criteria**:
  - Insert 3 chunks, 2 with `tenant_id="alice"`, 1 with `tenant_id="bob"`.
  - `search(..., tenant_id="alice")` returns only the 2 alice chunks.
  - `search(..., tenant_id="bob")` returns only the bob chunk.
  - `search(...)` (no `tenant_id`) returns all 3.

---

## Item 10 — `MemoryStore.insert` stores `tenant_id`

- **File(s)**: `raghub/store.py`, `tests/test_store_memory.py`
- **Change**: `MemoryStore.insert` reads `chunk.tenant_id` and stores it on `MemoryVectorRecord`.
- **Test**: `tests/test_store_memory.py::test_insert_stores_tenant_id_round_trip`.
- **Acceptance criteria**:
  - R8 — round-trip preserves `tenant_id` via `verify()`.
  - T1, T3 — pass.
- **Success criteria**:
  - Insert chunk with `tenant_id="acme"`, retrieve via `search`, verify `record.metadata["tenant_id"] == "acme"`.

---

## Item 11 — `SqliteStore` `tenant_id` column

- **File(s)**: `raghub/store.py`, `raghub/store/schema.py`, `tests/test_sqlite_store.py`
- **Change**: Add `tenant_id TEXT` to `DOCUMENTS_SCHEMA_SQL` (and to the legacy schema in `store.py` for upgrades). Update `SqliteStore.search` to append `WHERE tenant_id = ?` when `tenant_id` kwarg is set.
- **Test**: `tests/test_sqlite_store.py::test_sqlite_store_search_filters_by_tenant_id`.
- **Acceptance criteria**:
  - O2 — single source of truth for SQL schemas (the new column lives in `raghub/store/schema.py`).
  - R6 — format version bumped on the schema.
  - T1, T3 — pass.
- **Success criteria**:
  - Insert 3 chunks via `SqliteStore.insert`, 2 with `tenant_id="alice"`, 1 with `tenant_id="bob"`.
  - `SqliteStore.search(..., tenant_id="alice")` returns only the 2 alice chunks.
  - The schema contains the `tenant_id` column.

---

## Item 12 — `PgVectorStore.search` passes `tenant_id` through (test)

- **File(s)**: `tests/test_pgvector_integration.py` (new file)
- **Change**: Add an integration test that confirms `PgVectorStore.search(..., tenant_id="alice")` builds the correct WHERE clause. Test is gated by `RAG_TEST_PGVECTOR_DSN`.
- **Test**: `tests/test_pgvector_integration.py::test_pgvector_search_filters_by_tenant_id`.
- **Acceptance criteria**:
  - T1, T3 — pass.
  - Test skips cleanly when `RAG_TEST_PGVECTOR_DSN` is not set.
- **Success criteria**:
  - When `RAG_TEST_PGVECTOR_DSN` is set and a Postgres service runs in CI, the test passes.
  - Without the env var, `pytest tests/test_pgvector_integration.py` reports skip, not error.

---

## Item 13 — `SchemaPerTenant.ensure_schema` full implementation

- **File(s)**: `raghub/tenants/isolation.py`, `tests/test_tenants_isolation.py`
- **Change**: Replace `NotImplementedError` with: open asyncpg connection; `CREATE SCHEMA IF NOT EXISTS tenant_<id>`; `SET search_path TO tenant_<id>, public`; execute `SCHEMA_SQL`.
- **Test**: `tests/test_tenants_isolation.py::test_schema_per_tenant_creates_schema` (gated).
- **Acceptance criteria**:
  - R9 — assertion on schema existence.
  - T1, T3 — pass.
- **Success criteria**:
  - With `RAG_TEST_PGVECTOR_DSN` set, calling `SchemaPerTenant(dsn).ensure_schema("alice")` results in the `tenant_alice` schema existing and containing the `raghub_chunks` table.

---

## Item 14 — `DatabasePerTenant.connection_for` full implementation

- **File(s)**: `raghub/tenants/isolation.py`, `tests/test_tenants_isolation.py`
- **Change**: Replace `NotImplementedError` with asyncpg connection open against the tenant's DSN.
- **Test**: `tests/test_tenants_isolation.py::test_database_per_tenant_opens_connection` (gated).
- **Acceptance criteria**:
  - R8 — connection can be verified via `verify()` (caller's responsibility; document in docstring).
  - T1, T3 — pass.
- **Success criteria**:
  - With `RAG_TEST_PGVECTOR_DSN` set, `DatabasePerTenant(registry).connection_for("alice")` returns an open connection.

---

## Item 15 — `migrate_tenant_split.row_to_schema` implementation

- **File(s)**: `raghub/tenants/isolation.py`, `tests/test_tenants_isolation.py`
- **Change**: Implement `_migrate_row_to_schema` using `INSERT INTO tenant_<id>.raghub_chunks SELECT * FROM public.raghub_chunks WHERE tenant_id = ?`; validate counts match.
- **Test**: `tests/test_tenants_isolation.py::test_migrate_row_to_schema_round_trip` (gated).
- **Acceptance criteria**:
  - R9 — assertion on row count after migration.
  - T1, T3 — pass.
- **Success criteria**:
  - Source has 10 rows for tenant `alice`; after migration, the `tenant_alice` schema has 10 rows; the public table has 0 rows for tenant `alice`.

---

## Tier 2 acceptance gate

- `grep -n "NotImplementedError" raghub/tenants/isolation.py` returns empty.
- `RowLevel.filter_query` is wired into `MemoryStore.search` and `SqliteStore.search`.
- All gated tests skip cleanly without Postgres.
