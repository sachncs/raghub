# Tier 1 — Wire v0.7.x collaborators into `RAG.__init__` (Items 1-7)

The v0.7.x plan shipped new `Settings` blocks, new collaborator types,
and new accessors — but never wired the collaborators into
`RAG.__init__`. A user calling `RAG()` gets `queue() == None`,
`feedback_store() == None`, etc. Tier 1 fixes that.

Items are **one PR each**. File paths are relative to the repo root.

---

## Item 1 — `Settings.queue` block

- **File(s)**: `raghub/config.py`, `tests/test_config_validation.py`
- **Change**: Add `QueueConfig` BaseModel with `backend: Literal["memory", "sqlite"] = "memory"`, `db_path: Path | None = None`, `max_inflight: int = 256`. Add `queue: QueueConfig` field to `Settings`. Add env-var parser in `load_env`.
- **Test**: `tests/test_config_validation.py::test_queue_config_defaults` and `::test_queue_config_env_override`.
- **Acceptance criteria**:
  - N1 — `QueueConfig` is a single-word class (R3).
  - N4 — `MAX_INFLIGHT_DEFAULT` is a `UPPER_CASE` constant (named in `raghub/constants.py`).
  - T1 — `mypy --strict raghub/` passes.
  - T2 — Coverage gate at 70% (CI) passes.
  - T3 — `ruff check` zero errors.
- **Success criteria**:
  - `Settings().queue` returns the new block.
  - `os.environ["RAG_QUEUE_BACKEND"] = "sqlite"; Settings.load().queue.backend == "sqlite"` parses correctly.
  - `os.environ["RAG_QUEUE_MAX_INFLIGHT"] = "512"; Settings.load().queue.max_inflight == 512` parses correctly.

---

## Item 2 — `Settings.feedback` block

- **File(s)**: `raghub/config.py`, `tests/test_config_validation.py`
- **Change**: Add `FeedbackConfig` with `backend: Literal["sqlite", "postgres", "none"] = "none"`, `db_path: Path | None = None`, `dsn: str | None = None`. Wire env vars `RAG_FEEDBACK_BACKEND`, `RAG_FEEDBACK_DSN`.
- **Test**: `tests/test_config_validation.py::test_feedback_config_defaults` and `::test_feedback_config_env_override`.
- **Acceptance criteria**:
  - R7 — `Literal["sqlite", "postgres", "none"]` has 3 values.
  - T1, T3 — pass.
- **Success criteria**:
  - `Settings().feedback.backend == "none"` by default.
  - `os.environ["RAG_FEEDBACK_DSN"] = "postgres://..."; Settings.load().feedback.dsn` parses correctly.

---

## Item 3 — `Settings.rate_limit` block

- **File(s)**: `raghub/config.py`, `tests/test_config_validation.py`
- **Change**: Add `RateLimitConfig` with `backend: Literal["memory", "sqlite"] = "memory"`, `per_tenant_rps: float = 10.0`, `per_tenant_burst: int = 20`, `per_user_rps: float = 5.0`, `per_user_burst: int = 10`, `exempt_tenants: list[str] = []`. Wire env vars.
- **Test**: `tests/test_config_validation.py::test_rate_limit_config_defaults` and `::test_rate_limit_config_env_override`.
- **Acceptance criteria**:
  - O1 — `RATE_LIMIT_RPS`, `RATE_LIMIT_BURST` constants defined in `raghub/constants.py`.
  - T1, T3 — pass.
- **Success criteria**:
  - `Settings().rate_limit.per_tenant_rps == 10.0` (matches constant).
  - `RAG_RATE_LIMIT_EXEMPT_TENANTS="acme,beta"` parses to `["acme", "beta"]`.

---

## Item 4 — `Settings.archive` block

- **File(s)**: `raghub/config.py`, `tests/test_config_validation.py`
- **Change**: Add `ArchiveConfig` with `backend: Literal["local", "none"] = "none"`, `local_dir: Path = Path("./data/archives")`. Wire env vars `RAG_ARCHIVE_BACKEND`, `RAG_ARCHIVE_DIR`.
- **Test**: `tests/test_config_validation.py::test_archive_config_defaults` and `::test_archive_config_env_override`.
- **Acceptance criteria**:
  - O1 — `DEFAULT_ARCHIVE_DIR` constant defined in `raghub/constants.py`.
  - T1, T3 — pass.
- **Success criteria**:
  - `Settings().archive.backend == "none"` by default.
  - `RAG_ARCHIVE_DIR="/tmp/archives"` parses to `Path("/tmp/archives")`.

---

## Item 5 — `Settings.tenants` block

- **File(s)**: `raghub/config.py`, `tests/test_config_validation.py`
- **Change**: Add `TenantsConfig` with `resolver: Literal["none", "header", "jwt", "composite"] = "none"`, `isolation: Literal["row_level", "schema_per_tenant", "database_per_tenant"] = "row_level"`. Wire env vars `RAG_TENANTS_RESOLVER`, `RAG_TENANTS_ISOLATION`.
- **Test**: `tests/test_config_validation.py::test_tenants_config_defaults` and `::test_tenants_config_env_override`.
- **Acceptance criteria**:
  - R7 — `Literal["none", "header", "jwt", "composite"]` has 4 values.
  - T1, T3 — pass.
- **Success criteria**:
  - `Settings().tenants.isolation == "row_level"` by default.
  - `RAG_TENANTS_ISOLATION="schema_per_tenant"` parses correctly.

---

## Item 6 — `RAG.__init__` constructs `SqliteQueue`

- **File(s)**: `raghub/rag.py`, `tests/test_rag_facade.py`
- **Change**: When `Settings.queue.backend == "sqlite"`, instantiate `SqliteQueue(settings.data_dir / "queue.db", max_inflight=settings.queue.max_inflight)` and assign to `self.queue_`. Skip when `backend == "memory"` (no-op).
- **Test**: `tests/test_rag_facade.py::test_rag_constructs_sqlite_queue_from_settings`.
- **Acceptance criteria**:
  - R4 — no back-compat aliases.
  - C1 — `__init__` stays ≤ 30 LOC after this PR (extract helpers in item 34).
  - T1, T3 — pass.
- **Success criteria**:
  - `RAG(Settings(queue=QueueConfig(backend="sqlite"))).queue_` is a `SqliteQueue` instance.
  - `RAG().queue_` is `None` (default `backend == "memory"`).
  - `RAG(Settings(queue=QueueConfig(backend="sqlite"))).queue()` returns the `SqliteQueue` instance.

---

## Item 7 — `RAG.__init__` constructs `CompositeTenantResolver`

- **File(s)**: `raghub/rag.py`, `tests/test_rag_facade.py`
- **Change**: When `Settings.tenants.resolver in ("header", "jwt", "composite")`, instantiate the corresponding `TenantResolver` (default: `CompositeTenantResolver()`) and assign to `self.tenant_resolver_`. Skip when `resolver == "none"`.
- **Test**: `tests/test_rag_facade.py::test_rag_constructs_tenant_resolver_from_settings`.
- **Acceptance criteria**:
  - R8 — `CompositeTenantResolver.resolve` is the contract; constructor delegates to it.
  - T1, T3 — pass.
- **Success criteria**:
  - `RAG(Settings(tenants=TenantsConfig(resolver="composite"))).tenant_resolver_` is `CompositeTenantResolver`.
  - `RAG(Settings(tenants=TenantsConfig(resolver="jwt"))).tenant_resolver_` is `JwtClaimTenantResolver`.
  - `RAG().tenant_resolver_` is `None`.

---

## Tier 1 acceptance gate

When items 1-7 land:

- `RAG()` (with default `Settings`) returns `None` for every accessor — no change.
- `RAG(Settings(queue=QueueConfig(backend="sqlite")))` returns a working `SqliteQueue`.
- `RAG(Settings(tenants=TenantsConfig(resolver="composite")))` returns a working `TenantResolver`.
- `mypy --strict raghub/` passes.
- `pytest -q --no-cov` passes.
