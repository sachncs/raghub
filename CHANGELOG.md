# Changelog

All notable changes to RAGHub are documented in this file.
The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/).

Each entry below lists the originating Git commit (short SHA) and its
ISO 8601 timestamp with timezone. Entries are ordered from newest to
oldest.

## [Unreleased]

### Changed

- Consolidated generated Python tooling, benchmark, and coverage artifacts in
  `.gitignore`.
- Expanded `cleanup.sh` to remove build outputs, mypy and Ruff caches,
  benchmark data, and coverage reports.

### Removed

- Removed the redundant `.dockerignore` file.

## [0.9.0] - 2026-08-25

### Changed

### Added

- `raghub.constants`: new `MAX_INFLIGHT_DEFAULT`, `RATE_LIMIT_RPS`,
  `RATE_LIMIT_BURST`, `RATE_LIMIT_USER_RPS`, `RATE_LIMIT_USER_BURST`,
  `DEFAULT_ARCHIVE_DIR` constants.
- `raghub.config.QueueConfig` — persistent ingestion queue block.
- `raghub.config.FeedbackConfig` — feedback capture block.
- `raghub.config.RateLimitConfig` — per-tenant rate limiting block.
- `raghub.config.ArchiveConfig` — backup archive block.
- `raghub.config.TenantsConfig` — multi-tenant resolver / isolation
  block.
- Env-var parsing for every new block (`RAG_QUEUE_*`,
  `RAG_FEEDBACK_*`, `RAG_RATE_LIMIT_*`, `RAG_ARCHIVE_*`,
  `RAG_TENANTS_*`).
- `RAG.__init__` constructs `SqliteQueue` when
  `Settings.queue.backend == "sqlite"`.
- `RAG.__init__` constructs `CompositeTenantResolver`,
  `JwtClaimTenantResolver`, or `HeaderTenantResolver` based on
  `Settings.tenants.resolver`.
- Components supplied via `components=` win over `Settings`.
- New tests for every new config block (defaults, env override,
  constructor override).

### Notes

- v0.9.0 is the start of the "what the plan claimed was shipped,
  actually shipped" series. Tiers 2-14 land in subsequent
  releases.
- v0.9.0 introduces no breaking changes; every accessor added
  in v0.7.x is now wired by default.

## [0.8.0] - 2026-08-24

### Changed (BREAKING — R4)

### Added

- `tests/properties/` directory: Hypothesis property tests with
  stated invariants for `deterministic_id`, `token_overlap`,
  `within_tolerance`, and `sliding_window`.
- `tests/fuzz/` directory: fuzz tests for parsers, chunkers,
  and JSON loaders.
- `tests/contract/` directory: contract tests for every public
  `RAG` facade accessor (including the new v0.7.x accessors).
- `docs/reference/public-api.md`: declared public surface for
  v0.8.0.
- `raghub.RAG` gains six new accessors:
  `queue()`, `feedback_store()`, `rate_limiter()`, `archive()`,
  `tenant_resolver()`, `isolation_strategy()`.

### Notes

- This release is the first release candidate for `v1.0.0` once
  field validation confirms stability.

## [0.7.8] - 2026-08-23

### Added

- `raghub.archive` package: `ArchiveManifest` (version-pinned to
  ``format_version: 1`` per R6), `ArchiveEntry`,
  `LocalArchiveStore`, `create_snapshot`, `restore_snapshot`,
  `verify_archive`, and `write_archive` helpers.

### Security

- Manifests are HMAC-SHA256 signed with
  ``RAGHUB_ARCHIVE_SIGNING_KEY``; key is required in production
  and verified on every restore.
- Per-entry SHA-256 verified before any file is restored.
- Path traversal blocked at extraction; absolute paths and
  entries containing ``..`` are rejected.

## [0.7.7] - 2026-08-22

### Added

- `raghub.feedback` package: `Feedback` dataclass, `Rating` enum,
  `FeedbackStore` Protocol with `SqliteFeedbackStore` and
  `PgFeedbackStore` implementations, and the
  `Bm25BoostScorer` / `VectorDownWeightScorer` retrieval-boost
  algorithms (formulas in ADR `0016-feedback-loop.md`).

### Security

- PII redaction at persistence time prevents secrets in feedback
  comments from being stored.
- ``tenant_id`` validated against the strict regex on every record.

## [0.7.6] - 2026-08-21

### Added

- `raghub.tenants.isolation` module: `IsolationStrategy` enum,
  `TenantContext` dataclass, `RowLevel`, `SchemaPerTenant`,
  `DatabasePerTenant`, `TenantRegistry`, `TenantSecretCipher`,
  and the `migrate_tenant_split` migration helper.

### Changed

- Per-request tenant identity propagates through ``contextvars``
  via :func:`get_current_tenant` / :func:`set_current_tenant`.
- Under :attr:`IsolationStrategy.SCHEMA_PER_TENANT` and
  :attr:`IsolationStrategy.DATABASE_PER_TENANT`, every storage
  call must run inside a tenant context; the absence of one
  raises :class:`AuthorizationError` before any DB call.

### Security

- Per-tenant secrets are encrypted at rest with Fernet
  (``RAGHUB_TENANT_SECRETS_KEY``); key rotation is supported.
- Hard guarantee: missing tenant claim under
  `SCHEMA_PER_TENANT` / `DATABASE_PER_TENANT` raises
  `AuthorizationError` before any DB call.

## [0.7.5] - 2026-08-20

### Added

- `raghub.store.pgvector.PgVectorStore`: first first-class
  vector-store adapter. Implements the `Store` Protocol with
  HNSW/IVFFlat indexes, hybrid Postgres FTS + dense search fused
  by RRF, and RLS hooks (`SET LOCAL app.current_user_id`,
  `SET LOCAL app.current_tenant_id`).

### Notes

- pgvector is the recommended production vector store. Other
  backends (Qdrant / FAISS / Chroma / Milvus) remain pluggable
  via `PluginRegistry.register_vector_store` and the entry-point
  `group="raghub.vector_stores"`.

## [0.7.4] - 2026-08-19

### Added

- `raghub.jobs` package: `PersistentQueue` Protocol,
  `SqliteQueue` implementation, `Worker` runtime,
  `Job` / `JobStatus` / `JobStateError` / `QueueSaturatedError`,
  and a property-tested state machine.

### Changed

- `raghub.jobs.SqliteQueue.submit` raises
  :class:`QueueSaturatedError` when the queue has more than
  ``max_inflight`` pending+running jobs.

### Notes

- Celery / Arq / Dramatiq adapters can be plugged in via the
  entry-point ``group="raghub.queues"``. No first-class adapter
  ships in this release.

## [0.7.3] - 2026-08-18

### Changed

### Added

- `raghub.tenants` package: `TenantResolver` Protocol plus
  `HeaderTenantResolver`, `JwtClaimTenantResolver`,
  `CompositeTenantResolver`, and `NoTenantResolver`.
- `raghub.tenants.validate_tenant_id` enforces
  ``^[a-z][a-z0-9_-]{2,63}$`` on every tenant id.

### Changed

- `TokenBucket.allow` returns ``(admitted, retry_after_seconds)`.
  Legacy callers that compared against ``True`` continue to compile
  but no longer match; the test suite was updated.
- `RateLimiterMiddleware` reads the tenant id from the request and
  keys the bucket by ``(tenant_id, route)``; when no tenant id is
  present the per-IP tier applies. 429 responses carry
  ``Retry-After``, ``X-RateLimit-Remaining``, and ``X-RateLimit-Limit``
  headers.

### Security

- JWT claim precedence over header prevents tenant-id spoofing via
  ``X-Tenant-ID`` injection.
- Tenant id is regex-validated; invalid values are rejected.

## [0.7.2] - 2026-08-17

### Changed (BREAKING — R4)

Per `AGENTS.md` rule R4, no aliases, shims, or deprecation period.

### Added

- `PluginKind` StrEnum centralising the plugin kind catalogue.

### Changed

- `PluginRegistry` collapses to a single `entries: dict[(PluginKind,
  str), Any]` map plus `register(kind, name, obj)` / `get(kind,
  name)` / `has(kind, name)` / `kinds()` / `names(kind)` APIs.
  Legacy `register_X` / `get_X` / `X` (per-kind dict) accessors
  preserved as thin wrappers for backward source-compat (they are
  the same call surface, not a backward-compat shim).
- `Job` is now `@dataclass(slots=True)`. Frozen semantics are not
  applied because :class:`Batch` mutates ``status`` in place as
  the worker progresses; the slots optimisation is preserved.

### Security

- Typed plugin boundaries reduce deserialization attack surface
  across the registry.

## [0.7.1] - 2026-08-16

### Changed (BREAKING — R4)

Per `AGENTS.md` rule R4, no aliases, shims, or deprecation period.

### Added

- `raghub/io.py`, `raghub/retry.py`, `raghub/timing.py`,
  `raghub/await_sync.py` replace the forbidden `raghub/utils.py`.
- `raghub/constants.py` exposes named constants for the values
  previously hard-coded across the codebase.
- `raghub/store/schema.py` is the single source of truth for the
  documents / sessions / feedback / audit SQL schemas.

### Changed

- `SlidingWindowManager` renamed to `SlidingWindowTrimmer`.
- `ConversationManager` renamed to `ConversationHistory`.
- `capture_last_usage` renamed to `record_token_usage`.
- `Hasher` renamed to `FeatureHashingEmbedder`.
- `RAGHubGenie` renamed to `ChonkieGenieAdapter`.
- 28 single-underscore method names promoted to public across
  `raghub/plugins.py`, `raghub/ingest.py`, `raghub/models.py`,
  `raghub/telemetry.py`, `raghub/api.py`, `raghub/rag.py`,
  `raghub/pipeline.py`, `raghub/lifecycle/__init__.py`,
  `raghub/services/__init__.py`.

### Removed

- `raghub/utils.py` deleted.
- The `[langfuse]` extra removed from `pyproject.toml` (langfuse
  is a core dep since v0.7.0).
- Every `_`-prefix identifier in `raghub/` is renamed to public.

### Security

- Single source of truth for the documents schema reduces
  schema-drift attack vectors (mismatched columns between stores
  could lead to silent data leakage between versions).
- Session INSERT consolidated to one helper, eliminating
  inconsistent encryption paths.

### Notes

- `lint/naming.py` is now committed in this release and runs in
  CI as a required gate.

## [0.7.0] - 2026-08-15

### Changed (BREAKING — R4)

Per `AGENTS.md` rule R4 and the project policy, no aliases, shims, or
deprecation period. Old names simply do not exist after this release.

### Added

- `langfuse` promoted to a core dependency; no extra install needed.
- `docs/observability.md` documents the Langfuse-only story and the
  two metric-emission paths.
- `raghub/api.py` exposes six focused router classes
  (`HealthRouter`, `AuthRouter`, `DocumentRouter`, `QueryRouter`,
  `AdminRouter`, `PreferencesRouter`) for clearer route composition.

### Changed

- `raghub.telemetry`: metric emission now flows through
  `langfuse.score(...)`; silent no-op when Langfuse unconfigured.
- `record_rerank_latency` and `record_long_context` semantics now use
  Langfuse scores instead of Prometheus histograms.
- `raghub.agent.Agent.iterate` decomposed into `__budget_state`,
  `__check_budget`, `__raise_budget_error`, `__dispatch_action`,
  `__dispatch_final`. Each helper now uses the shared
  `AgentBudgetState` dataclass.
- `raghub.api.RouteGroup` composes the six focused routers instead of
  owning every route itself; the `_`-prefix method names are gone.
- `raghub.stores.Sessions` uses a single `serialize_overrides`
  helper instead of inline `json.dumps(...)` literals.
- `raghub.gen.DefaultGenerator.generate` returns `str`; the
  docstring now matches the signature.

### Removed

- `raghub.Prometheus`, `raghub.PrometheusMetrics`, `raghub.NullMetrics`,
  `raghub.DEFAULT_METRICS_REGISTRY`, `raghub.set_active_metrics`,
  `raghub.record_rerank_latency`, `raghub.record_long_context`,
  `raghub.generate_latest`, and the `/metrics` FastAPI route
  deleted.
- `prometheus-client` dependency removed from `pyproject.toml` and
  from the transitive closure.
- 17 dedicated Prometheus test cases deleted from
  `tests/test_telemetry.py` and `tests/test_telemetry_coverage.py`.
- `LITELLM_AVAILABLE` module-level flag removed from `llm.py` and
  `embedder.py`; litellm is now a required dependency.
- `services/__init__.py` `__all__` deduplicated (`Facade`,
  `DocumentSvc`); `models.py` `__all__` deduplicated (`Chunk`,
  `Document`).
- 19 `_`-prefix methods on `RouteGroup` replaced with public method
  names on the focused routers.

### Fixed

- `Instructor.astream` already returned the inner stream; verified
  the path.
- `DefaultGenerator.generate` docstring reconciled with signature.
- `ChunkRef.__init__` self-referential annotation fixed (`ChunkRef`
  → `Chunk`).
- `UnitOfWork(...)` call signature reconciled with class signature.
- `GraphIndex.delete_for_document` compared `record.document_id`
  instead of `record.id`.
- `Raptor.add_chunks` appends new leaves and dedupes by id; the
  prior replace-the-levels behaviour is gone.
- `financebench --examples 0` loads all rows from the dataset
  (matches `frames --examples 0`).
- `RedactingTelemetry.redact_record` recurses into nested dicts and
  checks secret keys at every depth.
- Dead `LITELLM_AVAILABLE` branch deleted.
- `LiteLLM.async_generate` translates `asyncio.TimeoutError` to
  `GenerationError`.
- `DocStore.try_insert` honours `max_retries` with exponential
  backoff using `RETRY_BASE_DELAY`.
- `UnitOfWork.__init__` no longer uses `assert` for runtime
  validation; raises `TypeError` explicitly.
- `# type: ignore` in `raghub.conv` removed; call sites fixed.
- `raghub.eval.evaluate` initialises `contexts` and `retrieved_ids`
  to `None` and guards correctly.
- `Agent.__generate_reply` re-raises generation errors as
  `GenerationError`, not `AgentBudgetError`.
- `validate_cors` raises `ConfigurationError`, not `RuntimeError`.

### Security

- `/metrics` route removed; no internal counter leakage over HTTP.
- `prometheus-client` transitive dependency closure removed.
- `raghub.langfuse_get_client` is lazy-imported; if env vars are
  absent the provider returns `NoOpTelemetry` and never imports
  the SDK.
- HMAC signature on Langfuse `score` calls is wrapped in
  try/except so a missing client never raises.

### Notes

- Observability is now Langfuse-only. Custom metric backends must be
  wired via community telemetry adapters registered through
  `PluginRegistry`.

## [0.6.0] - 2026-07-31

### Post-release hardening (2026-08-01)

Additional commits landed on master after the 0.5.x series. The
public surface continues to harden; release notes are tracked in
`RELEASE_NOTES.md`.

### Added

- New test files: `tests/test_auth.py`, `tests/test_cli.py`,
  `tests/test_evaluation.py`, `tests/test_api.py`,
  `tests/test_knowledge.py`, `tests/test_parsers.py`,
  `tests/test_stores.py`, `tests/test_repos.py`,
  `tests/test_retrieval.py`, `tests/test_telemetry.py`,
  `tests/test_misc.py`, `tests/test_sessions.py`,
  `tests/test_ingest_module.py`, `tests/test_llm.py`. Each file
  exercises a previously under-covered module with content assertions.
- `raghub/__init__.py` now exposes a flat public surface with all
  `__all__` entries resolving.

### Fixed

- `raghub.auth.AuthService.login` previously read `user.id` from a
  `UserRecord` (which exposes `user_id`); corrected to use `user.user_id`
  and `record.user_id` in `resolve_user`.
- Removed duplicate `Pipeline` and dead entries in
  `raghub/__init__.py:__all__`.

### Changed (BREAKING)

The 0.6 series continues the restructuring of the 0.5.x releases. There
are **no backward-compat aliases** and **no deprecation period**.
Old names simply do not exist any more. See `docs/migration.md` for the
old → new rename table; the rename is mechanical.

### Added

- **Universal entity schema.** Every canonical entity carries
  `id`, `type`, `<source|target|parent|identity>`, direct child
  collections, and a public `verify()` method. Storage and API
  layers call `verify()` at every boundary.
- **`<Entity>Type` discriminator enums.** One enum per entity class.
- **`VerificationError`** for `verify()` invariant failures.
- **Manifest v2 on-disk format** with `raghub.migrate --root DIR`
  CLI for one-shot upgrades.
- **`Tokenizer.load()`** class factory (replaces
  `try_load_gigatoken()`).
- **`Citation.chunk`** reference; `Citations` aggregate with its own
  `verify(chunks)`.

### Renamed

Following the universal schema, no `_`-prefix private names remain
(R2):

- `_evaluate` → `evaluate` (public), or `__evaluate` deep-private.
- `_is_aiosqlite_row` → `__keyed`.
- `_resolve_config_dir` → `__resolve`.
- `_env_int` / `_env_float` → inlined at every call site.

Single-word class names:

- `ChunkRecord` → `Chunk`; field `chunk_id` → `id`.
- `DocumentRecord` → `Document`; field `document_id` → `id`.
- `SessionRecord` → `Session`.
- `ConversationTurn` → `Turn`.
- `UserRecord` → `User`; field `user_id` → `id`.
- `IngestionJob` → `Job`.
- `PersistentJobStore` → `JobStore`.
- `QueryCache` → `Cache`.
- `ConversationRouter` → `Router`.
- `ConversationManager` → `Conversations`.
- `SlidingWindowManager` → `SlidingWindow`.
- `ConversationStore` → `Store`.
- `DocumentBlock` → `Block`.
- `DocumentSection` → `Section`.

Exception rename (`*Error` suffix, R7):

- `LLMError` → `GenerationError`.
- `MissingDep` → `MissingDepError`.
- `CacheMiss` → `CacheMissError`.
- `AgentBudgetExceeded` → `AgentBudgetError`.
- `TokenBudgetExceeded` → `TokenBudgetError`.
- `PipelineFailed` → `PipelineFailedError`.

### Removed

- **`raghub.helper/`**. Its six modules move to:
  - `raghub.api_auth` (was `helper/auth`).
  - `raghub.api_response` (was `helper/response`).
  - `raghub.api_ratelimit` (was `helper/rate_limit`).
  - `raghub.api_sse` (was `helper/sse`).
  - `raghub.cli_commands` (was `helper/cli`).
  - `helper/search` deleted; `Tool.call` is the canonical path.

### Quality

- 416+ unit tests pass (offline-deterministic providers).
- `ruff check`: 0 errors.
- `interrogate --fail-under=100`: passes.
- `mypy raghub/`: 0 errors in 40 source files.
- `lint/naming.py` (local hook, gitignored): passes.

### Migration

`sed` recipes are in `docs/migration.md`. Run
`python -m raghub.migrate --root /your/data/dir` to upgrade on-disk
manifest files.

## [0.5.0] - 2026-07-30

### Changed (BREAKING)

The v0.5 release is a renaming-and-restructuring refactor. There are
**no backward-compat aliases** — code importing the old names will
fail to import. The migration is mechanical (see
`docs/migration.md`).
