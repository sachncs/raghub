# Revex Public API Surface

This document declares every public symbol in `raghub` as of v0.9.5.
Additive changes in later releases extend the list; breaking changes
require an ADR + SemVer bump.

## `raghub.RAG`

The single recommended facade.

### Construction

| Symbol | Signature | Version | Notes |
|---|---|---|---|
| `RAG` | `RAG(*, settings=None, components=None, **kwargs)` | 0.6.0 | Lazy-imports every collaborator |
| `RAG.from_config` | `classmethod(path: str \| Path) -> RAG` | 0.6.0 | YAML or TOML |

### Ingest

| Symbol | Version | Notes |
|---|---|---|
| `ingest` | 0.6.0 | sync |
| `aingest` | 0.6.0 | async |
| `ingest_async` | 0.7.4 | returns job id; routed through `SqliteQueue` when configured |
| `ingest_dir` | 0.6.0 | ProcessPoolExecutor path |
| `sync_index` | 0.6.0 | Manifest-driven reconciliation |
| `delete` | 0.6.0 | Retires prior bundle ids |

### Query

| Symbol | Version | Notes |
|---|---|---|
| `query` | 0.6.0 | sync |
| `aquery` | 0.6.0 | async |
| `astream` | 0.6.0 | Token streaming |
| `astream_agent` | 0.6.0 | PlannerEvent streaming |

### Conversation

| Symbol | Version | Notes |
|---|---|---|
| `conversation_history` | 0.6.0 | |
| `clear_conversation` | 0.6.0 | |
| `scoped` | 0.7.0 | internal helper (renamed from `scoped_session_id`) |
| `session_overrides` | 0.6.0 | internal helper |

### Evaluation

| Symbol | Version | Notes |
|---|---|---|
| `evaluate` | 0.6.0 | Benchmark harness |

### Lifecycle

| Symbol | Version | Notes |
|---|---|---|
| `initialize` | 0.6.0 | Opens every held resource |
| `shutdown` | 0.6.0 | Closes every held resource |
| `job_status` | 0.7.4 | Inspect a queued job |

### v0.7.x collaborator accessors

Each returns the configured component or `None` when not configured.

| Symbol | Version | Returns |
|---|---|---|
| `queue` | 0.7.4 | `PersistentQueue` |
| `feedback_store` | 0.7.7 | `FeedbackStore` |
| `rate_limiter` | 0.7.3 | `RateLimiter` |
| `archive` | 0.7.8 | `ArchiveStore` |
| `tenant_resolver` | 0.7.3 | `TenantResolver` |
| `isolation_strategy` | 0.7.6 | `IsolationStrategy` enum |

### Diagnostics

| Symbol | Version | Notes |
|---|---|---|
| `health` | 0.6.0 | Returns a dict |

### Stable types

`raghub.User`, `raghub.Chunk`, `raghub.Document`, `raghub.Bundle`,
`raghub.Hit`, `raghub.SearchResponse`, `raghub.Response`,
`raghub.Query`, `raghub.Turn`, `raghub.Session`, `raghub.Citation`,
`raghub.Citations`, `raghub.Feedback`, `raghub.Job`, `raghub.Pipeline`,
`raghub.Result`, `raghub.ArchiveManifest`.

### Errors

`raghub.RagHubError` and the full hierarchy in `raghub.errors`. All
framework-raised exceptions now subclass `RagHubError` (see PR-1
commit "fix(errors): re-parent 4 framework errors to RagHubError").

### Renamed in 0.9.x

| Old | New | Replaced by |
|---|---|---|
| `RAG.scoped_session_id` | `RAG.scoped` | 0.9.0 |
| `RAG.resolve_agent_config` | (removed) | 0.9.0 |
| `RAG.settings_serialise_path` | (removed) | 0.9.0 |
| `app_service` parameter | `application_facade` parameter | 0.9.x |
| `db_manager` parameter | `database_handle` parameter | 0.9.x |
| `lifecycle_manager` parameter | `lifecycle_coordinator` parameter | 0.9.x |
| `helper` module name | `support` module name | 0.9.x |
| `raghub.authhelpers` package | `raghub.auth_support` package | 0.9.x |
| `raghub.services.helpers` | `raghub.services.diagnostics` | 0.9.x |
| `raghub.pipeline.helpers` | `raghub.pipeline.span_support` | 0.9.x |
| `raghub.pipeline.builder` | `raghub.pipeline.pipeline_assembly` | 0.9.x |
| `raghub.services.facade.Facade` | `ApplicationFacade` (Facade kept as deprecation alias) | 0.9.x |
| `raghub.telemetry.LoguruLoggerAdapter` | `Logger` | 0.9.x |
| `raghub.telemetry.build_logger` | (removed) | 0.9.x |
| `raghub.Loguru` alias | `raghub.Logger` (LoguruLoggerAdapter) | 0.9.x |
| Enum members `UPPER_CASE` | PascalCase (e.g. `JobStatus.PENDING` -> `JobStatus.Pending`) | 0.9.x |

## Stable entry-point groups

| Group | Purpose | Version |
|---|---|---|
| `raghub.plugins` | Plugin discovery | 0.6.0 |
| `raghub.queues` | Persistent queue backends | 0.7.4 |
| `raghub.archives` | Backup / archive backends | 0.7.8 |
| `raghub.vector_stores` | Vector store adapters | 0.7.5 |
| `raghub.feedback` | Feedback scorers and stores | 0.7.7 |

## Stability levels

* **Stable** — every public symbol listed above.
* **Evolving** — internal helpers (`scoped`, `session_overrides`,
  `ingest_dir`, `sync_one`, `remove_prior`). Subject to change
  between minor versions.
* **Experimental** — anything in `raghub.archive`,
  `raghub.feedback`, `raghub.jobs`, `raghub.store.pgvector`,
  `raghub.tenants`. Add API can change between minor versions.

## No-freeze commitment

This document is a snapshot of the current public surface; new
symbols may be added in future releases without an ADR, but
breaking changes to existing symbols always require an ADR +
SemVer bump.