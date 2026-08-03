# RAGHub Public API Surface

This document declares every public symbol in `raghub` as of v0.8.0.
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
| `scoped_session_id` | 0.6.0 | internal helper |
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
| `resolve_agent_config` | 0.6.0 | internal helper |

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
| `settings_serialise_path` | 0.6.0 | internal helper |

### Stable types

`raghub.User`, `raghub.Chunk`, `raghub.Document`, `raghub.Bundle`,
`raghub.Hit`, `raghub.SearchResponse`, `raghub.Response`,
`raghub.Query`, `raghub.Turn`, `raghub.Session`, `raghub.Citation`,
`raghub.Citations`, `raghub.Feedback`, `raghub.Job`, `raghub.Pipeline`,
`raghub.PipelineCtx`, `raghub.Result`, `raghub.ArchiveManifest`.

### Errors

`raghub.RagHubError` and the full hierarchy in `raghub.errors`.

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
* **Evolving** — internal helpers (`scoped_session_id`,
  `session_overrides`, `resolve_agent_config`,
  `settings_serialise_path`, `ingest_directory_sync`,
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
