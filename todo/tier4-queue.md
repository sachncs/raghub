# Tier 4 — Make v0.7.4 ingestion queue actually used (Items 21-24)

The v0.7.4 plan shipped `SqliteQueue` and `Worker`. `RAG.ingest_async`
still uses the old `Resumable(Batch)` threadpool. Tier 4 routes
through the new queue when configured, and ships CLI commands.

---

## Item 21 — `RAG.ingest_async` submits to `SqliteQueue` when configured

- **File(s)**: `raghub/rag.py`, `tests/test_rag_facade.py`
- **Change**: When `self.queue_` is set, `RAG.ingest_async` calls `self.queue_.submit(kind="ingest", payload={...}, tenant_id=...)` and returns the queue's job id. SHA-256 idempotency: check `queue.list_for_tenant` for an existing job with the same content hash; if found, return that job id.
- **Test**: `tests/test_rag_facade.py::test_ingest_async_submits_to_queue_when_configured`, `::test_ingest_async_idempotent_returns_existing_job_id`.
- **Acceptance criteria**:
  - R4 — no back-compat; old `Resumable` path is removed only when queue is configured.
  - R9 — assertion on idempotent return value.
  - T1, T3 — pass.
- **Success criteria**:
  - `RAG(Settings(queue=QueueConfig(backend="sqlite"))).ingest_async(b"hello")` returns a queue job id (UUID string).
  - Second call with the same bytes returns the same job id (idempotent).
  - Without `queue_` configured, falls back to the legacy `Resumable` path (returns the same string format as before).

---

## Item 22 — `RAG.job_status` reads from queue when configured

- **File(s)**: `raghub/rag.py`, `tests/test_rag_facade.py`
- **Change**: When `self.queue_` is set, `RAG.job_status(job_id)` looks up via `self.queue_.stats()` (count by status) or by id. Returns `JobStatus` enum value as string.
- **Test**: `tests/test_rag_facade.py::test_job_status_reads_from_queue`.
- **Acceptance criteria**:
  - R4 — no shim.
  - T1, T3 — pass.
- **Success criteria**:
  - After `ingest_async`, `job_status(job_id)` returns `"pending"` (or actual status).
  - For an unknown job id, returns `None`.

---

## Item 23 — CLI `raghub queue list`

- **File(s)**: `raghub/cli_commands/__init__.py`, `tests/test_cli.py`
- **Change**: New `QueueCommand` class with `list` sub-command. Supports `--status`, `--tenant`, `--limit` flags.
- **Test**: `tests/test_cli.py::test_queue_list_runs`.
- **Acceptance criteria**:
  - R3 — `QueueCommand` is single-word.
  - T3 — pass.
- **Success criteria**:
  - `raghub queue list` exits 0 and prints the queue contents (or empty list).
  - `raghub queue list --status pending` filters correctly.

---

## Item 24 — CLI `raghub queue run | retry | purge`

- **File(s)**: `raghub/cli_commands/__init__.py`, `tests/test_cli.py`
- **Change**: Three more sub-commands on `QueueCommand`. `run --workers 4` starts a worker pool; `retry <job_id> --delay 0` resets a failed job; `purge --status succeeded` removes by status.
- **Test**: `tests/test_cli.py::test_queue_run_retry_purge`.
- **Acceptance criteria**:
  - T3 — pass.
  - Each sub-command exits 0 against a mock queue.
- **Success criteria**:
  - `raghub queue run --workers 2` starts a `Worker` (verified by side effect in mock).
  - `raghub queue retry <id>` transitions the job back to `pending`.
  - `raghub queue purge --status succeeded` deletes matching rows.

---

## Tier 4 acceptance gate

- `RAG().ingest_async(b"hello")` with queue configured returns a UUID-shaped job id (not the legacy `Batch` handle).
- `RAG().job_status(job_id)` returns a string from `JobStatus`.
- `raghub queue list | run | retry | purge` each exit 0.
