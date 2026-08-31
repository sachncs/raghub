> ⚠️ **ARCHIVED** — This document describes the **raghub 0.9.x Python
> release**, preserved for historical reference in
> [`archive/`](../../archive/). It does **not** describe the active
> **Revex** TypeScript codebase.
>
> For current documentation, see [`README.md`](../../README.md) and the
> TypeScript source under `packages/`.

# Queue adapter

Revex ships a built-in persistent queue for durable document ingestion.
The default backend is **SQLite**; Celery and Arq adapters are available
as alternatives.

See [ADR 0015](../adr/0015-persistent-queue.md) for the rationale.

## Quick start

```python
from raghub.queue import SqliteQueue, Job, JobStatus

queue = SqliteQueue(path="queue.sqlite")
job = queue.enqueue({"document_id": "doc-1", "action": "ingest"})
print(job.id, job.status)  # → uuid, JobStatus.PENDING
```

## Classes

### `SqliteQueue`

| Parameter | Default | Description |
|-----------|---------|-------------|
| `path` | `"raghub_queue.sqlite"` | Path to the SQLite database file |
| `max_retries` | `3` | Maximum retry attempts per job |
| `visibility_timeout` | `600` | Seconds before a running job is requeued |

**Methods:**

- `enqueue(payload, *, idempotency_key=None) → Job` — insert a job.
- `dequeue() → Job | None` — pop the next pending job (sets state to RUNNING).
- `complete(job_id) → None` — mark a job as COMPLETED.
- `fail(job_id, *, error=None) → None` — mark a job as FAILED.
- `requeue_stale() → int` — requeue jobs that exceeded the visibility timeout.
- `list_jobs(state=None) → list[Job]` — query jobs by state.

### `Job`

Dataclass with fields: `id`, `payload`, `state`, `created_at`,
`updated_at`, `attempts`, `max_retries`, `idempotency_key`.

### `JobStatus`

Enum: `PENDING`, `RUNNING`, `COMPLETED`, `FAILED`.

## CLI commands

```bash
revex queue list [--state pending|running|completed|failed]
revex queue retry <job_id>
revex queue purge [--state failed]
```

## Idempotency

Passing the same `idempotency_key` twice will return the existing job
instead of creating a duplicate. The key is stored in the `jobs` table
with a UNIQUE constraint.

## Celery / Arq backends

For teams already running Celery or Arq:

```python
from raghub.queue import CeleryTaskQueue  # or ArqTaskQueue

queue = CeleryTaskQueue(app=celery_app)
queue.enqueue({"document_id": "doc-1", "action": "ingest"})
```

These adapters implement the same `Queue` protocol as `SqliteQueue`.
