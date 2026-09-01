# Workers & jobs

Revex processes background work (notably document ingestion) through a
persistent job queue and a set of workers.

## Job queue

`@revex/core` defines the `JobQueue` interface:

```ts
interface JobQueue {
  enqueue(job): Promise<JobRecord>;
  claim(): Promise<JobRecord | null>;
  complete(id, result): Promise<void>;
  fail(id, error): Promise<void>;
}
```

Implementations:

- `SqliteJobQueue` — persistent SQLite-backed queue (the production path).
- `MemoryQueue` — in-memory queue for tests.

`JobStatus`: `pending | running | done | failed`. `JobRecord` carries the
payload, attempts, and status. `QueueBaseRegistry` keys queues by kind.

## JobWorker

`@revex/api` ships `JobWorker` (`packages/api/src/job-worker.ts`):

- Polls the queue (default 500ms), processes in batches (default 4).
- Tracks stuck jobs; `resetStuckJobs()` re-queues any `running` rows so a fresh
  process can resume them.

## WorkspaceWorkerSupervisor

`WorkspaceWorkerSupervisor` scans the workspace registry every `pollMs`
(default 2s) and starts one `JobWorker` per registered workspace. The
`resolveHandler(workspaceId)` maps a workspace to its ingest handler
(`documentIngestHandler` in `packages/api/src/handlers/document-ingest.ts`).

## Worker roles

`REVEX_WORKER_ROLE` controls which API process runs workers:

| Role | Behavior |
|---|---|
| `leader` (default) | Runs the supervisor + workers. |
| `follower` | Serves HTTP only; supervisor disabled. |
| `disabled` | No worker at all (read-only API). |

`REVEX_RESET_STUCK_JOBS=1` re-queues stale `running` jobs on boot.

## Document ingestion job

`POST /v1/documents` (and `/v1/documents/ingest-stream`) persists bytes to
`LocalFileStorage`, enqueues a `document.ingest` job, and returns
`202 {status:'pending'}`. The worker drains the job, runs `ingest()`, flips the
document row to `ready`/`failed`, and writes an `ingest.complete` /
`ingest.failure` audit event.

## CLI

`revex queue list | stats | purge | submit` inspects the queue
(see [CLI reference](../reference/cli.md)).

## Supervision

The API's `start()` opens the registry, instantiates the supervisor, and
registers the first workspace. On `SIGINT`/`SIGTERM` it stops the supervisor
and closes the workspace pool so in-flight jobs drain before exit.