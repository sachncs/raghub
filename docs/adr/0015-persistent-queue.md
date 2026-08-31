# ADR 0015 — Persistent queue

## Status

Accepted (v0.7.4)

## Context

The ingestion pipeline needs a durable queue so that document processing
survives process restarts and can be inspected, retried, and scaled
independently of the HTTP layer.

Early versions used an in-memory `asyncio.Queue`, which lost jobs on
crash and offered no visibility into backlog depth.

## Decision

We use **SQLite as the default durable queue** via the `SqliteQueue`
class, with well-defined entry-point contracts (`CeleryTaskQueue` and
`ArqTaskQueue`) for deployments that already run Celery or Arq.

Key properties of `SqliteQueue`:

- **Schema**: Single `jobs` table with columns `id`, `payload`,
  `state` (pending | running | completed | failed), `created_at`,
  `updated_at`, `attempts`, `max_retries`.
- **Idempotency**: Jobs carry a caller-supplied `idempotency_key`;
  duplicate inserts are silently ignored.
- **Visibility timeout**: Running jobs that exceed the timeout are
  automatically requeued.
- **CLI**: `revex queue list`, `revex queue retry`, `revex queue purge`.

## Consequences

- **Operational simplicity**: SQLite requires no external service; the
  queue lives in the same file as the application database or in a
  dedicated WAL-mode file.
- **Community flexibility**: Celery and Arq adapters allow teams with
  existing task-queue infrastructure to plug in without code changes.
- **Single-writer limitation**: SQLite serialises writes; for
  high-throughput workloads (≥ 1 000 jobs/s) teams should migrate to
  the Celery or Arq backends.
- **Backup**: The queue file should be included in archive backups
  (see ADR 0017).

## Alternatives considered

- **Redis**: Fast but adds a dependency; no built-in job history.
- **RabbitMQ**: Full-featured but operationally heavy for small
  deployments.
- **In-memory queue only**: Rejected — no durability guarantee.
