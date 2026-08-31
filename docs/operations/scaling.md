# Scaling

This page covers vertical and horizontal scaling of the
Revex process. The current default runs one uvicorn worker per
data directory; the guidance below covers the next two steps up.

## Vertical scaling

The single-process ceiling is set by LiteLLM and the embedder at
query time, and by Marker and the chunker at ingest time. The
limiting dimensions are:

| Resource | Rough budget per API replica |
|---|---|
| CPU | 4 cores (dominated by LiteLLM and Marker at ingest time) |
| Memory | 6 GiB (index memory grows with `RAG_EMBEDDING_DIM × vectors` for the SQLite vector store) |
| Disk | `RAG_DATA_DIR` must hold the registry, sessions, and vector store; size scales with document volume |

These are budgets, not limits; the application does not impose any
internal cap. Raise them in your orchestrator (Kubernetes
`resources.requests` / `limits`, systemd `MemoryMax=`, etc.).

## Horizontal scaling: the API

The API is stateless once it has finished loading. You can run more
than one replica, but they will all attempt to open the same SQLite
files (`registry.db`, `sessions.db`).

Two options:

1. **Move the vector store to PostgreSQL.** Run
   `revex migrate pgvector --dsn <dsn>` against the target
   database, point the API at it through `RAG_VECTORSTORE_DSN`, and
   run the API behind a load balancer. The document registry and
   session store still ride on the SQLite files in
   `RAG_DATA_DIR`; if those need to scale too, point them at the
   same PostgreSQL with `RAG_REGISTRY_DSN` /
   `RAG_SESSIONS_DSN` (the same DSN can be reused for both). This
   is the supported production path for more than one replica.

2. **Keep SQLite, but run a single API replica.** This is the
   default. If you need additional ingest throughput, raise the
   `BACKGROUND_INGEST_WORKERS` environment variable (default `2`)
   and run the API service with `replicas: 1`.

To bump the background ingestion pool, set
`BACKGROUND_INGEST_WORKERS` in the process environment before
invoking `raghub run`; the value is read once at startup.

## Horizontal scaling: the vector store

The SQLite vector store (`SqliteStore` in `raghub.store`) is a
single-process database. To scale, switch to the PostgreSQL +
pgvector backend (`PgVectorStore` in `raghub.stores.pgvector`) and
back it with a managed PostgreSQL instance — single-node for most
production deployments, a hot-standby replica for read scaling.

For sharded deployments, switch to a managed PostgreSQL cluster
(Patroni, Aurora, Cloud SQL HA, etc.) and point `RAG_VECTORSTORE_DSN`
at the cluster endpoint. Provision `vector_dim` matching your
embedder; `revex migrate pgvector` creates the schema and indexes
on first run.

## Autoscaling signals

The FastAPI surface exposes Prometheus metrics on `/metrics` when
the `prometheus_client` instrumentation is enabled in `app.state`.
Key signals:

| Metric | Use |
|---|---|
| `raghub_query_duration_ms` (Histogram) | p95 latency, the canonical SLO signal |
| `raghub_ingestion_duration_ms` (Histogram) | Detects ingest contention |
| `raghub_auth_total{success}` (Counter) | Spike in failures ⇒ auth issue |
| `revex_error_total{error_type}` (Counter) | Tracks exceptions by category |

The `RAG` facade emits Langfuse spans (when configured) for every
ingest and query call. Span attributes are documented in
`operations/monitoring.md`.

## Connection pooling

Two pools matter in production:

- **Uvicorn workers.** The default entry point runs a single
  uvicorn process. For higher query concurrency, run with multiple
  workers via `raghub run --workers 4` (or wrap `App.create` in a
  uvicorn factory closure). Each worker has its own application
  instance and SQLite connection. Single-replica SQLite deployments
  should stay at `workers = 1` per process so the file lock isn't
  contested; PostgreSQL deployments scale linearly with worker count.
- **PG client.** `asyncpg` keeps a connection pool; the default
  pool size is fine for a handful of API replicas. Increase it
  via `PG_POOL_MIN_SIZE` / `PG_POOL_MAX_SIZE` (the conventional
  env vars) when you scale out.

## What to monitor first

When in doubt, watch these three in order:

1. The SQLite vector-store row count and on-disk size
   (`SELECT count(*) FROM chunks;` and the size of
   `RAG_VECTORSTORE_PATH`).
2. The API p95 query latency (`raghub_query_duration_ms`).
3. The disk usage of `RAG_DATA_DIR`.
