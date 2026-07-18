# Scaling

This page covers vertical and horizontal scaling of the production
Compose stack. The current shape runs one of each service; the
guidance below covers the next two steps up.

## Vertical scaling

The default resource limits in `docker-compose.yml` are a single
CPU and 1–2 GiB per service. Increase them with `--scale` is **not**
appropriate for these resources; edit the `deploy.resources` block or
override at the command line:

```bash
docker compose -f docker-compose.yml --profile production up -d \
    --scale api=1 \
    api
```

`deploy.resources` accepts a `limits` and `reservations` block. Set
`cpus` to a fractional value (e.g. `2.5`) and `memory` to a Docker
size string (`1G`, `2G`, …).

Recommended ceilings for the bundled stack:

| Service | CPU limit | Memory limit | Notes |
|---|---|---|---|
| `api` | 4.0 | 6G | Dominated by LiteLLM and Marker at ingest time |
| `ui`  | 1.0 | 1G | Streamlit is single-process; size for chat concurrency |
| `qdrant` | 4.0 | 8G | Index memory grows with `RAG_EMBEDDING_DIM × vectors` |

## Horizontal scaling: the API

The API is stateless once it has finished loading. You can run more
than one replica, but they will all attempt to open the same SQLite
files (`registry.db`, `sessions.db`).

Two options:

1. **Move persistence to a network database.** Replace the SQLite
   stores with a Postgres backend and run the API behind a load
   balancer. This is the supported production path for > 1 replica.

2. **Keep SQLite, but run a single API replica.** This is the
   default. If you need additional ingest throughput, scale the
   background ingestion worker pool (see below) and run the API
   service with `replicas: 1`.

To bump the background ingestion pool, override the
`BackgroundIngestionService` at runtime by setting the env var that
`create_app` reads (`BACKGROUND_INGEST_WORKERS` is the conventional
name; the default is `2`).

## Horizontal scaling: Qdrant

Qdrant scales independently. For a single node, set
`QDRANT_URL=http://qdrant:6333` and stay on a single replica. For
sharded deployments, switch to a managed Qdrant cluster and point
`QDRANT_URL` at the cluster endpoint.

The `qdrant` service in `docker-compose.yml` mounts a single named
volume (`raghub_qdrant_data`); do not scale it beyond one replica
without first moving to an external Qdrant cluster.

## Horizontal scaling: the UI

The Streamlit UI is single-process. To run more than one replica,
front them with a load balancer and set
`RAGHUB_API_URL` to a single API endpoint. The UI keeps session
state in `st.session_state`, which is per-replica; sticky sessions
are required for chat continuity.

## Autoscaling signals

The legacy FastAPI surface exposes Prometheus metrics on
`/metrics` (when the `prometheus_client` instrumentation is enabled
in `app.state`). Key signals:

| Metric | Use |
|---|---|
| `raghub_query_duration_ms` (Histogram) | p95 latency, the canonical SLO signal |
| `raghub_ingestion_duration_ms` (Histogram) | Detects ingest contention |
| `raghub_auth_total{success}` (Counter) | Spike in failures ⇒ auth issue |
| `raghub_error_total{error_type}` (Counter) | Tracks exceptions by category |

The `RAG` facade emits Langfuse spans (when configured) for every
ingest and query call. Span attributes are documented in
`operations/monitoring.md`.

## Connection pooling

Two pools matter in production:

- **Uvicorn workers.** The default `CMD` runs a single uvicorn
  process. For higher query concurrency, run with multiple workers
  by overriding the entry point: `uvicorn raghub.api.app:get_app
  --factory --workers 4`. Each worker has its own application
  instance and SQLite connections.
- **Qdrant client.** `qdrant-client` keeps an HTTP/2 connection
  pool; the default limits are fine for one API replica. If you
  scale the API, increase the pool size via
  `QDRANT_CLIENT_POOL_SIZE` (the conventional env var).

## What to monitor first

When in doubt, watch these three in order:

1. The Qdrant segment count (`GET /collections/raghub`).
2. The API p95 query latency (`raghub_query_duration_ms`).
3. The disk usage of `raghub_qdrant_data`.
