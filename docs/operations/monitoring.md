# Monitoring & Observability

The RAG facade emits telemetry automatically. The default
telemetry provider is Langfuse (when `LANGFUSE_PUBLIC_KEY` and
`LANGFUSE_SECRET_KEY` are set in the environment); otherwise
`NoOpTelemetry` is used and every method is a no-op.

## What the Facade Emits

For every call to `RAG.ingest` and `RAG.aquery` the facade emits:

| Span                          | Attributes recorded                          |
|-------------------------------|---------------------------------------------|
| `ingest`                      | source_uri, bundle_id, checksum              |
| `ingest.convert`              | (none)                                       |
| `ingest.chunk`                | (none)                                       |
| `ingest.embed`                | count                                        |
| `ingest.upsert`               | count                                        |
| `query`                       | question (truncated to 128 chars), top_k     |
| `query.embed_query`           | (none)                                       |
| `query.search`                | top_k                                        |
| `query.rerank`                | (none)                                       |
| `query.generate`              | (none)                                       |
| `query.structured`            | (none)                                       |
| `query.stream`                | question (truncated), top_k                  |

In addition, every span records its duration (in milliseconds) via
the `record_latency` hook. When the LLM provider exposes token
usage the facade records prompt / completion counts via
`record_tokens`.

## Secret Redaction

The default telemetry wrapper is `RedactingTelemetry`. It scrubs
kwargs whose keys match `password`, `secret`, `api_key`, `token`,
`jwt`, or `authorization` (case-insensitive) before forwarding to
the underlying provider. Nested dicts are scrubbed recursively.

## Prometheus Metrics

The legacy `PrometheusMetrics` is still wired into the FastAPI app
via `register_app()` (see `raghub.observability.metrics`). It
records:

| Metric | Type | Description |
|--------|------|-------------|
| `raghub_query_duration_ms` | Histogram | Query execution duration |
| `raghub_ingestion_duration_ms` | Histogram | Ingestion duration |
| `raghub_auth_duration_ms` | Histogram | Authentication duration |
| `raghub_auth_total` | Counter | Authentication attempts (label: `success`) |
| `raghub_error_total` | Counter | Error count (label: `error_type`) |

## OpenTelemetry

`OpenTelemetryTracer` lives at `raghub.observability.tracing`. The
`StructlogTelemetryProvider` adapter combines the structlog logger,
the Prometheus metrics sink, and an OTel tracer into a single
`TelemetryProvider`-conforming object that the facade can use as
its `telemetry=` argument.

## Structured Logging

`structlog` is configured by `build_logger()` in
`raghub.observability.logging`. Key log events: ingest start/stop,
query start/stop, error events.

## Health Check

`GET /health` returns service status. The RAG facade exposes
`RAG.health()` which returns a dict summarising every collaborator.
