# Monitoring & Observability

The `RAG` facade emits telemetry automatically through its
default provider chain:

```
RAG
  └─► RedactingTelemetry      # scrubs secrets
        └─► LangfuseTelemetryProvider      # when langfuse is configured
              or NoOpTelemetry             # otherwise
```

`RAG.telemetry` always exposes the protocol defined by
`raghub.interfaces.observability.TelemetryProvider`. The default
constructor wraps the underlying provider in
`RedactingTelemetry` so secret-looking kwargs are scrubbed before
forwarding to the sink.

## What the facade emits

`IngestPipeline.run` and `QueryPipeline.run` / `.stream` open
nested telemetry spans. Each span records its duration in
milliseconds. When the LLM provider exposes token usage the
pipeline forwards it through `telemetry.record_tokens`.

### Ingest

| Span | Attributes recorded |
|---|---|
| `ingest` | `source_uri`, `bundle_id`, `checksum` |
| `ingest.convert` | — |
| `ingest.chunk` | — |
| `ingest.embed` | `count` (number of texts) |
| `ingest.upsert` | `count` (number of chunks) |

### Query (and Stream)

| Span | Attributes recorded |
|---|---|
| `query` (or `query.stream`) | `question` (truncated to 128 chars), `top_k`, `user_id`, `session_id` |
| `query.embed_query` | — |
| `query.search` | `top_k` |
| `query.rerank` | — |
| `query.generate` | token-usage forwarded on completion |
| `query.structured` | — (only when `response_model=` is used) |
| `query.tokens` | `prompt_tokens`, `completion_tokens` (stream only) |

## Token usage

`LiteLLMProvider.generate` and `LiteLLMProvider.astream`
populate `self.last_usage` (prompt / completion / model).
`DefaultGenerator.record_tokens()` exposes that counter;
`QueryPipeline.run` and `QueryPipeline.stream` call it and pipe
the values to `telemetry.record_tokens("query.generate" / "query.stream", ...)`.

In `pipeline.run`:
```python
self.telemetry.record_tokens(
    "query.generate",
    prompt_tokens=...,
    completion_tokens=...,
    model=...,
)
```

In `pipeline.stream`, the same call uses `query.stream`. The
generator's `last_usage` is read once after the stream completes
so `record_tokens` doesn't run on every chunk.

## Secret redaction

`RedactingTelemetry` (in `raghub.observability.redact`) walks the
kwarg dict recursively and replaces any value whose key matches
the regex
`re.compile(r"(password|secret|api_key|token|jwt|authorization)", re.I)`
with the literal string `"***REDACTED***"` before forwarding to
the underlying provider. Nested dicts are scrubbed depth-first.

To opt out, pass your own `telemetry=` to the `RAG` constructor
and skip the redaction wrapper.

## Switching telemetry providers

Construct the facade with a custom telemetry provider:

```python
from raghub import RAG
from raghub.observability.noop import NoOpTelemetry

rag = RAG(telemetry=NoOpTelemetry())
```

…or with the redaction wrapper around any backend:

```python
from raghub.observability.redact import RedactingTelemetry
from raghub.telemetry.langfuse import LangfuseTelemetryProvider

rag = RAG(telemetry=RedactingTelemetry(LangfuseTelemetryProvider()))
```

## Legacy surface observability

The legacy FastAPI surface (`raghub.api.app:get_app`, served via
Uvicorn's `--factory`) continues to expose:

- `raghub_query_duration_ms` (Histogram) — query execution duration
- `raghub_ingestion_duration_ms` (Histogram) — ingestion duration
- `raghub_auth_duration_ms` (Histogram) — authentication duration
- `raghub_auth_total` (Counter, label `success`) — login attempts
- `raghub_error_total` (Counter, label `error_type`) — error count

These are wired through
`raghub.observability.metrics.PrometheusMetrics` +
`raghub.observability.tracing.OpenTelemetryTracer` +
`raghub.observability.structlog_provider.StructlogTelemetryProvider`.

The legacy surface is the only place Prometheus + OTel are
mounted; the new `RAG` facade does not register a Prometheus
collector or an OTel exporter itself.

## Container-level observability

The production compose stack attaches a JSON-file log driver with
rotation (10 MiB × 5 files) to every service. Tail with
`docker compose logs -f <service>`; inspect the rotation policy
with `docker inspect <container>`.

Each service has a `healthcheck` block; the API depends on Qdrant
(`condition: service_healthy`) and the UI depends on the API. The
`docker compose ps` view reports the live state for every
container.

## Structured logging

`build_logger()` in `raghub.observability.logging` configures
`structlog`. The recommended key log events are `ingest.start`,
`ingest.stop`, `query.start`, `query.stop`, and `error`-classed
events emitted from the pipelines.

The log level is controlled by `RAG_LOG_LEVEL` (default `INFO`).

## Health check

The CLI exposes the facade's health summary:

```bash
raghub health
```

…equivalent to:

```python
import json
from raghub import RAG

print(json.dumps(RAG().health(), indent=2))
```

```json
{
  "status": "ok",
  "vector_store": "InMemoryVectorStore",
  "embedder":     "HashingEmbeddingProvider",
  "llm":          "HeuristicLLMProvider",
  "chunker":      "WordWindowChunker",
  "converter":    "PlainTextConverter",
  "telemetry":    "RedactingTelemetry",
  "structured":   "NoneType",
  "reranker":     "IdentityReranker"
}
```

`GET /health` (FastAPI surface) is the liveness probe and
returns whatever `DynamicRagApplication.health()` reports.

## See also

- [`plugins.md`](../plugins.md) — register a custom telemetry
  pair via `PluginRegistry.register_telemetry(name, logger, metrics)`.
- [ADR-0005: telemetry scrubbing](../architecture/decisions.md#adr-0005-telemetry-scrubbing-is-the-default)
  — ADR-0005 (default scrubbing) and ADR-0007 (Langfuse v3+ spans).
- [`runbook.md`](runbook.md) — first-line triage for failing
  services; covers health, logs, restarts, and the canonical
  reset path.
