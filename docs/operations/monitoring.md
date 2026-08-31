> ⚠️ **ARCHIVED** — This document describes the **raghub 0.9.x Python
> release**, preserved for historical reference in
> [`archive/`](../../archive/). It does **not** describe the active
> **Revex** TypeScript codebase.
>
> For current documentation, see [`README.md`](../../README.md) and the
> TypeScript source under `packages/`.

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
`raghub.telemetry.TelemetryProvider`. The default
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

`LiteLLM.generate` and `LiteLLM.astream`
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

`RedactingTelemetry` (in `raghub.telemetry`) walks the
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
from raghub.telemetry import NoOpTelemetry

rag = RAG(telemetry=NoOpTelemetry())
```

…or with the redaction wrapper around any backend:

```python
from raghub.telemetry import RedactingTelemetry
from raghub.telemetry import LangfuseTelemetryProvider

rag = RAG(telemetry=RedactingTelemetry(LangfuseTelemetryProvider()))
```

## Legacy surface observability

The FastAPI surface (`raghub.api.App.create`, served via
Uvicorn's `--factory`) continues to expose:

- `raghub_query_duration_ms` (Histogram) — query execution duration
- `raghub_ingestion_duration_ms` (Histogram) — ingestion duration
- `raghub_auth_duration_ms` (Histogram) — authentication duration
- `raghub_auth_total` (Counter, label `success`) — login attempts
- `revex_error_total` (Counter, label `error_type`) — error count

These are wired through
`raghub.telemetry.metrics.PrometheusMetrics` (registered via
`metrics.register_app(app)` in `create_app`) and
`raghub.telemetry.tracing.Tracer` (constructor takes the OTLP
endpoint via `tracer.add_otlp_exporter(...)`).
`raghub.telemetry.logging.LoguruTelemetryProvider` provides the
loguru-backed span/logger surface; `RedactingTelemetry` is the
default wrapper that scrubs secret-like kwargs.

The RAG facade emits loguru records by default; configure
`LANGFUSE_*` env vars to forward spans through `LoguruTelemetryProvider`.

## Process-level observability

When Revex runs as a long-lived process (e.g. `raghub run`,
`uvicorn raghub.api:App.create --factory`, or a
Kubernetes deployment), configure your process supervisor to
capture the loguru records emitted from `raghub.telemetry.logging`
and the Prometheus metrics served on the standard `/metrics`
endpoint. Log rotation is the responsibility of the supervisor
(`logrotate`, the Kubernetes logging agent, `journald`, etc.) —
nothing in the application manages log files on disk.

Health checks for the process are exposed at:

* `GET /health` — FastAPI liveness probe (no auth).
* `GET /v1/health` — service-level health summary, identical to
  `RagApplication.health()`.
* `raghub health` — CLI equivalent, prints the same JSON payload.

An orchestrator can poll `/health` at any interval; the body is
cheap (no database calls, no auth lookup) and the response is
`200 ok` whenever the app is responsive.

## Structured logging

`build_logger()` in `raghub.telemetry.logging` configures
`loguru`. The recommended key log events are `ingest.start`,
`ingest.stop`, `query.start`, `query.stop`, `cli.ingest.start`,
`cli.run.starting`, and `error`-classed events emitted from the
pipelines. `RAG.ingest_directory_*` and `RAG.sync_index` emit
`tqdm.tqdm` progress bars to stderr at the configured log level;
set `show_progress=False` for non-interactive callers.

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
  "vector_store": "MemoryStore",
  "embedder":     "Hasher",
  "llm":          "HeuristicLLMProvider",
  "chunker":      "WordChunker",
  "converter":    "PlainTextConverter",
  "telemetry":    "RedactingTelemetry",
  "structured":   "NoneType",
  "reranker":     "IdentityReranker"
}
```

`GET /health` (FastAPI surface) is the liveness probe and
returns whatever `RagApplication.health()` reports.

## See also

- [`plugins.md`](../plugins.md) — register a custom telemetry
  pair via `PluginRegistry.register_telemetry(name, logger, metrics)`.
- [ADR-0005: telemetry scrubbing](../architecture/decisions.md#adr-0005-telemetry-scrubbing-is-the-default)
  — ADR-0005 (default scrubbing) and ADR-0007 (Langfuse v3+ spans).
- [`runbook.md`](runbook.md) — first-line triage for failing
  services; covers health, logs, restarts, and the canonical
  reset path.
