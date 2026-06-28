# Monitoring & Observability

## Prometheus Metrics

Prometheus metrics are available via `PrometheusMetrics` when registered with the FastAPI app (see `register_app()` in `observability/metrics.py`).

Available metrics:

| Metric | Type | Description |
|--------|------|-------------|
| `raghub_query_duration_ms` | Histogram | Query execution duration |
| `raghub_ingestion_duration_ms` | Histogram | Ingestion duration |
| `raghub_auth_duration_ms` | Histogram | Authentication duration |
| `raghub_auth_total` | Counter | Authentication attempts (label: `success`) |
| `raghub_error_total` | Counter | Error count (label: `error_type`) |

## OpenTelemetry

`OpenTelemetryTracer` in `observability/tracing.py`. Supports FastAPI auto-instrumentation.

## Structured Logging

Uses `structlog` via `build_logger()` in `observability/logging.py`. Key log events:

- User login/logout
- Document upload and processing status
- Query and response
- Authentication failures
- Rate limit hits

## Health Check

`GET /health` returns service status.
