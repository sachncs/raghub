# Telemetry

`@revex/core` exposes a pluggable telemetry surface. The default is a no-op
so the framework runs without any external service.

## Providers

| Provider | Class | Notes |
|---|---|---|
| No-op | `NoOpTelemetry` | Default; discards everything. |
| Langfuse | `LangfuseTelemetry` | Lazy-loaded; falls back to no-op when credentials are missing. |
| OpenTelemetry | `OtelTelemetry` | Lazy-loaded via `@opentelemetry/api`. |

`createTelemetry(settings)` picks the provider from
`settings.telemetry.provider` (`noop` | `langfuse` | `otel`). Unknown providers
fall back to `NoOpTelemetry` with a one-shot warning.

## Interfaces

```ts
interface Telemetry {
  span(name: string, opts?): Promise<TelemetrySpan>;
  // ...
}
interface TelemetrySpan {
  end(attrs?): Promise<void>;
  setAttribute(k, v): Promise<void>;
}
```

## Config

`telemetry` settings:

- `provider`: `noop | langfuse | otel` (default `noop`).
- `langfusePublicKey`, `langfuseSecretKey`, `langfuseBaseUrl`.
- `otelEndpoint`.

## Logging & redaction

- `createLogger()` / `setSink(sink)` — leveled logger with a pluggable sink.
- `SECRET_KEY_RE` + `scrubSecrets(obj)` — redact known secret-shaped keys.
- `RedactingTelemetry` — wraps a telemetry provider and scrubs secrets from
  every span/event, so API keys never leak into traces.

## Graph & web

- `SqliteGraphStore` (`graph/store.ts`) provides GraphRAG storage:
  `addMentions`, `searchEntities`, `expandNeighborhood`.
- `WebSearch` interface with `DuckDuckGoSearch` + `createDuckDuckGoSearch()`.