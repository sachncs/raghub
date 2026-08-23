/**
 * Telemetry — the cross-cutting observability surface. NoOp is the
 * default; Langfuse and OpenTelemetry are opt-in.
 */

export interface TelemetrySpan {
  setAttribute(key: string, value: string | number | boolean): void;
  setAttributes(attributes: Readonly<Record<string, string | number | boolean>>): void;
  recordException(err: unknown): void;
  end(): void;
}

export interface Telemetry {
  readonly provider: string;

  /** Begin a span around an operation; always returns a span (or no-op span). */
  span(name: string, attributes?: Readonly<Record<string, string | number | boolean>>): TelemetrySpan;

  /** Emit a discrete event (no duration). */
  event(name: string, attributes?: Readonly<Record<string, string | number | boolean>>): void;

  /** Score an entity (Langfuse semantics); may no-op for non-Langfuse providers. */
  score?(name: string, value: number, comment?: string): void;
}