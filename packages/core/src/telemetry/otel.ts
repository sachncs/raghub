/**
 * OpenTelemetry telemetry.
 *
 * Lazy-loads `@opentelemetry/api`; the SDK bootstrap is the
 * caller's responsibility (call `initOtel()` once at startup with
 * the SDK of their choice). We only consume the API surface here.
 */

import type { Telemetry, TelemetrySpan } from './types.js';

interface OtelApi {
  trace: {
    getTracer: (name: string) => Tracer;
  };
}

interface Tracer {
  startSpan: (name: string, opts?: { attributes?: Record<string, string | number | boolean> }) => OtelSpan;
}

interface OtelSpan {
  setAttribute: (key: string, value: string | number | boolean) => void;
  setAttributes: (attrs: Record<string, string | number | boolean>) => void;
  recordException: (err: unknown) => void;
  end: () => void;
}

const dynamicImport = (spec: string): Promise<unknown> => import(spec);

const loadOtel = async (): Promise<OtelApi | null> => {
  try {
    const mod = (await dynamicImport('@opentelemetry/api')) as { default?: OtelApi } & OtelApi;
    return mod.default ?? mod;
  } catch {
    return null;
  }
};

class OtelSpanWrapper implements TelemetrySpan {
  constructor(private readonly span: OtelSpan) {}

  public setAttribute(key: string, value: string | number | boolean): void {
    this.span.setAttribute(key, value);
  }

  public setAttributes(attributes: Readonly<Record<string, string | number | boolean>>): void {
    this.span.setAttributes({ ...attributes });
  }

  public recordException(err: unknown): void {
    this.span.recordException(err);
  }

  public end(): void {
    this.span.end();
  }
}

export class OtelTelemetry implements Telemetry {
  public readonly provider = 'otel';
  private readonly tracer: Tracer;

  constructor(tracer: Tracer) {
    this.tracer = tracer;
  }

  public span(name: string, attributes?: Readonly<Record<string, string | number | boolean>>): TelemetrySpan {
    const handle = this.tracer.startSpan(name, attributes ? { attributes: { ...attributes } } : undefined);
    return new OtelSpanWrapper(handle);
  }

  public event(name: string, attributes?: Readonly<Record<string, string | number | boolean>>): void {
    const handle = this.tracer.startSpan(name, attributes ? { attributes: { ...attributes } } : undefined);
    handle.end();
  }

  public static async create(serviceName: string): Promise<OtelTelemetry | null> {
    const api = await loadOtel();
    if (!api) return null;
    const tracer = api.trace.getTracer(serviceName);
    return new OtelTelemetry(tracer);
  }
}