/**
 * Secret redaction for telemetry payloads.
 *
 * `SECRET_KEY_RE` matches keys whose values look sensitive
 * (`password`, `secret`, `api_key`, `jwt`, `authorization`,
 * `access_token`, `refresh_token`). `scrubSecrets` returns a copy
 * of the input dict with secret-shaped values masked as `***`.
 *
 * `RedactingTelemetry` is a `Telemetry` wrapper that scrubs
 * before forwarding.
 */

import type { Telemetry, TelemetrySpan } from './types.js';

export const SECRET_KEY_RE =
  /(?:^|_)(?:password|passwd|secret|api[_-]?key|apikey|access[_-]?token|refresh[_-]?token|jwt|authorization)(?:$|_)/i;

export const scrubSecrets = <T extends Record<string, unknown>>(input: T): T => {
  const out: Record<string, unknown> = {};
  for (const [k, v] of Object.entries(input)) {
    if (SECRET_KEY_RE.test(k)) {
      out[k] = '***';
    } else if (v && typeof v === 'object' && !Array.isArray(v)) {
      out[k] = scrubSecrets(v as Record<string, unknown>);
    } else {
      out[k] = v;
    }
  }
  return out as T;
};

export class RedactingTelemetry implements Telemetry {
  readonly provider: string;

  private readonly inner: Telemetry;

  constructor(inner: Telemetry) {
    this.inner = inner;
    this.provider = `redacting(${inner.provider})`;
  }

  event(name: string, attributes?: Readonly<Record<string, string | number | boolean>>): void {
    this.inner.event(name, attributes ? scrubSecrets({ ...attributes }) : undefined);
  }

  span(name: string, attributes?: Readonly<Record<string, string | number | boolean>>): TelemetrySpan {
    return this.inner.span(name, attributes ? scrubSecrets({ ...attributes }) : undefined);
  }

  score?(name: string, value: number, comment?: string): void {
    this.inner.score?.(name, value, comment);
  }
}