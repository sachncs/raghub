/**
 * NoOp telemetry — the default. Every method is a no-op; spans are
 * trivial objects whose lifecycle calls do nothing.
 */

import type { Telemetry, TelemetrySpan } from './types.js';

class NoOpSpan implements TelemetrySpan {
  public setAttribute(): void {}
  public setAttributes(): void {}
  public recordException(): void {}
  public end(): void {}
}

export class NoOpTelemetry implements Telemetry {
  public readonly provider = 'noop';

  public span(): TelemetrySpan {
    return new NoOpSpan();
  }

  public event(): void {}
}