/**
 * NoOp telemetry — the default. Every method is a no-op; spans are
 * trivial objects whose lifecycle calls do nothing.
 */

import type { Telemetry, TelemetrySpan } from './types.js';

class NoOpSpan implements TelemetrySpan {
  public setAttribute(_key: string, _value: string | number | boolean): void {}
  public setAttributes(_attributes: Readonly<Record<string, string | number | boolean>>): void {}
  public recordException(_err: unknown): void {}
  public end(): void {}
}

export class NoOpTelemetry implements Telemetry {
  public readonly provider = 'noop';

  public span(
    _name: string,
    _attributes?: Readonly<Record<string, string | number | boolean>>,
  ): TelemetrySpan {
    return new NoOpSpan();
  }

  public event(_name: string, _attributes?: Readonly<Record<string, string | number | boolean>>): void {}
}