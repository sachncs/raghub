import { describe, expect, it } from 'vitest';

import { NoOpTelemetry } from '../../src/telemetry/index.js';

describe('telemetry', () => {
  it('NoOpTelemetry is the default and every call is safe', () => {
    const t = new NoOpTelemetry();
    expect(t.provider).toBe('noop');
    const span = t.span('test', { foo: 'bar' });
    span.setAttribute('k', 'v');
    span.setAttributes({ a: 1, b: true });
    span.recordException(new Error('ignored'));
    span.end();
    t.event('e1', { k: 'v' });
  });
});