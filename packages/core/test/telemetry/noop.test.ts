import { describe, expect, it } from 'vitest';

import { NoOpTelemetry } from '../src/telemetry/index.js';
import type { Settings } from '../src/settings/index.js';

const validSecrets = {
  RAGHUB_JWT_SECRET: 'x'.repeat(32),
  RAGHUB_TENANT_SECRETS_KEY: 'a'.repeat(64),
  OPENAI_API_KEY: 'sk-test',
};

const baseSettings = (): Settings => ({
  auth: { jwtSecret: validSecrets.RAGHUB_JWT_SECRET, jwtAlgorithm: 'HS256', tokenTtlSeconds: 60, bcryptRounds: 4 },
  tenants: { isolation: 'row_level' },
  vectorStore: { backend: 'sqlite_vec', path: './x.db', embeddingDim: 3072 },
  embedder: { provider: 'feature_hashing', model: 'x', batchSize: 1, apiKey: undefined },
  llm: { provider: 'openai', model: 'x', temperature: 0, apiKey: undefined },
  hybrid: { denseWeight: 0.6, sparseWeight: 0.4, rrfK: 60, colbert: false },
  orchestrator: {
    mode: 'graph',
    ordering: 'standard',
    topK: 10,
    reranker: 'identity',
    multimodal: { enabled: false, embeddingModel: 'x', embeddingDim: 3072 },
    traceCorpus: { enabled: false, representation: 'semantic', topK: 5 },
  },
  telemetry: { provider: 'noop' },
  secrets: { tenantSecretsKey: validSecrets.RAGHUB_TENANT_SECRETS_KEY },
});

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