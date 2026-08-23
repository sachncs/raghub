import { describe, expect, it, vi } from 'vitest';

import { Isolation, VectorBackend, loadSettings } from '../src/settings/index.js';

const valid = {
  RAGHUB_JWT_SECRET: 'x'.repeat(32),
  RAGHUB_TENANT_SECRETS_KEY: 'a'.repeat(64),
  OPENAI_API_KEY: 'sk-test',
};

describe('loadSettings', () => {
  it('returns defaults for every optional key', () => {
    const s = loadSettings(valid);
    expect(s.auth.jwtAlgorithm).toBe('HS256');
    expect(s.auth.bcryptRounds).toBe(10);
    expect(s.tenants.isolation).toBe(Isolation.RowLevel);
    expect(s.vectorStore.backend).toBe(VectorBackend.SqliteVec);
    expect(s.vectorStore.embeddingDim).toBe(3072);
    expect(s.embedder.provider).toBe('openai');
    expect(s.embedder.model).toBe('text-embedding-3-large');
    expect(s.llm.provider).toBe('openai');
    expect(s.llm.temperature).toBe(0);
    expect(s.hybrid.denseWeight).toBe(0.6);
    expect(s.hybrid.sparseWeight).toBe(0.4);
    expect(s.hybrid.rrfK).toBe(60);
    expect(s.hybrid.colbert).toBe(false);
    expect(s.orchestrator.mode).toBe('graph');
    expect(s.orchestrator.ordering).toBe('standard');
    expect(s.orchestrator.topK).toBe(10);
    expect(s.orchestrator.reranker).toBe('identity');
    expect(s.orchestrator.multimodal.enabled).toBe(false);
    expect(s.orchestrator.traceCorpus.enabled).toBe(false);
    expect(s.telemetry.provider).toBe('noop');
  });

  it('overrides every recognised env var', () => {
    const s = loadSettings({
      ...valid,
      RAGHUB_ISOLATION: Isolation.RowLevel,
      RAGHUB_VECTOR_PATH: '/tmp/x.db',
      RAGHUB_EMBEDDER_MODEL: 'text-embedding-3-small',
      RAGHUB_LLM_MODEL: 'gpt-4.1-mini',
      RAGHUB_HYBRID_DENSE_WEIGHT: '0.7',
      RAGHUB_HYBRID_SPARSE_WEIGHT: '0.3',
      RAGHUB_HYBRID_COLBERT: 'true',
      RAGHUB_ORCHESTRATOR_MODE: 'swarm',
      RAGHUB_ORCHESTRATOR_ORDERING: 'reverse',
      RAGHUB_ORCHESTRATOR_TOP_K: '25',
      RAGHUB_ORCHESTRATOR_RERANKER: 'bge',
      RAGHUB_MULTIMODAL_ENABLED: 'true',
      RAGHUB_TRACE_CORPUS_ENABLED: 'true',
      RAGHUB_TRACE_CORPUS_REPRESENTATION: 'reflect',
      RAGHUB_TELEMETRY_PROVIDER: 'langfuse',
    });
    expect(s.vectorStore.path).toBe('/tmp/x.db');
    expect(s.embedder.model).toBe('text-embedding-3-small');
    expect(s.llm.model).toBe('gpt-4.1-mini');
    expect(s.hybrid.denseWeight).toBe(0.7);
    expect(s.hybrid.sparseWeight).toBe(0.3);
    expect(s.hybrid.colbert).toBe(true);
    expect(s.orchestrator.mode).toBe('swarm');
    expect(s.orchestrator.ordering).toBe('reverse');
    expect(s.orchestrator.topK).toBe(25);
    expect(s.orchestrator.reranker).toBe('bge');
    expect(s.orchestrator.multimodal.enabled).toBe(true);
    expect(s.orchestrator.traceCorpus.enabled).toBe(true);
    expect(s.orchestrator.traceCorpus.representation).toBe('reflect');
    expect(s.telemetry.provider).toBe('langfuse');
  });

  it('rejects missing jwtSecret', () => {
    const { RAGHUB_JWT_SECRET: _omit, ...rest } = valid;
    expect(() => loadSettings(rest)).toThrow(/jwtSecret/);
  });

  it('rejects malformed workspaceSecretsKey', () => {
    expect(() => loadSettings({ ...valid, RAGHUB_TENANT_SECRETS_KEY: 'too-short' })).toThrow();
  });

  it('warns on unknown RAGHUB_* env var but does not throw', () => {
    const warn = vi.spyOn(console, 'warn').mockImplementation(() => {});
    const s = loadSettings({ ...valid, RAGHUB_NY: '1' });
    expect(s.auth.jwtSecret).toBe(valid.RAGHUB_JWT_SECRET);
    expect(warn).toHaveBeenCalledWith(expect.stringContaining('RAGHUB_NY'));
    warn.mockRestore();
  });
});