import { describe, expect, it, vi } from 'vitest';

import type { ChunkId, CollectionId, DocumentId, Hit, UserId, WorkspaceId } from '@revex/core';
import { Chunk, brandId } from '@revex/core';

import { AgentRegistry } from '../../src/agents/registry.js';
import { RagAgent, createSessionState, defaultRetryStrategy } from '../../src/agents/rag-agent.js';
import { createGeneratorAgent } from '../../src/agents/defaults.js';
import type { OrchestratorRequest, InvocationState } from '../../src/strands/types.js';

const chunk = (id: string, score: number): Hit => ({
  chunk: new Chunk({
    id: brandId<ChunkId>(id),
    workspaceId: brandId<WorkspaceId>('wsp_1'),
    ownerId: brandId<UserId>('usr_1'),
    collectionId: brandId<CollectionId>('col_1'),
    documentId: brandId<DocumentId>('doc_1'),
    modality: 'text',
    text: `chunk ${id}`,
    embedding: [],
    metadata: {},
    tokenCount: 1,
    createdAt: new Date(),
  }),
  score,
});

const fakeLlm = () =>
  ({
    generate: async () => ({ content: 'stub answer' }),
    stream: (async function* () {})(),
    model: 'gpt-4.1',
  }) as never;

const baseReq = (): OrchestratorRequest => ({
  question: 'what is revex?',
  user: null,
  sessionId: null,
});

const baseState = (): InvocationState => ({
  workspace_id: brandId<WorkspaceId>('wsp_1'),
  user_id: brandId<UserId>('usr_1'),
  is_admin: false,
  rbac_filter: {
    workspaceId: brandId<WorkspaceId>('wsp_1'),
    userId: brandId<UserId>('usr_1'),
    collectionId: null,
    allowedCompanies: [],
  },
  session_id: null,
  session_overrides: {},
  strategy: {
    mode: 'graph',
    hybrid: { denseWeight: 0.7, sparseWeight: 0.3, rrfK: 60, colbert: false },
    ordering: 'standard',
    k: 10,
    reranker: 'identity',
    multimodal: { enabled: false },
    traceCorpus: { enabled: false, representation: 'semantic', topK: 5 },
  },
  trace_id: 'trace_1',
  request_id: 'req_1',
  db: null,
  secrets: {},
});

describe('RagAgent', () => {
  it('fans out to sub-agents in parallel and merges hits', async () => {
    const vector = vi.fn(async () => [chunk('a', 0.9), chunk('b', 0.7)]);
    const keyword = vi.fn(async () => [chunk('b', 0.85), chunk('c', 0.6)]);
    const agents = new AgentRegistry();
    agents.register('generator', createGeneratorAgent({ llm: fakeLlm(), model: 'gpt-4.1' }));
    const rag = new RagAgent({
      agents,
      subAgents: [
        { role: 'vector', retrieve: vector as never },
        { role: 'keyword', retrieve: keyword as never },
      ],
    });
    const result = await rag.run(baseReq(), baseState());
    expect(result.answer).toBe('stub answer');
    expect(result.hits.length).toBe(3);
    expect(result.citations.length).toBe(3);
    expect(result.hits[0]?.chunk.id).toBe('a');
  });

  it('fires before/after retrieve hooks', async () => {
    const before = vi.fn();
    const after = vi.fn();
    const vector = vi.fn(async () => [chunk('a', 0.9)]);
    const agents = new AgentRegistry();
    agents.register('generator', createGeneratorAgent({ llm: fakeLlm(), model: 'gpt-4.1' }));
    const rag = new RagAgent({
      agents,
      subAgents: [{ role: 'vector', retrieve: vector as never }],
      hooks: { beforeRetrieve: before, afterRetrieve: after },
    });
    await rag.run(baseReq(), baseState());
    expect(before).toHaveBeenCalledWith('vector', expect.anything());
    expect(after).toHaveBeenCalled();
  });

  it('retries on transient sub-agent failure', async () => {
    let attempts = 0;
    const flaky = vi.fn(async () => {
      attempts++;
      if (attempts < 3) throw new Error('transient');
      return [chunk('a', 0.9)];
    });
    const agents = new AgentRegistry();
    agents.register('generator', createGeneratorAgent({ llm: fakeLlm(), model: 'gpt-4.1' }));
    const rag = new RagAgent({
      agents,
      subAgents: [{ role: 'vector', retrieve: flaky as never }],
      retry: { ...defaultRetryStrategy(), baseDelayMs: 1, maxDelayMs: 2 },
    });
    const result = await rag.run(baseReq(), baseState());
    expect(attempts).toBe(3);
    expect(result.hits.length).toBe(1);
  });

  it('summarizes history beyond turnLimit', async () => {
    const vector = vi.fn(async () => [chunk('a', 0.9)]);
    const agents = new AgentRegistry();
    let summarizeCalls = 0;
    agents.register(
      'generator',
      createGeneratorAgent({
        llm: {
          generate: async () => ({ content: 'final answer' }),
          stream: (async function* () {})(),
          model: 'gpt-4.1',
        } as never,
        model: 'gpt-4.1',
      }),
    );
    agents.register(
      'summarizer',
      createGeneratorAgent({
        llm: {
          generate: async () => {
            summarizeCalls++;
            return { content: 'summary text' };
          },
          stream: (async function* () {})(),
          model: 'gpt-4.1',
        } as never,
        model: 'gpt-4.1',
      }),
    );
    const rag = new RagAgent({
      agents,
      subAgents: [{ role: 'vector', retrieve: vector as never }],
      turnLimit: 2,
      summarizerId: 'summarizer',
    });
    const longHistory = [
      { role: 'user' as const, content: 'q1' },
      { role: 'assistant' as const, content: 'a1' },
      { role: 'user' as const, content: 'q2' },
      { role: 'assistant' as const, content: 'a2' },
      { role: 'user' as const, content: 'q3' },
      { role: 'assistant' as const, content: 'a3' },
    ];
    await rag.run({ ...baseReq(), history: longHistory }, baseState());
    expect(summarizeCalls).toBeGreaterThanOrEqual(1);
  });

  it('sessionState bag stores per-session values', () => {
    const bag = createSessionState({ turn: 0 });
    bag.set('plan', ['web', 'memory']);
    expect(bag.get<string[]>('plan')).toEqual(['web', 'memory']);
    bag.set('turn', 1);
    expect(bag.values['turn']).toBe(1);
  });
});