import { describe, expect, it } from 'vitest';

import {
  type CollectionId,
  type Llm,
  NoOpTelemetry,
  type Retrieval,
  type WorkspaceId,
  type UserId,
  User,
  UserRole,
  brandId,
} from '@revex/core';
import { AgentRegistry, type Agent, Orchestrator } from '../src/index.js';
import { ToolRegistry } from '../src/tools/registry.js';
import { resolveStrategy } from '../src/patterns/strategy.js';

const workspaceId = brandId<WorkspaceId>('tnt_1');
const userId = brandId<UserId>('usr_1');
const collectionId = brandId<CollectionId>('col_1');

const adminUser = new User({
  id: userId,
  workspaceId,
  email: 'a@x',
  role: UserRole.Admin,
  allowedCompanies: [],
  createdAt: new Date(),
});

describe('resolveStrategy', () => {
  it('returns defaults for an empty layer list', () => {
    const s = resolveStrategy([]);
    expect(s.mode).toBe('graph');
    expect(s.k).toBe(10);
    expect(s.ordering).toBe('standard');
    expect(s.reranker).toBe('identity');
    expect(s.multimodal.enabled).toBe(false);
    expect(s.traceCorpus.enabled).toBe(false);
  });

  it('overrides defaults with later layers', () => {
    const s = resolveStrategy([{ k: 20 }, { mode: 'swarm' }]);
    expect(s.k).toBe(20);
    expect(s.mode).toBe('swarm');
  });
});

describe('Orchestrator', () => {
  const fakeLlm: Llm = {
    provider: 'fake',
    model: 'test',
    async generate() {
      return { content: 'ok', toolCalls: [], usage: { promptTokens: 0, completionTokens: 0, totalTokens: 0 }, finishReason: 'stop' };
    },
    stream: async function* () {
      yield { delta: 'ok', toolCalls: [], finishReason: 'stop' };
    },
    async rawStream() {
      throw new Error('not used');
    },
  };
  const fakeRetrieval: Retrieval = {
    async retrieve() {
      return [];
    },
  } as unknown as Retrieval;
  const noopAgent: Agent = {
    id: 'noop',
    async retrieve() { return { ok: true, content: '', hits: [], latencyMs: 0 }; },
    async generate() { return { answer: '' }; },
  };

  it('dispatches by strategy.mode and propagates invocation state', async () => {
    const agents = new AgentRegistry();
    agents.register('retriever', noopAgent);
    agents.register('generator', noopAgent);

    const tools = new ToolRegistry();
    const orch = new Orchestrator({
      telemetry: new NoOpTelemetry(),
      workspaceId,
      defaultStrategy: resolveStrategy([{ mode: 'swarm' }]),
      agents,
      tools,
    });

    const result = await orch.run({
      question: 'hello',
      user: adminUser,
      sessionId: null,
    });
    expect(result.mode).toBe('swarm');
    expect(result.events.length).toBeGreaterThan(0);
  });

  it('exposes invocationState with tenant + user + strategy', () => {
    const agents = new AgentRegistry();
    agents.register('generator', noopAgent);
    const tools = new ToolRegistry();
    const orch = new Orchestrator({
      telemetry: new NoOpTelemetry(),
      workspaceId,
      agents,
      tools,
    });
    const state = orch.resolveInvocationState({
      question: 'q',
      user: adminUser,
      sessionId: null,
    });
    expect(state.workspace_id).toBe(workspaceId);
    expect(state.user_id).toBe(userId);
    expect(state.is_admin).toBe(true);
    expect(state.rbac_filter.workspaceId).toBe(workspaceId);
  });

  it('emits PlannerEvents from stream()', async () => {
    const agents = new AgentRegistry();
    agents.register('retriever', noopAgent);
    agents.register('generator', noopAgent);
    const tools = new ToolRegistry();
    const orch = new Orchestrator({
      telemetry: new NoOpTelemetry(),
      workspaceId,
      agents,
      tools,
    });
    const events: string[] = [];
    for await (const ev of orch.stream({ question: 'q', user: adminUser, sessionId: null })) {
      events.push(ev.kind);
    }
    expect(events).toContain('thought');
  });
});