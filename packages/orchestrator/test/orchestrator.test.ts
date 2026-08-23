import { describe, expect, it } from 'vitest';

import {
  type CollectionId,
  NoOpTelemetry,
  type TenantId,
  type UserId,
  User,
  UserRole,
  brandId,
} from '@raghub/core';
import { AgentRegistry, Orchestrator } from '../src/index.js';
import {
  createGeneratorAgent,
  createRetrieverAgent,
} from '../src/agents/defaults.js';
import { ToolRegistry } from '../src/tools/registry.js';
import { resolveStrategy } from '../src/patterns/strategy.js';

const tenantId = brandId<TenantId>('tnt_1');
const userId = brandId<UserId>('usr_1');
const collectionId = brandId<CollectionId>('col_1');

const adminUser = new User({
  id: userId,
  tenantId,
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
  it('dispatches by strategy.mode and propagates invocation state', async () => {
    const agents = new AgentRegistry();
    agents.register('retriever', createRetrieverAgent({
      retrieve: (async () => ({ ok: true, content: 'r', hits: [], latencyMs: 1 })) as never,
    } as never));
    agents.register('generator', createGeneratorAgent());

    const tools = new ToolRegistry();
    const orch = new Orchestrator({
      telemetry: new NoOpTelemetry(),
      tenantId,
      collectionId,
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
    const tools = new ToolRegistry();
    const orch = new Orchestrator({
      telemetry: new NoOpTelemetry(),
      tenantId,
      agents,
      tools,
    });
    const state = orch.resolveInvocationState({
      question: 'q',
      user: adminUser,
      sessionId: null,
    });
    expect(state.tenant_id).toBe(tenantId);
    expect(state.user_id).toBe(userId);
    expect(state.is_admin).toBe(true);
    expect(state.rbac_filter.tenantId).toBe(tenantId);
  });

  it('emits PlannerEvents from stream()', async () => {
    const agents = new AgentRegistry();
    agents.register('retriever', createRetrieverAgent({
      retrieve: (async () => ({ ok: true, content: 'r', hits: [], latencyMs: 1 })) as never,
    } as never));
    agents.register('generator', createGeneratorAgent());
    const tools = new ToolRegistry();
    const orch = new Orchestrator({
      telemetry: new NoOpTelemetry(),
      tenantId,
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