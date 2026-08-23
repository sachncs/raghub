/**
 * The Orchestrator — single façade, three patterns, one call surface.
 *
 * Construction wires the Strands adapter, the agent registry, and
 * the tool registry. `run()` and `stream()` dispatch into the
 * pattern chosen by the resolver (request > session > user > tenant
 * > global). Mode overrides (`mode: 'graph' | ...`) take effect
 * immediately.
 */

import type { Telemetry } from '@raghub/core';
import { runWithTenantAsync, type CollectionId, type TenantId } from '@raghub/core';

import { buildGraph, buildSwarm, buildWorkflow, type PatternBuilder } from './patterns/builders.js';
import { resolveStrategy, type StrategyOverrides } from './patterns/strategy.js';
import type { AgentRegistry } from './agents/registry.js';
import type { ToolRegistry } from './tools/registry.js';
import type { StrandsAdapter } from './strands/adapter.js';
import { InProcessAdapter } from './strands/in-process-adapter.js';
import { buildInvocationState } from './strands/invocation-state.js';
import type {
  Citation,
  InvocationState,
  OrchestratorRequest,
  OrchestratorResult,
  PlannerEvent,
  Strategy,
  Strategy as StrategyShape,
} from './strands/types.js';
import type { User } from '@raghub/core';

export interface OrchestratorOptions {
  readonly telemetry: Telemetry;
  readonly tenantId: TenantId;
  readonly collectionId?: CollectionId | null;
  readonly defaultStrategy?: Strategy;
  readonly sessionOverrides?: Readonly<Record<string, unknown>>;
  readonly adapters?: { readonly graph?: StrandsAdapter; readonly swarm?: StrandsAdapter; readonly workflow?: StrandsAdapter };
  readonly agents: AgentRegistry;
  readonly tools: ToolRegistry;
}

export class Orchestrator {
  private readonly telemetry: Telemetry;
  private readonly tenantId: TenantId;
  private readonly collectionId: CollectionId | null;
  private readonly defaultStrategy: Strategy;
  private readonly sessionOverrides: Readonly<Record<string, unknown>>;
  private readonly patterns: Record<StrategyShape['mode'], PatternBuilder>;
  private readonly agents: AgentRegistry;
  private readonly tools: ToolRegistry;

  constructor(opts: OrchestratorOptions) {
    this.telemetry = opts.telemetry;
    this.tenantId = opts.tenantId;
    this.collectionId = opts.collectionId ?? null;
    this.defaultStrategy = opts.defaultStrategy ?? resolveStrategy([]);
    this.sessionOverrides = opts.sessionOverrides ?? {};
    this.agents = opts.agents;
    this.tools = opts.tools;
    const adapter: StrandsAdapter = new InProcessAdapter({ agents: this.agents, tools: this.tools });
    this.patterns = {
      graph: buildGraph(opts.adapters?.graph ?? adapter),
      swarm: buildSwarm(opts.adapters?.swarm ?? adapter),
      workflow: buildWorkflow(opts.adapters?.workflow ?? adapter),
    };
  }

  public async run(req: OrchestratorRequest): Promise<OrchestratorResult> {
    const state = this.makeInvocationState(req);
    const mode = state.strategy.mode;
    return runWithTenantAsync(
      {
        tenantId: state.tenant_id,
        userId: state.user_id,
        isAdmin: state.is_admin,
        sessionId: state.session_id,
      },
      () => this.dispatch(mode, req, state),
    );
  }

  public async *stream(req: OrchestratorRequest): AsyncGenerator<PlannerEvent> {
    const result = await this.run(req);
    for (const ev of result.events) yield ev;
  }

  public resolveInvocationState(req: OrchestratorRequest): InvocationState {
    return this.makeInvocationState(req);
  }

  private makeInvocationState(req: OrchestratorRequest): InvocationState {
    const userOverrides = req.user ? extractUserOverrides(req.user) : {};
    const strategy = resolveStrategy([
      this.defaultStrategy,
      this.sessionOverrides['strategy'] as never,
      userOverrides,
    ]);
    return buildInvocationState({
      tenantId: this.tenantId,
      user: req.user,
      sessionId: req.sessionId,
      sessionOverrides: this.sessionOverrides,
      strategy,
      db: this.sessionOverrides['db'],
      secrets: this.sessionOverrides['secrets'],
    });
  }

  private async dispatch(
    mode: StrategyShape['mode'],
    req: OrchestratorRequest,
    state: InvocationState,
  ): Promise<OrchestratorResult> {
    const span = this.telemetry.span(`orchestrator.${mode}`, { question_len: req.question.length });
    try {
      const builder = this.patterns[mode];
      const result = await builder.run(req, state);
      span.setAttribute('citations', result.citations.length);
      span.setAttribute('hits', result.hits.length);
      return result;
    } catch (e) {
      span.recordException(e);
      throw e;
    } finally {
      span.end();
    }
  }
}

const extractUserOverrides = (user: User): { strategy?: StrategyOverrides } => {
  const meta = (user as unknown as { toJSON?: () => Record<string, unknown> }).toJSON?.() ?? {};
  const strat = meta['strategy'];
  if (!strat || typeof strat !== 'object') return {};
  return { strategy: strat as StrategyOverrides };
};

// Re-export so consumers don't need a second import.
export type { Citation, PlannerEvent };