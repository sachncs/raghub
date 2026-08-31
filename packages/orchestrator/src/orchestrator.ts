/**
 * The Orchestrator — single façade, three patterns, one call surface.
 *
 * Construction wires the Strands adapter, the agent registry, and
 * the tool registry. `run()` and `stream()` dispatch into the
 * pattern chosen by the resolver (request > session > user > tenant
 * > global). `stream()` drives the generator through its streaming
 * variant so SSE proxies surface tokens incrementally.
 */

import type { Llm, Retrieval, Telemetry } from '@revex/core';
import { runWithWorkspaceAsync, type CollectionId, type WorkspaceId } from '@revex/core';

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
import type { User } from '@revex/core';

export interface OrchestratorOptions {
  readonly telemetry: Telemetry;
  readonly workspaceId: WorkspaceId;
  readonly defaultStrategy?: Strategy;
  readonly sessionOverrides?: Readonly<Record<string, unknown>>;
  readonly adapters?: { readonly graph?: StrandsAdapter; readonly swarm?: StrandsAdapter; readonly workflow?: StrandsAdapter };
  readonly agents: AgentRegistry;
  readonly tools: ToolRegistry;
  readonly llm?: Llm;
  readonly retrieval?: Retrieval;
  readonly model?: string;
}

export class Orchestrator {
  private readonly telemetry: Telemetry;
  private readonly workspaceId: WorkspaceId;
  private readonly defaultStrategy: Strategy;
  private readonly sessionOverrides: Readonly<Record<string, unknown>>;
  private readonly patterns: Record<StrategyShape['mode'], PatternBuilder>;
  private readonly adapter: StrandsAdapter;
  private readonly agents: AgentRegistry;
  private readonly tools: ToolRegistry;

  constructor(opts: OrchestratorOptions) {
    this.telemetry = opts.telemetry;
    this.workspaceId = opts.workspaceId;
    this.defaultStrategy = opts.defaultStrategy ?? resolveStrategy([]);
    this.sessionOverrides = opts.sessionOverrides ?? {};
    this.agents = opts.agents;
    this.tools = opts.tools;

    const adapter: StrandsAdapter = opts.llm && opts.retrieval && opts.model
      ? new InProcessAdapter({
          agents: this.agents,
          tools: this.tools,
          llm: opts.llm,
          retrieval: opts.retrieval,
          model: opts.model,
        })
      : (opts.adapters?.graph ?? new InProcessAdapter({ agents: this.agents, tools: this.tools, llm: opts.llm!, retrieval: opts.retrieval!, model: opts.model ?? 'gpt-4.1' }));

    this.patterns = {
      graph: buildGraph(opts.adapters?.graph ?? adapter),
      swarm: buildSwarm(opts.adapters?.swarm ?? adapter),
      workflow: buildWorkflow(opts.adapters?.workflow ?? adapter),
    };
    this.adapter = adapter;
  }

  /**
   * Wire the InProcessAdapter's streaming generator hook to the
   * supplied callback so the orchestrator can push incremental
   * 'answer_chunk' events as the LLM emits them. No-op for
   * adapters that don't expose the hook (e.g. the Strands SDK
   * before streaming support ships).
   */
  private installStreamingHook(onDelta: (delta: string) => void): void {
    this.adapter.useStreamingGenerator?.(onDelta);
  }

  public async run(req: OrchestratorRequest): Promise<OrchestratorResult> {
    const state = this.makeInvocationState(req);
    const mode = state.strategy.mode;
    return runWithWorkspaceAsync(
      {
        workspaceId: state.workspace_id,
        userId: state.user_id,
        isAdmin: state.is_admin,
        sessionId: state.session_id,
      },
      () => this.dispatch(mode, req, state),
    );
  }

  public async *stream(req: OrchestratorRequest): AsyncGenerator<PlannerEvent> {
    const state = this.makeInvocationState(req);
    const mode = state.strategy.mode;
    yield { kind: 'thought', step: 0, payload: { text: `mode=${mode} streaming` } };

    const queue: PlannerEvent[] = [];
    let resolveNext: (() => void) | null = null;
    let done = false;
    let thrownError: Error | null = null;
    const push = (ev: PlannerEvent): void => {
      queue.push(ev);
      if (resolveNext) {
        const r = resolveNext;
        resolveNext = null;
        r();
      }
    };

    /* Stream incremental deltas through the adapter's streaming
     * generator hook (when available) so the chat UI sees tokens
     * appear as the LLM emits them. */
    let stepCounter = 1;
    this.installStreamingHook((delta) => {
      push({ kind: 'answer_chunk', step: stepCounter++, payload: { delta } });
    });

    const task = runWithWorkspaceAsync(
      {
        workspaceId: state.workspace_id,
        userId: state.user_id,
        isAdmin: state.is_admin,
        sessionId: state.session_id,
      },
      async (): Promise<OrchestratorResult> => {
        try {
          const builder = this.patterns[mode];
          return await builder.run(req, state);
        } catch (e) {
          thrownError = e instanceof Error ? e : new Error(String(e));
          push({ kind: 'final', step: 99, payload: { answer: '', citations: [] } });
          throw thrownError;
        } finally {
          done = true;
          if (resolveNext) {
            const r = resolveNext;
            resolveNext = null;
            r();
          }
        }
      },
    );

    void task;

    while (true) {
      if (queue.length > 0) {
        const ev = queue.shift();
        if (ev) yield ev;
        continue;
      }
      if (done) {
        if (thrownError) throw thrownError;
        return;
      }
      await new Promise<void>((r) => {
        resolveNext = r;
      });
    }
  }

  public resolveInvocationState(req: OrchestratorRequest): InvocationState {
    return this.makeInvocationState(req);
  }

  private makeInvocationState(req: OrchestratorRequest): InvocationState {
    const userOverrides: StrategyOverrides = req.user ? extractUserOverrides(req.user) : {};
    const strategy = resolveStrategy([
      this.defaultStrategy,
      this.sessionOverrides['strategy'] as never,
      userOverrides,
    ]);
    return buildInvocationState({
      workspaceId: this.workspaceId,
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

const extractUserOverrides = (user: User): StrategyOverrides => {
  const meta = (user as unknown as { toJSON?: () => Record<string, unknown> }).toJSON?.() ?? {};
  const strat = meta['strategy'];
  if (!strat || typeof strat !== 'object') return {};
  return strat as StrategyOverrides;
};

export type { Citation, PlannerEvent };