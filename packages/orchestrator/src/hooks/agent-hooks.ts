/**
 * Agent hooks — structured telemetry / observability surface.
 *
 * Six hook points model the agent loop:
 *   1. beforeRetrieve(role, state)         -> sub-agent about to run
 *   2. afterRetrieve(role, hits, state)    -> sub-agent returned
 *   3. beforeLLM(req, state)               -> LLM call starts
 *   4. afterLLM(req, answer, state)        -> LLM call finished
 *   5. beforeTool(name, args, state)       -> tool call starts
 *   6. afterTool(name, result, state)      -> tool call finished
 *
 * The hooks are designed for both human telemetry (log spans,
 * trace IDs, latencies) and machine telemetry (counters, gauges).
 * Implementations may throw — RagAgent catches and continues.
 */

import type { Hit } from '@revex/core';

import type { InvocationState, OrchestratorRequest } from '../strands/types.js';

export interface ToolResultLike {
  readonly ok: boolean;
  readonly content: string;
}

export interface AgentHookEvents {
  readonly beforeRetrieve: { role: string; state: InvocationState };
  readonly afterRetrieve: { role: string; hits: readonly Hit[]; state: InvocationState; error?: string };
  readonly beforeLLM: { request: OrchestratorRequest; state: InvocationState };
  readonly afterLLM: { request: OrchestratorRequest; answer: string; state: InvocationState };
  readonly beforeTool: { name: string; args: Readonly<Record<string, unknown>>; state: InvocationState };
  readonly afterTool: { name: string; result: ToolResultLike; state: InvocationState };
}

export type AgentHook<K extends keyof AgentHookEvents> = (event: AgentHookEvents[K]) => void | Promise<void>;

export interface AgentHookBus {
  readonly beforeRetrieve: AgentHook<'beforeRetrieve'>[];
  readonly afterRetrieve: AgentHook<'afterRetrieve'>[];
  readonly beforeLLM: AgentHook<'beforeLLM'>[];
  readonly afterLLM: AgentHook<'afterLLM'>[];
  readonly beforeTool: AgentHook<'beforeTool'>[];
  readonly afterTool: AgentHook<'afterTool'>[];
}

export class HookRegistry {
  public readonly bus: AgentHookBus = {
    beforeRetrieve: [],
    afterRetrieve: [],
    beforeLLM: [],
    afterLLM: [],
    beforeTool: [],
    afterTool: [],
  };

  public on<K extends keyof AgentHookEvents>(kind: K, fn: AgentHook<K>): () => void {
    const list = this.bus[kind] as AgentHook<K>[];
    list.push(fn);
    return () => {
      const i = list.indexOf(fn);
      if (i >= 0) list.splice(i, 1);
    };
  }

  public async emit<K extends keyof AgentHookEvents>(kind: K, event: AgentHookEvents[K]): Promise<void> {
    const list = this.bus[kind] as AgentHook<K>[];
    for (const fn of list) {
      try {
        await fn(event);
      } catch {
        // Hook failures are swallowed — telemetry must never break the agent loop.
      }
    }
  }
}

/**
 * Default logging hook — emits one structured log entry per event.
 * Used by the HTTP layer to surface spans in dev consoles; a
 * production deployment should swap it for OTLP / Prometheus.
 */
export const createConsoleLoggingHook = (): { hooks: AgentHookBus } => {
  const reg = new HookRegistry();
  reg.on('beforeRetrieve', ({ role }) => {
    // eslint-disable-next-line no-console
    console.log(`[hook] beforeRetrieve role=${role}`);
  });
  reg.on('afterRetrieve', ({ role, hits }) => {
    // eslint-disable-next-line no-console
    console.log(`[hook] afterRetrieve role=${role} hits=${hits.length}`);
  });
  reg.on('beforeLLM', ({ request }) => {
    // eslint-disable-next-line no-console
    console.log(`[hook] beforeLLM question_len=${request.question.length}`);
  });
  reg.on('afterLLM', ({ answer }) => {
    // eslint-disable-next-line no-console
    console.log(`[hook] afterLLM answer_len=${answer.length}`);
  });
  return { hooks: reg.bus };
};