/**
 * Agent registry.
 *
 * The orchestrator resolves agents by stable string id. Concrete
 * agents register themselves via the registry, then the adapter
 * pulls them out by id at run time. Mirrors the Strands `Agent`
 * lookup semantics.
 */

import { ConfigurationError } from '@raghub/core';
import type { Hit } from '@raghub/core';

import type { InvocationState, OrchestratorRequest } from '../strands/types.js';

export interface ToolExecution {
  readonly ok: boolean;
  readonly content: string;
  readonly hits: readonly Hit[];
  readonly latencyMs: number;
}

export interface Agent {
  readonly id: string;
  retrieve(req: OrchestratorRequest, state: InvocationState): Promise<ToolExecution>;
  generate(
    req: OrchestratorRequest,
    hits: readonly Hit[],
    state: InvocationState,
  ): Promise<{ readonly answer: string }>;
}

export type AgentFactory = () => Agent;

export class AgentRegistry {
  private readonly agents = new Map<string, Agent>();

  public register(id: string, agent: Agent): void {
    if (this.agents.has(id)) {
      throw new ConfigurationError(`agent id already registered: ${id}`, { details: { id } });
    }
    this.agents.set(id, agent);
  }

  public require(id: string): Agent {
    const a = this.agents.get(id);
    if (!a) throw new ConfigurationError(`unknown agent id: ${id}`, { details: { id } });
    return a;
  }

  public ids(): readonly string[] {
    return [...this.agents.keys()];
  }
}