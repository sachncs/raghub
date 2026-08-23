/**
 * Pattern builders — Graph, Swarm, Workflow.
 *
 * Each builder composes an adapter call into a configuration object
 * the orchestrator consumes. They share the same call surface;
 * picking a mode is the orchestrator's only difference.
 */

import type { InvocationState, OrchestratorRequest, OrchestratorResult, Strategy } from '../strands/types.js';
import type { StrandsAdapter } from '../strands/adapter.js';

export interface PatternBuilder {
  readonly mode: Strategy['mode'];
  run(req: OrchestratorRequest, state: InvocationState): Promise<OrchestratorResult>;
}

export const buildGraph = (adapter: StrandsAdapter): PatternBuilder => ({
  mode: 'graph',
  run: (req, state) => adapter.runGraph(req, state),
});

export const buildSwarm = (adapter: StrandsAdapter): PatternBuilder => ({
  mode: 'swarm',
  run: (req, state) => adapter.runSwarm(req, state),
});

export const buildWorkflow = (adapter: StrandsAdapter): PatternBuilder => ({
  mode: 'workflow',
  run: (req, state) => adapter.runWorkflow(req, state),
});