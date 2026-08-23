/**
 * The Strands adapter boundary.
 *
 * Two implementations live behind the same contract:
 *  - `StrandsAdapter` — uses the real `strands-agents` package
 *    (loaded via dynamic import; missing package surfaces a clear
 *    MissingDepError at first use).
 *  - `InProcessAdapter` — a minimal in-process implementation used
 *    when `strands-agents` is not installed, for tests, and for the
 *    Phase 1 scaffold.
 *
 * Phase 1 ships only `InProcessAdapter` with the Graph/Swarm/Workflow
 * pattern builders. Phase 2 swaps the adapter once the Strands SDK
 * is published on npm and the team confirms the contract.
 */

import type { InvocationState, OrchestratorRequest, OrchestratorResult } from './types.js';

export interface StrandsAdapter {
  readonly name: string;
  runGraph(req: OrchestratorRequest, state: InvocationState): Promise<OrchestratorResult>;
  runSwarm(req: OrchestratorRequest, state: InvocationState): Promise<OrchestratorResult>;
  runWorkflow(req: OrchestratorRequest, state: InvocationState): Promise<OrchestratorResult>;
}