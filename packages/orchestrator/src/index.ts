/**
 * @raghub/orchestrator — public surface.
 *
 * The Orchestrator façade, the agent/tool registries, the three
 * pattern builders, and the strategy resolver.
 */

export { Orchestrator } from './orchestrator.js';
export type { OrchestratorOptions } from './orchestrator.js';

export { AgentRegistry } from './agents/registry.js';
export type { Agent, ToolExecution, AgentFactory } from './agents/registry.js';
export {
  createGeneratorAgent,
  createRetrieverAgent,
} from './agents/defaults.js';

export { ToolRegistry } from './tools/registry.js';
export type { Tool, ToolContext, ToolResult } from './tools/registry.js';
export {
  createGraphSearchTool,
  createHybridSearchTool,
  createKeywordSearchTool,
  createSummarySearchTool,
  createTodayTool,
  createTraceSearchTool,
  createVectorSearchTool,
  createWebSearchTool,
  registerBuiltInTools,
} from './tools/built-in.js';

export { buildGraph, buildSwarm, buildWorkflow } from './patterns/builders.js';
export type { PatternBuilder } from './patterns/builders.js';
export { resolveStrategy } from './patterns/strategy.js';
export type { StrategyOverrides } from './patterns/strategy.js';

export type {
  Citation,
  InvocationState,
  OrchestratorRequest,
  OrchestratorResult,
  PlannerEvent,
  Strategy,
} from './strands/types.js';
export { buildInvocationState } from './strands/invocation-state.js';
export type { StrandsAdapter } from './strands/adapter.js';
export { InProcessAdapter } from './strands/in-process-adapter.js';