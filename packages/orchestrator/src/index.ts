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
  createStreamingGeneratorAgent,
} from './agents/defaults.js';

export { RagAgent } from './agents/rag-agent.js';
export type {
  AgentRole,
  SubAgent,
  SubAgentInput,
  SubAgentOutput,
  RagAgentHooks,
  RagAgentDeps,
  RetryStrategy,
  SessionState,
} from './agents/rag-agent.js';
export {
  defaultRetryStrategy,
  createSessionState,
  RAG_AGENT_DEFAULT_ROLES,
} from './agents/rag-agent.js';

export {
  buildVectorSubAgent,
  buildKeywordSubAgent,
  buildGraphSubAgent,
  buildTraceSubAgent,
  buildWebSubAgent,
  buildMemorySubAgent,
  buildSummarySubAgent,
  buildDefaultSubAgents,
} from './agents/sub-agents.js';
export type { SubAgentDeps } from './agents/sub-agents.js';

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