# Orchestrator & agents

`@revex/orchestrator` is the Strands-shaped execution layer. It wraps an
adapter, an agent registry, and a tool registry behind one `Orchestrator`
façade.

## Orchestrator

```ts
import { Orchestrator } from '@revex/orchestrator';

const orch = new Orchestrator({
  telemetry,
  workspaceId,
  agents,
  tools,
  llm,
  retrieval,
  model,
});
```

Methods:

- `run(req): Promise<OrchestratorResult>` — one-shot answer.
- `stream(req): AsyncGenerator<PlannerEvent>` — streaming events.
- `resolveInvocationState(req): InvocationState` — resolve the effective
  strategy/session state.

`OrchestratorRequest` carries `question`, `user`, `sessionId`, optional
`history`, `signal`, `roles`, and `overrides`. `OrchestratorResult` has
`answer`, `citations`, `hits`, `events`, and `mode`.

## Strategy resolution

The effective `Strategy` is resolved from layers in order:
**request > session > user > tenant > global**. `resolveStrategy(layers)`
merges `StrategyOverrides` (partials). Default strategy:

```ts
{
  mode: 'graph',
  hybrid: { denseWeight: 0.6, sparseWeight: 0.4, rrfK: 60, colbert: false },
  ordering: 'standard',
  k: 10,
  reranker: 'identity',
  multimodal: { enabled: false },
  traceCorpus: { enabled: false, representation: 'semantic', topK: 5 },
}
```

`mode` is one of `graph | swarm | workflow | deep_research`.

## Pattern builders

- `buildGraph(adapter)` — graph-based retrieval.
- `buildSwarm(adapter)` — multi-agent swarm.
- `buildWorkflow(adapter)` — deterministic workflow.
- `buildDeepResearch({ llm, tools, model })` — deep research (ReAct loop,
  context builder, citation extraction). Used internally by `Orchestrator`.

## Adapters

The `StrandsAdapter` boundary defines `runGraph` / `runSwarm` / `runWorkflow`
(+ optional `useStreamingGenerator`). Phase 1 ships `InProcessAdapter`, a
retriever→generator pipeline that implements all three modes. `buildInvocationState`
builds the `Strands-shaped` record propagated to every node and tool.

## Agents

`AgentRegistry` maps string ids to `Agent`s (`register`, `require`, `get`,
`ids`). Built-ins:

- `createRetrieverAgent` (`retriever`) — runs the retrieval pipeline.
- `createGeneratorAgent` (`generator`) — calls the LLM with context.
- `createStreamingGeneratorAgent` — streams via `llm.stream()`.

### `RagAgent`

The multi-agent root that fans out to specialist sub-agents, merges hits, and
generates. `RagAgentDeps` wires the registry, sub-agents, hooks, a retry
strategy, turn limit, summarizer id, and default roles.

`createSessionState(initial)` returns a per-session `SessionState` key/value
bag. `defaultRetryStrategy()` is `{ maxAttempts: 3, baseDelayMs: 200,
maxDelayMs: 4000 }`.

### Sub-agents

`buildDefaultSubAgents(deps)` wires these roles:

| Role | Builder |
|---|---|
| `vector` | `buildVectorSubAgent` |
| `keyword` | `buildKeywordSubAgent` |
| `graph` | `buildGraphSubAgent` |
| `trace` | `buildTraceSubAgent` |
| `web` | `buildWebSubAgent` |
| `memory` | `buildMemorySubAgent` |
| `summary` | `buildSummarySubAgent` |

### ReAct agent

`createReActAgent(deps)` implements a JSON tool-call loop with a step budget
(default 5). `parseTurn` and `renderSystemPrompt` drive the planner.

## Hooks

`HookRegistry` exposes a `bus` with six event kinds:

- `beforeRetrieve` / `afterRetrieve`
- `beforeLLM` / `afterLLM`
- `beforeTool` / `afterTool`

`on(kind, fn)` registers a hook and returns an unsubscribe function; `emit`
dispatches. `createConsoleLoggingHook()` provides a console logger.

## Tools

`ToolRegistry` registers `Tool`s (`register`, `require`, `names`). Built-ins:

| Tool creator | Tool name | Description |
|---|---|---|
| `createHybridSearchTool` | `hybrid_search` | Dense + BM25 fused hybrid. |
| `createVectorSearchTool` | `vector_search` | Cosine-similarity over chunks. |
| `createKeywordSearchTool` | `keyword_search` | BM25 via FTS5. |
| `createTodayTool` | `today` | Current UTC date. |
| `createWebSearchTool` | `web_search` | Web search (DuckDuckGo default). |
| `createTraceSearchTool` | `trace_search` | Retrieve thinking traces. |
| `createSummarySearchTool` | `summary_search` | RAPTOR summary index. |
| `createGraphSearchTool` | `graph_search` | GraphRAG entity search. |

`registerBuiltInTools(registry, deps)` registers the four core tools
unconditionally and the four optional ones only when their dependencies are
provided.

## Deep research

`runDeepResearch(input, deps)` runs a ReAct loop with a context builder and
citation extraction. `DeepResearchInput` carries `question`, `invocationState`,
`history`, `sessionId`. `DeepResearchOutput` reports `answer`, `citations`,
`events`, `toolCalls`, `tokens`, and `contextStats`.

## Query pipeline

- `QueryCache` — in-memory TTL cache (default 60s) keyed by question +
  workspace + user + top-K.
- `PipelineRouter` — resolves `query | ingest | ingest_then_query` from
  request → session → user → tenant default.
- `shapeContext` — renders `QueryContextResult` (numbered blocks + citations +
  truncation).