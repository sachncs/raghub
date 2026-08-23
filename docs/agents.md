# raghub — agent runtime

`RagAgent` is the multi-agent root that runs a single user turn.
Modeled on Strands Agents' Swarm pattern + MultiAgentGraph
"agents-as-tools" composition.

## Components

| name | role |
| --- | --- |
| `RagAgent` | orchestrator over `SubAgent`s + a generator agent |
| `SubAgent` (vector / keyword / graph / trace / memory / web / summary) | one specialist retrieval role |
| `AgentRegistry` | name → `Agent` lookup, used for the summarizer |
| `ToolRegistry` | name → `Tool` lookup, used by Strands adapters |
| `HookRegistry` | 6 hook points, swallow errors so telemetry never breaks the loop |
| `SessionState` | per-session key/value bag |

## Hook surface

```
beforeRetrieve(role, state)
afterRetrieve(role, hits, state, error?)
beforeLLM(request, state)
afterLLM(request, answer, state)
beforeTool(name, args, state)
afterTool(name, result, state)
```

The structured `HookRegistry` (`AgentHookBus`) is independent of the
legacy per-instance `RagAgentHooks` interface; both fire on every
turn. Hook errors are swallowed so a misbehaving subscriber never
breaks the agent loop.

## Retry

`RetryStrategy` defaults to 3 attempts with exponential backoff
(base 200ms, cap 4s) plus 25% jitter. Every `SubAgent.retrieve` call
runs through `withRetry`; the policy is configurable per `RagAgent`.

## Conversation management

When `history.length > turnLimit` (default 20), `RagAgent` asks the
summarizer agent (id configurable) to collapse the conversation
into one `system` message and keeps the last two turns verbatim.
This mirrors Strands' `ConversationManager.condense` strategy.

## Session state

`createSessionState(initial)` returns a per-session key/value bag
that the orchestrator can read at any hook point. The hook layer
can mutate the bag freely; the agent loop never inspects it.

## Building a sub-agent set

```ts
import {
  RagAgent,
  AgentRegistry,
  buildDefaultSubAgents,
  createGeneratorAgent,
  HookRegistry,
} from '@raghub/orchestrator';

const subAgents = buildDefaultSubAgents({
  retrieval, embedder, vectorStore,
  graphStore, traceCorpus, webSearch, memory,
});

const agents = new AgentRegistry();
agents.register('generator', createGeneratorAgent({ llm, model: 'gpt-4.1' }));

const hooks = new HookRegistry();
hooks.on('afterRetrieve', ({ role, hits }) => console.log(role, hits.length));

const rag = new RagAgent({ agents, subAgents, hookBus: hooks.bus });

const result = await rag.run(
  { question: 'what is raghub?', user: null, sessionId: null },
  invocationState,
);
```

## SSE plumbing

`Orchestrator.stream()` already converts the agent's `PlannerEvent`s
into `event:` lines. Each `afterRetrieve` on the hook bus maps to
`tool_result` events the chat UI consumes.