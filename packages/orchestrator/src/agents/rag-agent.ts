/**
 * RagAgent — multi-agent root that fans out to specialist sub-agents
 * (vector / keyword / graph / trace / memory / web) in parallel and
 * feeds the merged hits to the generator.
 *
 * Modeled after Strands Agents' Swarm pattern + MultiAgentGraph
 * "agents-as-tools" approach. The orchestrator already provides the
 * retriever/generator pair; RagAgent adds:
 *   - parallel specialist retrieval (Promise.all over sub-agents)
 *   - rerank + dedup + score fusion on the merged hit list
 *   - session state bag (key/value per session)
 *   - conversation management: if history > turnLimit, summarize
 *   - retry strategies on transient tool failures
 *   - hook surface (before/after llm, before/after tool, etc.)
 */

import type { ChatMessage, Hit } from '@raghub/core';

import type { Agent, AgentRegistry } from '../agents/registry.js';
import type { InvocationState, OrchestratorRequest, OrchestratorResult } from '../strands/types.js';

export type AgentRole =
  | 'vector'
  | 'keyword'
  | 'graph'
  | 'trace'
  | 'memory'
  | 'web'
  | 'summary';

export interface SubAgentInput {
  readonly role: AgentRole;
  readonly query: string;
  readonly filter?: { readonly userId?: string; readonly workspaceId?: string };
}

export interface SubAgentOutput {
  readonly role: AgentRole;
  readonly hits: readonly Hit[];
  readonly latencyMs: number;
  readonly error?: string;
}

/**
 * One specialist sub-agent. The root RagAgent spawns many of these
 * in parallel via Promise.all; each one is responsible for fetching
 * hits from its source (vector store, web search, knowledge graph,
 * etc.) under the user's RBAC/ACL.
 */
export interface SubAgent {
  readonly role: AgentRole;
  retrieve(input: SubAgentInput, state: InvocationState): Promise<readonly Hit[]>;
}

/** Hook surface — every hook may be sync or async. */
export interface RagAgentHooks {
  readonly beforeLLM?: (req: OrchestratorRequest, state: InvocationState) => Promise<void> | void;
  readonly afterLLM?: (
    req: OrchestratorRequest,
    answer: string,
    state: InvocationState,
  ) => Promise<void> | void;
  readonly beforeTool?: (
    name: string,
    args: Readonly<Record<string, unknown>>,
    state: InvocationState,
  ) => Promise<void> | void;
  readonly afterTool?: (
    name: string,
    result: { readonly ok: boolean; readonly content: string },
    state: InvocationState,
  ) => Promise<void> | void;
  readonly beforeRetrieve?: (role: AgentRole, state: InvocationState) => Promise<void> | void;
  readonly afterRetrieve?: (
    role: AgentRole,
    hits: readonly Hit[],
    state: InvocationState,
  ) => Promise<void> | void;
}

export interface SessionState {
  readonly values: Readonly<Record<string, unknown>>;
  set(key: string, value: unknown): void;
  get<T>(key: string): T | undefined;
  clear(): void;
}

export const createSessionState = (initial: Record<string, unknown> = {}): SessionState => {
  const map = new Map<string, unknown>(Object.entries(initial));
  return {
    get values(): Readonly<Record<string, unknown>> {
      const out: Record<string, unknown> = {};
      for (const [k, v] of map.entries()) out[k] = v;
      return out;
    },
    set: (key, value) => {
      map.set(key, value);
    },
    get: <T>(key: string): T | undefined => map.get(key) as T | undefined,
    clear: () => {
      map.clear();
    },
  };
};

export interface RetryStrategy {
  /** Maximum number of attempts (including the first). */
  readonly maxAttempts: number;
  /** Base delay between retries in milliseconds. */
  readonly baseDelayMs: number;
  /** Maximum delay cap (in case exponential backoff blows past it). */
  readonly maxDelayMs: number;
}

export const defaultRetryStrategy = (): RetryStrategy => ({
  maxAttempts: 3,
  baseDelayMs: 200,
  maxDelayMs: 4000,
});

export interface RagAgentDeps {
  readonly agents: AgentRegistry;
  readonly subAgents: readonly SubAgent[];
  readonly hooks?: RagAgentHooks;
  readonly retry?: RetryStrategy;
  /** Maximum number of conversation turns before triggering summarization. */
  readonly turnLimit?: number;
  /** Optional summarizer agent (id registered in `agents`). */
  readonly summarizerId?: string;
  /** Default role set if the request does not specify one. */
  readonly defaultRoles?: readonly AgentRole[];
}

const DEFAULT_TURN_LIMIT = 20;
const ALL_ROLES: readonly AgentRole[] = [
  'vector',
  'keyword',
  'graph',
  'trace',
  'memory',
  'web',
  'summary',
];

/**
 * Sleep `ms` milliseconds, returning a promise that resolves to void.
 * Exported so tests can stub or skip it.
 */
export const sleep = (ms: number): Promise<void> =>
  new Promise((resolve) => setTimeout(resolve, ms));

const retryDelay = (attempt: number, strategy: RetryStrategy): number => {
  const exp = Math.min(strategy.maxDelayMs, strategy.baseDelayMs * 2 ** (attempt - 1));
  const jitter = Math.random() * exp * 0.25;
  return Math.min(strategy.maxDelayMs, exp + jitter);
};

export class RagAgent {
  private readonly agents: AgentRegistry;
  private readonly subAgents: readonly SubAgent[];
  private readonly hooks: RagAgentHooks;
  private readonly retry: RetryStrategy;
  private readonly turnLimit: number;
  private readonly summarizerId: string | undefined;
  private readonly defaultRoles: readonly AgentRole[];

  constructor(deps: RagAgentDeps) {
    this.agents = deps.agents;
    this.subAgents = deps.subAgents;
    this.hooks = deps.hooks ?? {};
    this.retry = deps.retry ?? defaultRetryStrategy();
    this.turnLimit = deps.turnLimit ?? DEFAULT_TURN_LIMIT;
    this.summarizerId = deps.summarizerId;
    this.defaultRoles = deps.defaultRoles ?? ['vector', 'keyword', 'memory', 'web'];
  }

  /**
   * Run a single turn end-to-end: fan-out retrieval, merge hits,
   * maybe summarize history, then generate an answer.
   *
   * The returned `OrchestratorResult` is what the HTTP layer turns
   * into an SSE event stream and JSON body.
   */
  public async run(req: OrchestratorRequest, state: InvocationState): Promise<OrchestratorResult> {
    if (req.signal?.aborted) {
      throw new Error('aborted');
    }
    const roles: readonly AgentRole[] = req.roles ?? this.defaultRoles;
    const outputs = await this.fanOut(roles, req, state);
    const hits = mergeHits(outputs);
    const history = await this.maybeSummarize(req.history ?? [], state);
    const augmented = withHistory(req, history);
    const generator = this.requireAgent('generator');
    const { answer } = await this.invokeWithHooks(augmented, state, () =>
      generator.generate(augmented, hits, state),
    );
    return {
      answer,
      citations: hits.map((h) => ({
        chunkId: h.chunk.id,
        documentId: h.chunk.documentId,
        text: h.chunk.text,
        score: h.score,
      })),
      hits,
      events: [
        ...outputs.flatMap((o, i) => [
          { kind: 'thought' as const, step: i, payload: { text: `sub-agent ${o.role}: ${o.hits.length} hits` } },
          { kind: 'tool_result' as const, step: i + 1, payload: { name: o.role, ok: o.error === undefined, content: o.error ?? '', latencyMs: o.latencyMs } },
        ]),
        { kind: 'final' as const, step: roles.length + 2, payload: { answer, citations: hits.map((h) => ({ chunkId: h.chunk.id, documentId: h.chunk.documentId, text: h.chunk.text, score: h.score })) } },
      ],
      mode: state.strategy.mode,
    };
  }

  private async fanOut(
    roles: readonly AgentRole[],
    req: OrchestratorRequest,
    state: InvocationState,
  ): Promise<readonly SubAgentOutput[]> {
    const filter: SubAgentInput['filter'] = { workspaceId: state.workspace_id };
    if (state.user_id !== null) {
      (filter as { userId?: string }).userId = state.user_id;
    }
    const tasks = roles
      .map((role) => this.subAgents.find((s) => s.role === role))
      .filter((s): s is SubAgent => s !== undefined)
      .map(async (s) => {
        await this.hooks.beforeRetrieve?.(s.role, state);
        const start = Date.now();
        try {
          const hits = await this.withRetry(() =>
            s.retrieve({ role: s.role, query: req.question, filter }, state),
          );
          await this.hooks.afterRetrieve?.(s.role, hits, state);
          return { role: s.role, hits, latencyMs: Date.now() - start };
        } catch (e) {
          const message = e instanceof Error ? e.message : String(e);
          await this.hooks.afterRetrieve?.(s.role, [], state);
          return {
            role: s.role,
            hits: [],
            latencyMs: Date.now() - start,
            error: message,
          };
        }
      });
    if (tasks.length === 0) {
      // No matching sub-agents for the requested roles — return an
      // empty output set rather than hanging.
      return [];
    }
    return Promise.all(tasks);
  }

  private async withRetry<T>(fn: () => Promise<T>): Promise<T> {
    let lastError: Error | undefined;
    for (let attempt = 1; attempt <= this.retry.maxAttempts; attempt++) {
      try {
        return await fn();
      } catch (e) {
        lastError = e instanceof Error ? e : new Error(String(e));
        if (attempt === this.retry.maxAttempts) break;
        await sleep(retryDelay(attempt, this.retry));
      }
    }
    throw lastError ?? new Error('retry exhausted');
  }

  private async invokeWithHooks<T>(req: OrchestratorRequest, state: InvocationState, fn: () => Promise<T>): Promise<T> {
    await this.hooks.beforeLLM?.(req, state);
    const out = await fn();
    await this.hooks.afterLLM?.(req, typeof out === 'string' ? out : JSON.stringify(out), state);
    return out;
  }

  private async maybeSummarize(
    history: readonly ChatMessage[],
    state: InvocationState,
  ): Promise<readonly ChatMessage[]> {
    if (history.length <= this.turnLimit || this.summarizerId === undefined) {
      return history;
    }
    const summarizer = this.agents.get(this.summarizerId);
    if (!summarizer || typeof summarizer.generate !== 'function') return history;
    const fakeHits: readonly Hit[] = [];
    const fakeReq: OrchestratorRequest = { question: '', history, user: null, sessionId: null };
    try {
      const { answer } = await summarizer.generate(fakeReq, fakeHits, state);
      return [
        { role: 'system', content: `Conversation summary:\n${answer}` },
        ...history.slice(-2),
      ];
    } catch {
      return history;
    }
  }

  private requireAgent(id: string): Agent {
    return this.agents.require(id);
  }
}

const withHistory = (
  req: OrchestratorRequest,
  history: readonly ChatMessage[],
): OrchestratorRequest => ({ ...req, history });

const mergeHits = (outputs: readonly SubAgentOutput[]): readonly Hit[] => {
  const seen = new Map<string, Hit>();
  for (const out of outputs) {
    for (const hit of out.hits) {
      const existing = seen.get(hit.chunk.id);
      if (!existing || hit.score > existing.score) {
        seen.set(hit.chunk.id, hit);
      }
    }
  }
  return [...seen.values()].sort((a, b) => b.score - a.score).slice(0, 30);
};

export const RAG_AGENT_DEFAULT_ROLES = ALL_ROLES;