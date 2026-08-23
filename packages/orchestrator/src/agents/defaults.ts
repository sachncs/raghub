/**
 * Default agent implementations.
 *
 * `retriever` runs the real @raghub/core Retrieval pipeline (dense
 * + sparse + RRF) under the active RBAC. `generator` produces a
 * grounded answer using the real LLM, with token-by-token streaming
 * surfaced via the Agent contract.
 *
 * Phase 1: no PluggableTools/Swarm yet — those arrive when the
 * Strands SDK swap lands. Until then, the agents below ARE the
 * orchestrator's only callable unit.
 */

import type { ChatMessage, Hit, Llm, Retrieval } from '@raghub/core';
import { RaghubError, User } from '@raghub/core';

import type { Agent, ToolExecution } from './registry.js';
import type { InvocationState, OrchestratorRequest } from '../strands/types.js';

const synthesizeContext = (hits: readonly Hit[]): string =>
  hits
    .slice(0, 10)
    .map((h, i) => `[${i + 1}] (score=${h.score.toFixed(3)}) ${h.chunk.text}`)
    .join('\n');

const SYSTEM_PROMPT = `You are a precise, retrieval-grounded assistant. Use the numbered context snippets provided to answer the user's question. Cite sources inline like [1], [2] where the answer draws from a snippet. If the context is insufficient, say so explicitly rather than fabricating an answer.`;

const buildMessages = (
  req: OrchestratorRequest,
  history: readonly ChatMessage[],
  hits: readonly Hit[],
): ChatMessage[] => {
  const systemContent = `${SYSTEM_PROMPT}\n\nContext:\n${synthesizeContext(hits)}`;
  const messages: ChatMessage[] = [{ role: 'system', content: systemContent }];
  for (const h of history) {
    if (h.role === 'system') continue;
    messages.push(h);
  }
  messages.push({ role: 'user', content: req.question });
  return messages;
};

const rebuildUserFromState = (state: InvocationState): User =>
  new User({
    id: state.user_id ?? ('' as never),
    tenantId: state.tenant_id,
    email: '',
    role: state.is_admin ? 'admin' : 'member',
    allowedCompanies: state.rbac_filter.allowedCompanies,
    createdAt: new Date(),
  });

export interface RetrieverAgentDeps {
  readonly retrieval: Retrieval;
  readonly llm: Llm;
  readonly model: string;
}

export const createRetrieverAgent = (deps: RetrieverAgentDeps): Agent => ({
  id: 'retriever',
  async retrieve(req, state) {
    if (!state.user_id) {
      return { ok: false, content: 'no authenticated user in invocation state', hits: [], latencyMs: 0 };
    }
    const start = Date.now();
    try {
      const user = rebuildUserFromState(state);
      const hits = await deps.retrieval.retrieve(user, req.question, state.strategy.k);
      return {
        ok: true,
        content: JSON.stringify(hits.map((h: Hit) => ({ id: h.chunk.id, score: h.score }))),
        hits,
        latencyMs: Date.now() - start,
      };
    } catch (e) {
      return {
        ok: false,
        content: e instanceof Error ? e.message : String(e),
        hits: [],
        latencyMs: Date.now() - start,
      };
    }
  },
  async generate() {
    throw new RaghubError('pipeline_error', 'retriever agent cannot generate');
  },
});

export interface GeneratorAgentDeps {
  readonly llm: Llm;
  readonly model: string;
}

export const createGeneratorAgent = (deps: GeneratorAgentDeps): Agent => ({
  id: 'generator',
  async retrieve() {
    throw new RaghubError('pipeline_error', 'generator agent cannot retrieve');
  },
  async generate(req, hits, _state) {
    const history: ChatMessage[] = (req.history ?? []).map((h) => ({
      role: h.role,
      content: h.content,
    }));
    const messages = buildMessages(req, history, hits);
    const result = await deps.llm.generate({
      model: deps.model,
      messages,
      temperature: 0,
    });
    return { answer: result.content };
  },
});

/**
 * Streaming variant of the generator agent. Yields tokens through an
 * event-emitting callback so the adapter can convert them into
 * PlannerEvents without buffering the full response.
 */

export interface StreamingGeneratorAgentDeps extends GeneratorAgentDeps {
  readonly onDelta: (delta: string) => void;
}

export const createStreamingGeneratorAgent = (deps: StreamingGeneratorAgentDeps): Agent => ({
  id: 'generator',
  async retrieve() {
    throw new RaghubError('pipeline_error', 'generator agent cannot retrieve');
  },
  async generate(req, hits, _state) {
    const history: ChatMessage[] = (req.history ?? []).map((h) => ({
      role: h.role,
      content: h.content,
    }));
    const messages = buildMessages(req, history, hits);
    let final = '';
    for await (const chunk of deps.llm.stream({
      model: deps.model,
      messages,
      temperature: 0,
    })) {
      if (chunk.delta) {
        final += chunk.delta;
        deps.onDelta(chunk.delta);
      }
    }
    return { answer: final };
  },
});