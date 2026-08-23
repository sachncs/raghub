/**
 * Built-in tools — the eight that mirror the legacy Python tool
 * surface. Each is a thin wrapper around @raghub/core capabilities
 * with the same JSON schema contract the Strands `@tool`
 * decorator produces.
 */

import type { Hit, Retrieval, VectorStore } from '@raghub/core';
import { allowedCompanyFilter, RaghubError, User } from '@raghub/core';
import type { Embedder } from '@raghub/core';

import type { Tool, ToolContext, ToolRegistry, ToolResult } from './registry.js';
import type { InvocationState } from '../strands/types.js';
void (null as unknown as ToolContext);

const okResult = (content: string, data?: Record<string, unknown>): ToolResult => {
  const r: { ok: true; content: string; latencyMs: number; data?: Readonly<Record<string, unknown>> } = {
    ok: true,
    content,
    latencyMs: 0,
  };
  if (data !== undefined) r.data = data;
  return r;
};

const errResult = (error: string, latencyMs: number): ToolResult => ({
  ok: false,
  content: '',
  error,
  latencyMs,
});

const wrap = async (
  start: number,
  fn: () => Promise<ToolResult>,
): Promise<ToolResult> => {
  try {
    const r = await fn();
    return { ...r, latencyMs: r.latencyMs === 0 ? Date.now() - start : r.latencyMs };
  } catch (e) {
    return errResult(e instanceof Error ? e.message : String(e), Date.now() - start);
  }
};

const requireUser = (state: InvocationState): User => {
  if (!state.user_id) throw new RaghubError('authorization_error', 'tool requires an authenticated user');
  const tenantId = state.tenant_id;
  const userId = state.user_id;
  return new User({
    id: userId,
    tenantId,
    email: '',
    role: state.is_admin ? ('admin' as never) : ('member' as never),
    allowedCompanies: state.rbac_filter.allowedCompanies,
    createdAt: new Date(),
  });
};

export const createHybridSearchTool = (
  retrieval: Retrieval,
  store: VectorStore,
): Tool => ({
  name: 'hybrid_search',
  description: 'Dense + BM25 fused hybrid retrieval scoped to the active tenant and user.',
  jsonSchema: {
    type: 'object',
    properties: { question: { type: 'string' } },
    required: ['question'],
  },
  async execute(args, ctx) {
    return wrap(Date.now(), async () => {
      const user = requireUser(ctx.invocationState);
      const q = String(args['question'] ?? '');
      const hits = await retrieval.retrieve(user, q, ctx.invocationState.strategy.k);
      return okResult(JSON.stringify(hits.map((h: Hit) => ({ id: h.chunk.id, score: h.score, text: h.chunk.text }))), { hits });
    });
  },
});

export const createVectorSearchTool = (embedder: Embedder, store: VectorStore): Tool => ({
  name: 'vector_search',
  description: 'Cosine-similarity search over the tenant-scoped chunk store.',
  jsonSchema: {
    type: 'object',
    properties: { question: { type: 'string' }, top_k: { type: 'number' } },
    required: ['question'],
  },
  async execute(args, ctx) {
    return wrap(Date.now(), async () => {
      const user = requireUser(ctx.invocationState);
      const q = String(args['question'] ?? '');
      const topK = Number(args['top_k'] ?? ctx.invocationState.strategy.k);
      const filter = allowedCompanyFilter(user);
      const vec = await embedder.embedQuery(q);
      const hits = await store.searchVector({ vector: vec, topK, filter });
      return okResult(JSON.stringify(hits.map((h: Hit) => ({ id: h.chunk.id, score: h.score }))), { hits });
    });
  },
});

export const createKeywordSearchTool = (store: VectorStore): Tool => ({
  name: 'keyword_search',
  description: 'BM25 keyword search via SQLite FTS5.',
  jsonSchema: {
    type: 'object',
    properties: { query: { type: 'string' }, top_k: { type: 'number' } },
    required: ['query'],
  },
  async execute(args, ctx) {
    return wrap(Date.now(), async () => {
      const user = requireUser(ctx.invocationState);
      const q = String(args['query'] ?? '');
      const topK = Number(args['top_k'] ?? ctx.invocationState.strategy.k);
      const filter = allowedCompanyFilter(user);
      const hits = await store.searchKeyword({ query: q, topK, filter });
      return okResult(JSON.stringify(hits), { hits });
    });
  },
});

export const createTodayTool = (): Tool => ({
  name: 'today',
  description: 'Return the current UTC date.',
  jsonSchema: { type: 'object', properties: {} },
  async execute() {
    return wrap(Date.now(), async () => okResult(new Date().toISOString()));
  },
});

export const createWebSearchTool = (): Tool => ({
  name: 'web_search',
  description: 'Web search (stub — Phase 2 wires a real provider).',
  jsonSchema: {
    type: 'object',
    properties: { query: { type: 'string' } },
    required: ['query'],
  },
  async execute() {
    return wrap(Date.now(), async () => okResult('[]'));
  },
});

export const createTraceSearchTool = (): Tool => ({
  name: 'trace_search',
  description: 'Retrieve transformed thinking traces for a query (Phase 2 wires the corpus).',
  jsonSchema: {
    type: 'object',
    properties: { query: { type: 'string' }, representation: { type: 'string' } },
    required: ['query'],
  },
  async execute(args) {
    return wrap(Date.now(), async () => {
      const q = String(args['query'] ?? '');
      return okResult(`[] /* trace search not enabled; question="${q.slice(0, 80)}" */`);
    });
  },
});

export const createSummarySearchTool = (): Tool => ({
  name: 'summary_search',
  description: 'RAPTOR summary search (Phase 2 wires the tree index).',
  jsonSchema: {
    type: 'object',
    properties: { question: { type: 'string' } },
    required: ['question'],
  },
  async execute() {
    return wrap(Date.now(), async () => okResult('[]'));
  },
});

export const createGraphSearchTool = (): Tool => ({
  name: 'graph_search',
  description: 'GraphRAG entity/community search (Phase 3 wires the graph store).',
  jsonSchema: {
    type: 'object',
    properties: { question: { type: 'string' } },
    required: ['question'],
  },
  async execute() {
    return wrap(Date.now(), async () => okResult('[]'));
  },
});

export const registerBuiltInTools = (
  registry: ToolRegistry,
  deps: { retrieval: Retrieval; embedder: Embedder; store: VectorStore },
): void => {
  const tools: Tool[] = [
    createHybridSearchTool(deps.retrieval, deps.store),
    createVectorSearchTool(deps.embedder, deps.store),
    createKeywordSearchTool(deps.store),
    createTodayTool(),
    createWebSearchTool(),
    createTraceSearchTool(),
    createSummarySearchTool(),
    createGraphSearchTool(),
  ];
  for (const t of tools) registry.register(t);
};