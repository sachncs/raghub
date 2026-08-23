/**
 * Built-in tools — the eight that mirror the legacy Python tool
 * surface. Every tool reads the active `invocation_state` to
 * honour RBAC + per-user strategy and wraps failures as
 * `{ ok: false, error }` rather than throwing.
 *
 * The storage-backed tools (`graph_search`, `summary_search`,
 * `trace_search`, `web_search`) take their backend collaborators
 * as a single `toolDeps` bag; the registry wires them up via
 * `registerBuiltInTools(registry, deps)`.
 */

import {
  allowedCompanyFilter,
  type Embedder,
  type Hit,
  type Retrieval,
  type SqliteGraphStore,
  type SqliteTraceCorpus,
  type SummaryIndex,
  type VectorStore,
  type WebSearch,
  RaghubError,
  User,
} from '@raghub/core';

import type { Tool, ToolContext, ToolResult } from './registry.js';

const okResult = (content: string, data?: Record<string, unknown>): ToolResult => {
  const r: {
    ok: true;
    content: string;
    latencyMs: number;
    data?: Readonly<Record<string, unknown>>;
  } = { ok: true, content, latencyMs: 0 };
  if (data !== undefined) r.data = data;
  return r;
};

const errResult = (error: string, start: number): ToolResult => ({
  ok: false,
  content: '',
  error,
  latencyMs: Date.now() - start,
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

const requireUser = (
  state: {
    user_id: unknown;
    tenant_id: unknown;
    is_admin: boolean;
    rbac_filter: { allowedCompanies: readonly string[] };
  },
): User => {
  if (!state.user_id || !state.tenant_id) {
    throw new RaghubError('authorization_error', 'tool requires an authenticated user');
  }
  return new User({
    id: state.user_id as never,
    tenantId: state.tenant_id as never,
    email: '',
    role: state.is_admin ? 'admin' : 'member',
    allowedCompanies: state.rbac_filter.allowedCompanies,
    createdAt: new Date(),
  });
};

export interface ToolDeps {
  readonly retrieval: Retrieval;
  readonly embedder: Embedder;
  readonly store: VectorStore;
  readonly webSearch?: WebSearch;
  readonly graphStore?: SqliteGraphStore;
  readonly summaryIndex?: SummaryIndex;
  readonly traceCorpus?: SqliteTraceCorpus;
}

export const createHybridSearchTool = (retrieval: Retrieval): Tool => ({
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
      return okResult(
        JSON.stringify(hits.map((h: Hit) => ({ id: h.chunk.id, score: h.score, text: h.chunk.text }))),
        { hits },
      );
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
      return okResult(
        JSON.stringify(hits.map((h: Hit) => ({ id: h.chunk.id, score: h.score }))),
        { hits },
      );
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

export const createWebSearchTool = (search: WebSearch): Tool => ({
  name: 'web_search',
  description: 'Web search via the configured provider (DuckDuckGo by default).',
  jsonSchema: {
    type: 'object',
    properties: { query: { type: 'string' }, max_results: { type: 'number' } },
    required: ['query'],
  },
  async execute(args, ctx) {
    return wrap(Date.now(), async () => {
      const q = String(args['query'] ?? '');
      const max = Number(args['max_results'] ?? 5);
      const result = await search.search({
        query: q,
        maxResults: max,
        ...(ctx.signal ? { signal: ctx.signal } : {}),
      });
      return okResult(JSON.stringify(result.hits), { took: result.took, hits: result.hits });
    });
  },
});

export const createTraceSearchTool = (
  corpus: SqliteTraceCorpus,
  embedder: Embedder,
): Tool => ({
  name: 'trace_search',
  description:
    'Retrieve transformed thinking traces from the active tenant\'s trace corpus.',
  jsonSchema: {
    type: 'object',
    properties: {
      question: { type: 'string' },
      representation: { type: 'string' },
      top_k: { type: 'number' },
    },
    required: ['question'],
  },
  async execute(args, ctx) {
    return wrap(Date.now(), async () => {
      const user = requireUser(ctx.invocationState);
      const q = String(args['question'] ?? '');
      const repRaw = String(
        args['representation'] ?? ctx.invocationState.strategy.traceCorpus.representation,
      );
      const representation: 'struct' | 'semantic' | 'reflect' =
        repRaw === 'struct' || repRaw === 'reflect' ? repRaw : 'semantic';
      const topK = Number(args['top_k'] ?? ctx.invocationState.strategy.traceCorpus.topK);
      const vec = await embedder.embedQuery(q);
      const hits = await corpus.search({
        tenantId: user.tenantId,
        vector: vec,
        representation,
        topK,
      });
      return okResult(JSON.stringify(hits), { hits });
    });
  },
});

export const createSummarySearchTool = (
  store: VectorStore,
  embedder: Embedder,
): Tool => ({
  name: 'summary_search',
  description: 'Search the RAPTOR-style summary index (summaries stored as chunks with modality=summary).',
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
      const summaryHits = hits.filter((h) => h.chunk.modality === 'summary');
      return okResult(
        JSON.stringify(
          summaryHits.map((h: Hit) => ({
            id: h.chunk.id,
            score: h.score,
            depth: h.chunk.metadata['depth'] ?? '',
          })),
        ),
        { hits: summaryHits },
      );
    });
  },
});

export const createGraphSearchTool = (graph: SqliteGraphStore): Tool => ({
  name: 'graph_search',
  description: 'GraphRAG entity search with hop-bounded neighborhood expansion.',
  jsonSchema: {
    type: 'object',
    properties: { question: { type: 'string' }, hop: { type: 'number' } },
    required: ['question'],
  },
  async execute(args, ctx) {
    return wrap(Date.now(), async () => {
      const user = requireUser(ctx.invocationState);
      const q = String(args['question'] ?? '');
      const hop = Math.min(3, Math.max(1, Number(args['hop'] ?? 2)));
      const seeds = await graph.searchEntities(user.tenantId, q, 10);
      const expanded = await graph.expandNeighborhood(
        user.tenantId,
        seeds.map((s) => s.name),
        hop,
        20,
      );
      return okResult(JSON.stringify({ seeds, expanded }), { seeds, expanded });
    });
  },
});

export const registerBuiltInTools = (
  registry: { register: (tool: Tool) => void },
  deps: ToolDeps,
): void => {
  const tools: Tool[] = [
    createHybridSearchTool(deps.retrieval),
    createVectorSearchTool(deps.embedder, deps.store),
    createKeywordSearchTool(deps.store),
    createTodayTool(),
  ];
  if (deps.webSearch) tools.push(createWebSearchTool(deps.webSearch));
  if (deps.traceCorpus && deps.embedder) tools.push(createTraceSearchTool(deps.traceCorpus, deps.embedder));
  if (deps.store && deps.embedder) tools.push(createSummarySearchTool(deps.store, deps.embedder));
  if (deps.graphStore) tools.push(createGraphSearchTool(deps.graphStore));

  for (const t of tools) registry.register(t);
};