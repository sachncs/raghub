/**
 * Sub-agent adapters — wrap the built-in tools so they can be
 * consumed by `RagAgent.fanOut(...)`.
 *
 * Each `SubAgent` is responsible for one retrieval role (vector,
 * keyword, graph, trace, web, memory). They take their backend
 * collaborators at construction time and re-use the active user's
 * RBAC/ACL via the `invocation_state` argument.
 *
 * `buildDefaultSubAgents(deps)` produces a tuple of sub-agents that
 * covers the default role set the RagAgent requests.
 */

import {
  type Embedder,
  type Hit,
  type Retrieval,
  type SqliteGraphStore,
  type SqliteTraceCorpus,
  type SummaryIndex,
  type VectorStore,
  type WebSearch,
  allowedCompanyFilter,
} from '@raghub/core';

import type { SubAgent, SubAgentInput } from '../agents/rag-agent.js';
import type { InvocationState } from '../strands/types.js';

const userFromState = (
  state: InvocationState,
): { id: string; workspaceId: string; allowedCompanies: readonly string[] } => {
  return {
    id: String(state.user_id ?? ''),
    workspaceId: String(state.workspace_id ?? ''),
    allowedCompanies: state.rbac_filter.allowedCompanies,
  };
};

const keywordToHits = (k: { chunkId: string; score: number; text: string }, vec?: VectorStore): Hit | null => {
  if (!vec) {
    return {
      chunk: {
        id: k.chunkId as never,
        workspaceId: '' as never,
        ownerId: '' as never,
        collectionId: '' as never,
        documentId: '' as never,
        modality: 'text',
        text: k.text,
        embedding: [],
        metadata: {},
        tokenCount: k.text.split(/\s+/).length,
        createdAt: new Date(),
      } as never,
      score: k.score,
    };
  }
  return null;
};

void keywordToHits;

export interface SubAgentDeps {
  readonly retrieval: Retrieval;
  readonly embedder: Embedder;
  readonly vectorStore: VectorStore;
  readonly webSearch?: WebSearch;
  readonly graphStore?: SqliteGraphStore;
  readonly summaryIndex?: SummaryIndex;
  readonly traceCorpus?: SqliteTraceCorpus;
}

export const buildVectorSubAgent = (deps: Pick<SubAgentDeps, 'embedder' | 'vectorStore'>): SubAgent => ({
  role: 'vector',
  async retrieve(input: SubAgentInput, state: InvocationState) {
    const user = userFromState(state);
    const topK = state.strategy.k;
    const filter = {
      workspaceId: state.workspace_id,
      userId: state.user_id,
      collectionId: null,
      principals: [{ type: 'user' as const, id: user.id }],
      allowedCompanies: user.allowedCompanies,
    };
    const vec = await deps.embedder.embedQuery(input.query);
    return deps.vectorStore.searchVector({ vector: vec, topK, filter });
  },
});

export const buildKeywordSubAgent = (deps: Pick<SubAgentDeps, 'vectorStore'>): SubAgent => ({
  role: 'keyword',
  async retrieve(input: SubAgentInput, state: InvocationState) {
    const user = userFromState(state);
    const topK = state.strategy.k;
    const filter = {
      workspaceId: state.workspace_id,
      userId: state.user_id,
      collectionId: null,
      principals: [{ type: 'user' as const, id: user.id }],
      allowedCompanies: user.allowedCompanies,
    };
    const keywordHits = await deps.vectorStore.searchKeyword({ query: input.query, topK, filter });
    return keywordHits.map((k) => ({
      chunk: {
        id: k.chunkId,
        workspaceId: state.workspace_id,
        ownerId: state.user_id ?? ('' as never),
        collectionId: ('' as never) as never,
        documentId: ('' as never) as never,
        modality: 'text' as const,
        text: k.text,
        embedding: [],
        metadata: {},
        tokenCount: k.text.split(/\s+/).length,
        createdAt: new Date(),
      } as never,
      score: k.score,
    }));
  },
});

export const buildGraphSubAgent = (deps: Pick<SubAgentDeps, 'graphStore'>): SubAgent | null => {
  const store = deps.graphStore;
  if (!store) return null;
  return {
    role: 'graph',
    async retrieve(input: SubAgentInput, state: InvocationState) {
      const entities = await store.searchEntities(state.workspace_id, input.query, state.strategy.k);
      const neighbors = entities.length > 0
        ? await store.expandNeighborhood(
            state.workspace_id,
            entities.map((e) => e.name),
            1,
            state.strategy.k,
          )
        : [];
      const out: Hit[] = [];
      for (const e of entities) {
        out.push({
          chunk: {
            id: (`graph_${e.name}` as never) as never,
            workspaceId: state.workspace_id,
            ownerId: state.user_id ?? ('' as never),
            collectionId: ('' as never) as never,
            documentId: e.name as never,
            modality: 'text' as const,
            text: `entity: ${e.name} (chunks=${e.chunkCount})`,
            embedding: [],
            metadata: { source: 'graph', entity: e.name },
            tokenCount: 0,
            createdAt: new Date(),
          } as never,
          score: 1,
        });
      }
      for (const n of neighbors) {
        out.push({
          chunk: {
            id: (`graph_n_${n.name}` as never) as never,
            workspaceId: state.workspace_id,
            ownerId: state.user_id ?? ('' as never),
            collectionId: ('' as never) as never,
            documentId: n.name as never,
            modality: 'text' as const,
            text: `related: ${n.name} (chunks=${n.chunkCount})`,
            embedding: [],
            metadata: { source: 'graph', neighbor: n.name },
            tokenCount: 0,
            createdAt: new Date(),
          } as never,
          score: 0.5,
        });
      }
      return out;
    },
  };
};

export const buildTraceSubAgent = (deps: Pick<SubAgentDeps, 'traceCorpus' | 'embedder'>): SubAgent | null => {
  const corpus = deps.traceCorpus;
  const embedder = deps.embedder;
  if (!corpus || !embedder) return null;
  return {
    role: 'trace',
    async retrieve(input: SubAgentInput, state: InvocationState) {
      const vector = await embedder.embedQuery(input.query);
      const hits = await corpus.search({
        workspaceId: state.workspace_id,
        vector,
        representation: 'semantic',
        topK: state.strategy.k,
      });
      return hits.map((t, i) => ({
        chunk: {
          id: (`trace_${t.id}` as never) as never,
          workspaceId: state.workspace_id,
          ownerId: state.user_id ?? ('' as never),
          collectionId: ('' as never) as never,
          documentId: t.id as never,
          modality: 'text' as const,
          text: t.text,
          embedding: [],
          metadata: { source: 'trace' },
          tokenCount: t.text.split(/\s+/).length,
          createdAt: new Date(),
        } as never,
        score: 1 - i * 0.01,
      }));
    },
  };
};

export const buildWebSubAgent = (deps: Pick<SubAgentDeps, 'webSearch'>): SubAgent | null => {
  const search = deps.webSearch;
  if (!search) return null;
  return {
    role: 'web',
    async retrieve(input: SubAgentInput, state: InvocationState) {
      const result = await search.search({
        query: input.query,
        maxResults: state.strategy.k,
      });
      return result.hits.map((r, i) => ({
        chunk: {
          id: (`web_${i}_${r.url}` as never) as never,
          workspaceId: state.workspace_id,
          ownerId: state.user_id ?? ('' as never),
          collectionId: ('' as never) as never,
          documentId: r.url as never,
          modality: 'text' as const,
          text: r.snippet,
          embedding: [],
          metadata: { url: r.url, title: r.title, source: 'web' },
          tokenCount: r.snippet.split(/\s+/).length,
          createdAt: new Date(),
        } as never,
        score: 1 - i * 0.05,
      }));
    },
  };
};

export const buildMemorySubAgent = (_deps: SubAgentDeps): SubAgent | null => {
  // Memory sub-agent wires in once WorkspaceMemoryStore (C-13) lands.
  void _deps;
  return null;
};

export const buildSummarySubAgent = (_deps: Pick<SubAgentDeps, 'summaryIndex'>): SubAgent | null => {
  // SummaryIndex is a write-only summarizer; we treat it as a no-op
  // retrieval source for now and only expose it via the ingestion
  // pipeline. Returns null so the registry skips it.
  void _deps;
  return null;
};

export const buildDefaultSubAgents = (deps: SubAgentDeps): readonly SubAgent[] => {
  const list: SubAgent[] = [];
  const v = buildVectorSubAgent(deps);
  if (v) list.push(v);
  const k = buildKeywordSubAgent(deps);
  if (k) list.push(k);
  const g = buildGraphSubAgent(deps);
  if (g) list.push(g);
  const t = buildTraceSubAgent(deps);
  if (t) list.push(t);
  const w = buildWebSubAgent(deps);
  if (w) list.push(w);
  return list;
};

void allowedCompanyFilter;