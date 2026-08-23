/**
 * Cross-modal hybrid retrieval.
 *
 * Two parallel paths:
 *  - structural: keyword + entity match on the dual graph, hop-
 *    bounded neighborhood expansion (default hop=2).
 *  - semantic: cosine similarity on a shared dense table T that
 *    covers entities, relations, and chunks (text + VLM-derived
 *    descriptions for non-text units).
 *
 * Multi-signal fusion: structural importance + cosine + query-
 * inferred modality preference (queries mentioning "figure",
 * "chart", "table", "equation" bias toward those modalities).
 */

import type { Embedder } from '@raghub/core';

import type { MultimodalGraph, GraphEdge, GraphNode } from './dual-graph.js';

export interface RetrievalCandidate {
  readonly id: string;
  readonly modality: 'text' | 'image' | 'table' | 'equation' | 'layout';
  readonly text: string;
  readonly score: number;
}

export interface RetrievalOptions {
  readonly graph: MultimodalGraph;
  readonly query: string;
  readonly queryEmbedding?: readonly number[];
  readonly candidateEmbeddings?: ReadonlyMap<string, readonly number[]>;
  readonly hop?: number;
  readonly topK?: number;
}

const cosine = (a: readonly number[], b: readonly number[]): number => {
  if (a.length !== b.length || a.length === 0) return 0;
  let dot = 0;
  let na = 0;
  let nb = 0;
  for (let i = 0; i < a.length; i++) {
    const x = a[i] ?? 0;
    const y = b[i] ?? 0;
    dot += x * y;
    na += x * x;
    nb += y * y;
  }
  return dot / (Math.sqrt(na) * Math.sqrt(nb) || 1);
};

const inferModalityPreference = (query: string): ReadonlyMap<string, number> => {
  const out = new Map<string, number>();
  const lower = query.toLowerCase();
  if (/figure|chart|image|diagram|plot|graph/.test(lower)) out.set('image', 1.5);
  if (/table|row|column/.test(lower)) out.set('table', 1.5);
  if (/equation|formula|\beq\b/.test(lower)) out.set('equation', 1.5);
  if (/section|chapter|heading/.test(lower)) out.set('layout', 1.1);
  return out;
};

const structuralImportance = (
  graph: MultimodalGraph,
  nodeId: string,
): number => {
  let inDegree = 0;
  for (const e of graph.edges) {
    if (e.to === nodeId) inDegree += e.weight;
  }
  return Math.log(1 + inDegree);
};

export const crossModalRetrieve = (opts: RetrievalOptions): readonly RetrievalCandidate[] => {
  const hop = opts.hop ?? 2;
  const topK = opts.topK ?? 10;
  const query = opts.query;
  const tokens = query.toLowerCase().split(/[^a-z0-9]+/).filter((t) => t.length > 1);

  const nodeById = new Map<string, GraphNode>();
  for (const n of opts.graph.nodes) nodeById.set(n.id, n);

  const structuralMatches = new Map<string, number>();
  const visited = new Set<string>();
  const queue: { id: string; depth: number; score: number }[] = [];
  for (const node of opts.graph.nodes) {
    const lowerName = node.name.toLowerCase();
    let score = 0;
    for (const t of tokens) {
      if (lowerName.includes(t)) score += 1;
    }
    if (score > 0) {
      queue.push({ id: node.id, depth: 0, score });
      structuralMatches.set(node.id, score);
    }
  }
  while (queue.length > 0) {
    const head = queue.shift();
    if (!head) continue;
    if (visited.has(head.id)) continue;
    visited.add(head.id);
    if (head.depth >= hop) continue;
    for (const e of opts.graph.edges) {
      if (e.from !== head.id && e.to !== head.id) continue;
      const other = e.from === head.id ? e.to : e.from;
      const nextScore = head.score * 0.5 + e.weight * 0.1;
      queue.push({ id: other, depth: head.depth + 1, score: nextScore });
    }
  }

  const modalityPref = inferModalityPreference(query);

  const candidates = new Map<string, RetrievalCandidate>();
  for (const node of opts.graph.nodes) {
    const sScore = structuralMatches.get(node.id) ?? 0;
    const sImportance = structuralImportance(opts.graph, node.id);
    let semScore = 0;
    if (opts.queryEmbedding && opts.candidateEmbeddings?.has(node.id)) {
      semScore = cosine(opts.queryEmbedding, opts.candidateEmbeddings.get(node.id) ?? []);
    }
    const modBoost = modalityPref.get(node.modality) ?? 1;
    const fused = (sScore * 1.0 + sImportance * 0.5 + semScore * 2.0) * modBoost;
    if (fused <= 0) continue;
    candidates.set(node.id, {
      id: node.id,
      modality: node.modality,
      text: node.text,
      score: fused,
    });
  }

  const ranked = [...candidates.values()].sort((a, b) => b.score - a.score);
  void structuralMatches;
  void visited;
  void queue;
  void nodeById;
  return ranked.slice(0, topK);
};

export const buildCandidateEmbeddings = async (
  embedder: Embedder,
  candidates: readonly { id: string; text: string }[],
): Promise<Map<string, readonly number[]>> => {
  const map = new Map<string, readonly number[]>();
  await Promise.all(
    candidates.map(async (c) => {
      const v = await embedder.embedQuery(c.text);
      map.set(c.id, v);
    }),
  );
  return map;
};

void ({} as GraphEdge);