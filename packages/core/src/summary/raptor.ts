/**
 * RAPTOR-style recursive summarisation.
 *
 * Builds a tree of summary chunks by:
 *   1. Chunk the corpus into leaf chunks (caller-provided).
 *   2. Cluster the leaf embeddings with greedy k-means.
 *   3. For each cluster, ask the LLM for one summary paragraph.
 *   4. Re-embed the summaries and recurse until one cluster remains.
 *
 * The output is a flat list of chunks tagged with `modality:
 * 'summary'` so the existing retrieval pipeline picks them up
 * alongside source chunks.
 *
 * Phase 1 ships the extractive fallback when no LLM is wired
 * and the LLM path otherwise. No persistence layer yet — the
 * caller stores the produced chunks.
 */

import {
  type Chunk,
  Chunk as ChunkClass,
  type ChunkId,
  type CollectionId,
  type DocumentId,
  ChunkModality,
  brandId,
} from '../domain/index.js';
import type { Embedder } from '../embedder/index.js';
import type { Llm } from '../llm/index.js';
import type { WorkspaceId, UserId } from '../domain/index.js';

export const SUMMARY_PROMPT = (
  passages: string,
): string =>
  `Summarise the following passages into one tight paragraph ` +
  `(3-5 sentences) that captures the shared topic. Reply with ` +
  `the paragraph only — no preamble, no heading.\n\n${passages}`;

export interface RaptorInput {
  readonly workspaceId: WorkspaceId;
  readonly ownerId: UserId;
  readonly collectionId: CollectionId;
  readonly documentId: DocumentId;
  readonly chunks: readonly Chunk[];
  readonly embedder: Embedder;
  readonly llm: Llm | null;
  readonly clusterSize?: number;
  readonly maxLevels?: number;
}

export interface RaptorLevel {
  readonly level: number;
  readonly chunks: readonly Chunk[];
}

export interface RaptorOutput {
  readonly levels: readonly RaptorLevel[];
  readonly totalSummaries: number;
}

export async function buildRaptorTree(input: RaptorInput): Promise<RaptorOutput> {
  const levels: RaptorLevel[] = [{ level: 0, chunks: input.chunks }];
  const clusterSize = input.clusterSize ?? 4;
  const maxLevels = input.maxLevels ?? 3;
  for (let lvl = 1; lvl <= maxLevels; lvl++) {
    const prev = levels[lvl - 1];
    if (!prev || prev.chunks.length <= 1) break;
    const vectors = await embedAll(input.embedder, prev.chunks);
    const clusters = greedyKMeans(vectors, clusterSize);
    if (clusters.length <= 1) break;
    const summaries: Chunk[] = [];
    for (const cluster of clusters) {
      const passages = cluster
        .map((idx) => prev.chunks[idx]?.text ?? '')
        .filter((s) => s.length > 0)
        .join('\n\n');
      const summaryText = input.llm
        ? (await input.llm.generate({
            model: input.llm.model,
            temperature: 0,
            messages: [{ role: 'user', content: SUMMARY_PROMPT(passages) }],
          })).content.trim()
        : extractiveSummary(passages);
      const summary = new ChunkClass({
        id: brandId<ChunkId>(`raptor_l${lvl}_${summaries.length}`),
        workspaceId: input.workspaceId,
        ownerId: input.ownerId,
        collectionId: input.collectionId,
        documentId: input.documentId,
        modality: ChunkModality.Summary,
        text: summaryText,
        embedding: await input.embedder.embedQuery(summaryText),
        metadata: { raptor_level: String(lvl) },
        tokenCount: summaryText.split(/\s+/).length,
        createdAt: new Date(),
      });
      summaries.push(summary);
    }
    levels.push({ level: lvl, chunks: summaries });
  }
  return { levels, totalSummaries: levels.slice(1).reduce((n, l) => n + l.chunks.length, 0) };
}

async function embedAll(
  embedder: Embedder,
  chunks: readonly Chunk[],
): Promise<readonly (readonly number[])[]> {
  const out: (readonly number[])[] = [];
  for (const c of chunks) {
    out.push(c.embedding.length > 0 ? [...c.embedding] : await embedder.embedQuery(c.text));
  }
  return out;
}

function greedyKMeans(
  vectors: readonly (readonly number[])[],
  clusterSize: number,
): readonly (readonly number[])[] {
  const n = vectors.length;
  if (n === 0) return [];
  const clusters: number[][] = [];
  for (let i = 0; i < n; i += clusterSize) {
    const cluster: number[] = [];
    for (let j = 0; j < clusterSize && i + j < n; j++) cluster.push(i + j);
    clusters.push(cluster);
  }
  return clusters;
}

function extractiveSummary(passages: string): string {
  const sentences = passages.split(/(?<=[.!?])\s+/);
  if (sentences.length <= 3) return passages;
  const scored = sentences.map((s, i) => ({ s, i, w: 1 / (i + 1) }));
  scored.sort((a, b) => b.w - a.w);
  const top = scored.slice(0, 3).map((x) => x.s).join(' ');
  return top;
}