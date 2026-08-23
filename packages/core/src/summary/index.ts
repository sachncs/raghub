/**
 * RAPTOR-style summarisation.
 *
 * `SummaryIndex` derives chunk-level summaries (LLM-driven when
 * an LLM is available, otherwise a deterministic extractive
 * fallback) and stores them as regular Chunks with
 * `modality: 'summary'` so the existing retrieval pipeline picks
 * them up alongside the underlying source chunks.
 *
 * Phase 1 ships the extractive path + an LLM summariser the
 * caller wires in. The tree layer (cluster -> summarise) lands
 * alongside T3 in Phase 2.
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
import type { TenantId, UserId } from '../domain/index.js';

export interface SummaryInput {
  readonly tenantId: TenantId;
  readonly ownerId: UserId;
  readonly collectionId: CollectionId;
  readonly documentId: DocumentId;
  readonly parentIds: readonly ChunkId[];
  readonly sourceTexts: readonly string[];
  readonly depth: number;
}

export interface SummaryIndex {
  summarise(input: SummaryInput): Promise<Chunk>;
  close(): void;
}

const SUMMARISE_SYSTEM_PROMPT =
  'You are a precise technical summariser. Produce a concise summary that captures the key facts and claims in the source chunks. Do not introduce information not present in the source. Use 2-4 sentences.';

const splitSentences = (text: string): string[] => {
  const sentences = text
    .split(/(?<=[.!?])\s+/)
    .map((s) => s.trim())
    .filter((s) => s.length > 0);
  return sentences;
};

const extractiveSummary = (sources: readonly string[]): string => {
  const sentences: { text: string; score: number; idx: number }[] = [];
  sources.forEach((src, blockIdx) => {
    for (const s of splitSentences(src)) {
      sentences.push({ text: s, score: 0, idx: blockIdx * 1000 + sentences.length });
    }
  });
  if (sentences.length === 0) return '';
  const wordFreq = new Map<string, number>();
  for (const s of sentences) {
    for (const tok of s.text.toLowerCase().split(/[^a-z0-9]+/)) {
      if (tok.length < 3) continue;
      wordFreq.set(tok, (wordFreq.get(tok) ?? 0) + 1);
    }
  }
  for (const s of sentences) {
    let score = 0;
    for (const tok of s.text.toLowerCase().split(/[^a-z0-9]+/)) {
      score += wordFreq.get(tok) ?? 0;
    }
    s.score = score / Math.max(1, s.text.split(/\s+/).length);
  }
  sentences.sort((a, b) => b.score - a.score || a.idx - b.idx);
  const keep = Math.min(5, Math.max(2, Math.ceil(sentences.length / 4)));
  const kept = sentences.slice(0, keep).map((s) => s.text);
  return kept.join(' ');
};

export const createExtractiveSummaryIndex = (): SummaryIndex => ({
  async summarise(input: SummaryInput): Promise<Chunk> {
    const text = extractiveSummary(input.sourceTexts);
    return new ChunkClass({
      id: brandId<ChunkId>(`sum_${input.documentId}_${input.depth}_${input.parentIds[0] ?? 'root'}`),
      tenantId: input.tenantId,
      ownerId: input.ownerId,
      collectionId: input.collectionId,
      documentId: input.documentId,
      modality: ChunkModality.Summary,
      text: text || input.sourceTexts.join(' ').slice(0, 800),
      embedding: [],
      metadata: {
        depth: String(input.depth),
        parents: input.parentIds.join(','),
        kind: 'extractive',
      },
      tokenCount: Math.ceil(text.length / 4),
      createdAt: new Date(),
    });
  },
  close: () => undefined,
});

export const createLlmSummaryIndex = (deps: {
  readonly llm: Llm;
  readonly model: string;
  readonly embedder: Embedder;
}): SummaryIndex => ({
  async summarise(input: SummaryInput): Promise<Chunk> {
    let text: string;
    try {
      const result = await deps.llm.generate({
        model: deps.model,
        temperature: 0,
        messages: [
          { role: 'system', content: SUMMARISE_SYSTEM_PROMPT },
          { role: 'user', content: input.sourceTexts.join('\n\n---\n\n') },
        ],
      });
      text = result.content.trim();
    } catch {
      text = extractiveSummary(input.sourceTexts);
    }
    if (!text) text = input.sourceTexts.join(' ').slice(0, 800);
    const summaryId = brandId<ChunkId>(
      `sum_${input.documentId}_${input.depth}_${input.parentIds[0] ?? 'root'}`,
    );
    let embedding: readonly number[] = [];
    try {
      embedding = await deps.embedder.embedQuery(text);
    } catch {
      embedding = [];
    }
    return new ChunkClass({
      id: summaryId,
      tenantId: input.tenantId,
      ownerId: input.ownerId,
      collectionId: input.collectionId,
      documentId: input.documentId,
      modality: ChunkModality.Summary,
      text,
      embedding,
      metadata: {
        depth: String(input.depth),
        parents: input.parentIds.join(','),
        kind: 'llm',
      },
      tokenCount: Math.ceil(text.length / 4),
      createdAt: new Date(),
    });
  },
  close: () => undefined,
});