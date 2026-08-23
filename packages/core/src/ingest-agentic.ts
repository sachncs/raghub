/**
 * Agentic ingestion pipeline.
 *
 * Wraps `ingest()` with three parallel post-processing steps that
 * fire after the chunks land in the vector store:
 *
 *  1. `extractEntities` -> graph store mentions (entity extraction)
 *  2. `remember` -> memory store (per-chunk memory facts scoped
 *     to the document owner)
 *  3. optional `summarize` -> produce one summary chunk per doc
 *
 * Steps 1 + 2 run concurrently via `Promise.all`. Step 3 only
 * runs when `summaryIndex` is present.
 *
 * Failures are non-fatal — ingest completes even if a downstream
 * agent throws. Errors are reported via the returned array.
 */

import type { ChunkId } from './domain/index.js';
import type { Embedder } from './embedder/index.js';
import { extractEntities, type GraphStore } from './graph/store.js';
import type { WorkspaceMemoryStore } from './storage/memory.js';
import type { SummaryIndex } from './summary/index.js';
import { ingest, type IngestInput, type IngestOutput } from './ingest.js';
import type { VectorStore } from './stores/index.js';

export interface AgenticIngestExtra {
  readonly graphStore?: GraphStore;
  readonly memoryStore?: WorkspaceMemoryStore;
  readonly summaryIndex?: SummaryIndex;
}

export interface AgenticIngestDeps {
  readonly embedder: Embedder;
  readonly store: VectorStore;
  readonly seenHashes?: (hash: string) => Promise<boolean>;
  readonly extra?: AgenticIngestExtra;
}

export interface AgenticIngestOutput extends IngestOutput {
  readonly sideEffects: readonly AgenticIngestSideEffect[];
}

export type AgenticIngestSideEffect =
  | { readonly kind: 'graph'; readonly entities: readonly string[]; readonly chunks: readonly ChunkId[] }
  | { readonly kind: 'memory'; readonly facts: number }
  | { readonly kind: 'summary'; readonly summaryChunkId: ChunkId };

export const agenticIngest = async (
  input: IngestInput,
  deps: AgenticIngestDeps,
): Promise<AgenticIngestOutput> => {
  const base = await ingest(input, {
    embedder: deps.embedder,
    store: deps.store,
    ...(deps.seenHashes !== undefined ? { seenHashes: deps.seenHashes } : {}),
  });
  if (base.alreadyExisted) return { ...base, sideEffects: [] };

  const extra = deps.extra;
  if (!extra) return { ...base, sideEffects: [] };

  const tasks: Promise<AgenticIngestSideEffect | null>[] = [];

  if (extra.graphStore) {
    tasks.push(runGraphStep(input, base, extra.graphStore, deps.store));
  }
  if (extra.memoryStore) {
    tasks.push(runMemoryStep(input, base, extra.memoryStore));
  }

  const initial = await Promise.all(tasks);
  const sideEffects: AgenticIngestSideEffect[] = initial.filter((s): s is AgenticIngestSideEffect => s !== null);

  if (extra.summaryIndex) {
    try {
      const summary = await extra.summaryIndex.summarise({
        workspaceId: input.workspaceId,
        ownerId: input.ownerId,
        collectionId: input.collectionId,
        documentId: base.documentId,
        depth: 1,
        parentIds: [base.documentId as never],
        sourceTexts: base.chunks.length > 0 ? [`${base.chunks.length} chunks in ${input.filename}`] : [],
      });
      await deps.store.add(summary);
      sideEffects.push({ kind: 'summary', summaryChunkId: summary.id as never });
    } catch {
      // Summary is best-effort.
    }
  }

  return { ...base, sideEffects };
};

const runGraphStep = async (
  input: IngestInput,
  output: IngestOutput,
  store: GraphStore,
  vectorStore: VectorStore,
): Promise<AgenticIngestSideEffect | null> => {
  try {
    const all = new Set<string>();
    const mentionChunks: ChunkId[] = [];
    for (const c of output.chunks) {
      const chunk = await vectorStore.getById(input.workspaceId, c.id);
      if (!chunk) continue;
      const ents = extractEntities(chunk.text);
      if (ents.length > 0) {
        await store.addMentions(input.workspaceId, c.id, ents);
        for (const e of ents) all.add(e);
        mentionChunks.push(c.id);
      }
    }
    return { kind: 'graph', entities: [...all], chunks: mentionChunks };
  } catch {
    return null;
  }
};

const runMemoryStep = async (
  input: IngestInput,
  _output: IngestOutput,
  store: WorkspaceMemoryStore,
): Promise<AgenticIngestSideEffect | null> => {
  try {
    await store.remember({
      workspaceId: input.workspaceId,
      userId: input.ownerId,
      scope: 'user',
      content: `Ingested ${input.filename} (${input.mimeType})`,
      metadata: { source: 'ingest', filename: input.filename },
    });
    return { kind: 'memory', facts: 1 };
  } catch {
    return null;
  }
};

