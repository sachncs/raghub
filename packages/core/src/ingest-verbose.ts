/**
 * Verbose ingestion — same pipeline as `ingest()`, but emits
 * lifecycle events as it goes so callers can stream progress back
 * to the user (SSE, polling endpoint, log lines).
 *
 * Event timeline for a normal run:
 *
 *   { phase: 'start',     filename, byteSize, mimeType, hash, documentId }
 *   { phase: 'parsed',    kind, characterCount, parserLatencyMs, metadata }
 *   { phase: 'chunked',   chunkCount, totalTokens }
 *   { phase: 'embedding', processed, total, batchIndex, batchSize }
 *   { phase: 'persisting', chunkCount }
 *   { phase: 'indexed',   chunkCount, totalTokens, totalLatencyMs }
 *   { phase: 'skipped',   reason, documentId }
 *   { phase: 'failed',    error, documentId }
 *
 * Listeners receive every event in order. Errors in listeners are
 * swallowed so they don't break the ingest pipeline.
 */

import type { Embedder } from './embedder/index.js';
import type { VectorStore } from './stores/index.js';
import type { IngestInput, IngestDeps, IngestOutput } from './ingest.js';

export type IngestEvent =
  | {
      readonly phase: 'start';
      readonly filename: string;
      readonly byteSize: number;
      readonly mimeType: string;
      readonly hash: string;
      readonly documentId: string;
    }
  | {
      readonly phase: 'parsed';
      readonly kind: 'pdf' | 'html' | 'markdown' | 'text';
      readonly characterCount: number;
      readonly parserLatencyMs: number;
      readonly metadata: Readonly<Record<string, string>>;
    }
  | {
      readonly phase: 'chunked';
      readonly chunkCount: number;
      readonly totalTokens: number;
      readonly chunkerLatencyMs: number;
    }
  | {
      readonly phase: 'embedding';
      readonly processed: number;
      readonly total: number;
      readonly batchIndex: number;
      readonly batchSize: number;
    }
  | {
      readonly phase: 'persisting';
      readonly chunkCount: number;
    }
  | {
      readonly phase: 'indexed';
      readonly chunkCount: number;
      readonly totalTokens: number;
      readonly totalLatencyMs: number;
    }
  | {
      readonly phase: 'skipped';
      readonly reason: 'duplicate';
      readonly documentId: string;
      readonly hash: string;
    }
  | {
      readonly phase: 'failed';
      readonly error: string;
      readonly documentId: string;
    };

export type IngestListener = (event: IngestEvent) => void;

export class IngestEmitter {
  private readonly listeners = new Set<IngestListener>();

  public on(listener: IngestListener): () => void {
    this.listeners.add(listener);
    return () => this.listeners.delete(listener);
  }

  public emit(event: IngestEvent): void {
    for (const listener of this.listeners) {
      try {
        listener(event);
      } catch {
        /* listener errors are swallowed */
      }
    }
  }

  public listenerCount(): number {
    return this.listeners.size;
  }
}

export interface VerboseIngestOptions {
  readonly emitter: IngestEmitter;
  readonly batchSize?: number;
}

export const ingestVerbose = async (
  input: IngestInput,
  deps: IngestDeps,
  options: VerboseIngestOptions,
): Promise<IngestOutput> => {
  const start = performance.now();
  const emit = options.emitter.emit.bind(options.emitter);

  const { hashDocument, ingest: silentIngest } = await import('./ingest.js');
  const { chunkText, chunkPdf, chunkStructured, chunkMarkdown } = await import('./chunker/index.js');

  const hash = hashDocument(input.content);
  const documentId = `doc_${hash.slice(0, 16)}`;
  emit({
    phase: 'start',
    filename: input.filename,
    byteSize: input.content.byteLength,
    mimeType: input.mimeType,
    hash,
    documentId,
  });

  if (deps.seenHashes) {
    const seen = await deps.seenHashes(hash);
    if (seen) {
      emit({ phase: 'skipped', reason: 'duplicate', documentId, hash });
      return { documentId: documentId as never, hash, chunks: [], alreadyExisted: true };
    }
  }

  const parseStart = performance.now();
  const m = input.mimeType.toLowerCase();
  let kind: 'pdf' | 'html' | 'markdown' | 'text';
  let rawText = '';
  let extraMetadata: Record<string, string> = {};
  if (m === 'application/pdf' || m.startsWith('application/pdf')) {
    const result = await chunkPdf(input.content);
    rawText = result.chunks.map((c) => c.text).join('\n');
    extraMetadata = { pages: String(result.pages) };
    kind = 'pdf';
  } else if (m === 'text/html' || m === 'application/xhtml+xml') {
    rawText = input.content.toString('utf8').replace(/<[^>]+>/g, ' ');
    kind = 'html';
  } else if (m === 'text/markdown') {
    rawText = chunkMarkdown(input.content.toString('utf8')).map((c) => c.text).join('\n\n');
    kind = 'markdown';
  } else {
    rawText = chunkStructured(input.content.toString('utf8')).map((c) => c.text).join('\n\n');
    kind = 'text';
  }
  const parserLatencyMs = performance.now() - parseStart;
  emit({
    phase: 'parsed',
    kind,
    characterCount: rawText.length,
    parserLatencyMs,
    metadata: extraMetadata,
  });

  const chunkStart = performance.now();
  const chunks = (kind === 'pdf' ? chunkText(rawText) : chunkStructured(rawText)) as ReadonlyArray<{ text: string; tokenCount: number }>;
  const chunkerLatencyMs = performance.now() - chunkStart;
  const totalTokens = chunks.reduce((n, c) => n + c.tokenCount, 0);
  emit({
    phase: 'chunked',
    chunkCount: chunks.length,
    totalTokens,
    chunkerLatencyMs,
  });

  if (chunks.length === 0) {
    return {
      documentId: documentId as never,
      hash,
      chunks: [],
      alreadyExisted: false,
    };
  }

  const batchSize = options.batchSize ?? 16;
  let processed = 0;
  const allEmbeddings: number[][] = [];
  for (let i = 0; i < chunks.length; i += batchSize) {
    const batchIndex = Math.floor(i / batchSize);
    const batch = chunks.slice(i, i + batchSize).map((c) => c.text);
    const embeddings = await deps.embedder.embedDocuments(batch);
    for (const e of embeddings) allEmbeddings.push([...e]);
    processed += batch.length;
    emit({
      phase: 'embedding',
      processed,
      total: chunks.length,
      batchIndex,
      batchSize: batch.length,
    });
  }

  emit({ phase: 'persisting', chunkCount: chunks.length });

  const { Chunk, ChunkModality, brandId } = await import('./domain/index.js');
  const persisted: { id: string; tokenCount: number }[] = [];
  const toAdd: unknown[] = [];
  for (let i = 0; i < chunks.length; i++) {
    const c = chunks[i];
    const e = allEmbeddings[i];
    if (!c || !e) continue;
    const chunkId = `chk_${documentId}_${i.toString(36)}`;
    const metadata: Record<string, string> = {
      hash,
      kind,
      ...(input.metadata ? { ...input.metadata } : {}),
      ...(input.filename ? { filename: input.filename } : {}),
      ...extraMetadata,
    };
    toAdd.push(
      new Chunk({
        id: brandId(chunkId as never),
        workspaceId: input.workspaceId,
        ownerId: input.ownerId,
        collectionId: input.collectionId,
        documentId: brandId(documentId as never),
        modality: ChunkModality.Text,
        text: c.text,
        embedding: e,
        metadata,
        tokenCount: c.tokenCount,
        createdAt: new Date(),
      }),
    );
    persisted.push({ id: chunkId, tokenCount: c.tokenCount });
  }
  await deps.store.addBatch(toAdd as never);

  emit({
    phase: 'indexed',
    chunkCount: chunks.length,
    totalTokens,
    totalLatencyMs: performance.now() - start,
  });

  void silentIngest;

  return {
    documentId: documentId as never,
    hash,
    chunks: persisted.map((p) => ({ id: brandId(p.id as never), tokenCount: p.tokenCount })),
    alreadyExisted: false,
  };
};