/**
 * Ingest pipeline.
 *
 * `ingest()` is the single entry point used by every code path that
 * turns raw bytes into searchable chunks:
 *
 * 1. detect the source kind from the MIME type / extension,
 * 2. run the right parser (PDF, plain text, HTML, fallback),
 * 3. chunk the extracted text with the text/PDF chunkers,
 * 4. embed each chunk through the embedder (batched),
 * 5. persist the chunks to the vector store with content-addressed
 *    SHA-256 hashing so re-ingest is idempotent.
 *
 * `hashDocument` matches the legacy "skip if hash unchanged" semantics.
 */

import { createHash } from 'node:crypto';

import {
  brandId,
  Chunk,
  ChunkModality,
  type ChunkId,
  type CollectionId,
  type DocumentId,
  type WorkspaceId,
  type UserId,
} from './domain/index.js';
import type { Embedder } from './embedder/index.js';
import type { VectorStore } from './stores/index.js';

import { chunkMarkdown, chunkPdf, chunkStructured, chunkText } from './chunker/index.js';
import type { TextChunk } from './chunker/index.js';

export interface IngestInput {
  readonly workspaceId: WorkspaceId;
  readonly ownerId: UserId;
  readonly collectionId: CollectionId;
  readonly filename: string;
  readonly mimeType: string;
  readonly content: Buffer;
  readonly metadata?: Readonly<Record<string, string>>;
}

export interface IngestOutput {
  readonly documentId: DocumentId;
  readonly hash: string;
  readonly chunks: readonly { readonly id: ChunkId; readonly tokenCount: number }[];
  readonly alreadyExisted: boolean;
}

export interface IngestDeps {
  readonly embedder: Embedder;
  readonly store: VectorStore;
  /** Optional idempotency hook; when the hash is seen the ingest is skipped. */
  readonly seenHashes?: (hash: string) => Promise<boolean>;
}

const EXT_KIND: Record<string, 'pdf' | 'html' | 'markdown' | 'text'> = {
  pdf: 'pdf',
  html: 'html',
  htm: 'html',
  md: 'markdown',
  markdown: 'markdown',
  txt: 'text',
  log: 'text',
  csv: 'text',
  json: 'text',
};

const detectKind = (mime: string, filename: string): 'pdf' | 'html' | 'markdown' | 'text' => {
  const m = mime.toLowerCase();
  if (m === 'application/pdf' || m.startsWith('application/pdf')) return 'pdf';
  if (m === 'text/html' || m === 'application/xhtml+xml') return 'html';
  if (m === 'text/markdown') return 'markdown';
  if (m.startsWith('text/')) return 'text';
  const ext = filename.toLowerCase().split('.').pop() ?? '';
  return EXT_KIND[ext] ?? 'text';
};

const stripHtml = (html: string): string =>
  html
    .replace(/<\s*br\s*\/?>/gi, '\n')
    .replace(/<\/(p|div|li|h\d|tr|td|th)>/gi, '\n')
    .replace(/<style[^>]*>[\s\S]*?<\/style>/gi, ' ')
    .replace(/<script[^>]*>[\s\S]*?<\/script>/gi, ' ')
    .replace(/<[^>]+>/g, ' ')
    .replace(/&nbsp;/gi, ' ')
    .replace(/&amp;/gi, '&')
    .replace(/&lt;/gi, '<')
    .replace(/&gt;/gi, '>')
    .replace(/\s+/g, ' ')
    .trim();

const newDocumentId = (hash: string): DocumentId =>
  brandId<DocumentId>(`doc_${hash.slice(0, 16)}`);

const newChunkId = (documentId: DocumentId, index: number): ChunkId =>
  brandId<ChunkId>(`chk_${documentId}_${index.toString(36)}`);

export const hashDocument = (input: Buffer | Uint8Array | string): string =>
  createHash('sha256').update(input).digest('hex');

export const ingest = async (input: IngestInput, deps: IngestDeps): Promise<IngestOutput> => {
  const hash = hashDocument(input.content);
  const documentId = newDocumentId(hash);
  if (deps.seenHashes) {
    const seen = await deps.seenHashes(hash);
    if (seen) {
      return { documentId, hash, chunks: [], alreadyExisted: true };
    }
  }

  const kind = detectKind(input.mimeType, input.filename);
  let rawText = '';
  let extraMetadata: Record<string, string> = {};
  switch (kind) {
    case 'pdf': {
      const result = await chunkPdf(input.content);
      rawText = result.chunks.map((c: TextChunk) => c.text).join('\n');
      extraMetadata = {
        pages: String(result.pages),
        ...stringify(result.metadata),
      };
      break;
    }
    case 'html': {
      rawText = stripHtml(input.content.toString('utf8'));
      break;
    }
    case 'markdown': {
      rawText = chunkMarkdown(input.content.toString('utf8')).map((c) => c.text).join('\n\n');
      break;
    }
    default: {
      rawText = chunkStructured(input.content.toString('utf8')).map((c) => c.text).join('\n\n');
    }
  }

  const chunks: readonly TextChunk[] =
    kind === 'pdf' ? chunkText(rawText) : chunkStructured(rawText);

  if (chunks.length === 0) {
    return { documentId, hash, chunks: [], alreadyExisted: false };
  }

  const embeddings = await deps.embedder.embedDocuments(chunks.map((c) => c.text));

  const persisted: { id: ChunkId; tokenCount: number }[] = [];
  const toAdd: Chunk[] = [];
  for (let i = 0; i < chunks.length; i++) {
    const c = chunks[i];
    const e = embeddings[i];
    if (!c || !e) continue;
    const chunkId = newChunkId(documentId, i);
    const metadata: Record<string, string> = {
      hash,
      kind,
      ...(input.metadata ? { ...input.metadata } : {}),
      ...(input.filename ? { filename: input.filename } : {}),
      ...extraMetadata,
    };
    const chunk = new Chunk({
      id: chunkId,
      workspaceId: input.workspaceId,
      ownerId: input.ownerId,
      collectionId: input.collectionId,
      documentId,
      modality: ChunkModality.Text,
      text: c.text,
      embedding: [...e],
      metadata,
      tokenCount: c.tokenCount,
      createdAt: new Date(),
    });
    toAdd.push(chunk);
    persisted.push({ id: chunkId, tokenCount: c.tokenCount });
  }
  await deps.store.addBatch(toAdd);
  return { documentId, hash, chunks: persisted, alreadyExisted: false };
};

const stringify = (obj: Readonly<Record<string, unknown>>): Record<string, string> => {
  const out: Record<string, string> = {};
  for (const [k, v] of Object.entries(obj)) {
    if (v === null || v === undefined) continue;
    out[k] = typeof v === 'string' ? v : JSON.stringify(v);
  }
  return out;
};

