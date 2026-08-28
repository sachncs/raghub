/**
 * documentIngestHandler — closes the async ingest loop.
 *
 * The HTTP layer enqueues a `document.ingest` job and returns 202
 * (see routes/documents.ts). This handler picks it up:
 *
 *   1. loads the bytes from LocalFileStorage via documentBytesKey()
 *   2. parses the collection id from document metadata
 *   3. calls ingest({ embedder, vectorStore })
 *   4. flips the document row to ready (or failed)
 *   5. writes an audit event on success/failure
 *
 * Failures are caught and recorded as audit events so the
 * document row's status is the source of truth — the job row's
 * status is also flipped to 'failed' by the JobWorker.
 */

import {
  brandId,
  type CollectionId,
  type DocumentId,
  type DocumentLifecycleStatusValue,
  type Embedder,
  type LocalFileStorage,
  type SqliteAuditEventStore,
  type SqliteDocumentStore,
  type UserId,
  type VectorStore,
  type WorkspaceId,
  documentBytesKey,
  ingest,
  DocumentLifecycleStatus,
} from '@raghub/core';

import type { JobHandler } from '../job-worker.js';

export interface DocumentIngestHandlerDeps {
  readonly workspaceId: WorkspaceId;
  readonly db: unknown;
  readonly documentStore: SqliteDocumentStore;
  readonly audit: SqliteAuditEventStore;
  readonly fileStorage: LocalFileStorage;
  readonly embedder: Embedder;
  readonly vectorStore: VectorStore;
}

const status = (s: string): DocumentLifecycleStatusValue => {
  switch (s) {
    case 'ready':
      return DocumentLifecycleStatus.Ready;
    case 'failed':
      return DocumentLifecycleStatus.Failed;
    case 'indexing':
      return DocumentLifecycleStatus.Indexing;
    default:
      return DocumentLifecycleStatus.Pending;
  }
};

export const documentIngestHandler =
  (deps: DocumentIngestHandlerDeps): JobHandler =>
  async (job): Promise<void> => {
    const payload = job.payload as {
      documentId?: string;
      filename?: string;
      mimeType?: string;
    };
    if (!payload.documentId) throw new Error('document.ingest payload missing documentId');
    const documentId = brandId<DocumentId>(payload.documentId);
    const ownerId = brandId<UserId>(job.ownerId);
    const key = documentBytesKey(deps.workspaceId, documentId);

    const bytes = await deps.fileStorage.get(key);
    if (bytes === null) {
      throw new Error(`document bytes not found at ${key}`);
    }
    const buffer = Buffer.isBuffer(bytes) ? bytes : Buffer.from(bytes);

    const doc = await deps.documentStore.getById(deps.workspaceId, documentId);
    if (!doc) throw new Error(`document row not found: ${documentId}`);

    const collectionIdStr =
      (doc.metadata['collection_id'] as string | undefined) ?? `col_${ownerId}`;
    const collectionId = brandId<CollectionId>(collectionIdStr);

    await deps.documentStore.setStatus(documentId, deps.workspaceId, DocumentLifecycleStatus.Indexing);

    try {
      const result = await ingest(
        {
          workspaceId: deps.workspaceId,
          ownerId,
          collectionId,
          filename: payload.filename ?? doc.filename,
          mimeType: payload.mimeType ?? doc.mimeType,
          content: buffer,
          metadata: doc.metadata,
        },
        { embedder: deps.embedder, store: deps.vectorStore },
      );

      await deps.documentStore.setStatus(
        documentId,
        deps.workspaceId,
        status(result.alreadyExisted ? 'ready' : 'ready'),
      );

      await deps.audit.record({
        kind: 'ingest.complete',
        workspaceId: deps.workspaceId,
        actorId: ownerId,
        resourceId: documentId,
        detail: {
          chunks: result.chunks.length,
          alreadyExisted: result.alreadyExisted,
        },
      });
    } catch (err) {
      await deps.documentStore.setStatus(documentId, deps.workspaceId, DocumentLifecycleStatus.Failed);
      await deps.audit.record({
        kind: 'ingest.failure',
        workspaceId: deps.workspaceId,
        actorId: ownerId,
        resourceId: documentId,
        detail: { error: err instanceof Error ? err.message : String(err) },
      });
      throw err;
    }
  };