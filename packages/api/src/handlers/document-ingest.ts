/**
 * documentIngestHandler — closes the async ingest loop.
 *
 * The HTTP layer enqueues a `document.ingest` job and returns 202
 * (see routes/documents.ts). This handler picks it up:
 *
 *   1. opens the workspace handle via the pool + passphrase vault
 *   2. loads the bytes from LocalFileStorage via documentBytesKey()
 *   3. parses the collection id from document metadata
 *   4. calls ingest({ embedder, vectorStore })
 *   5. flips the document row to ready (or failed)
 *   6. writes an audit event on success/failure
 *
 * The handler resolves the workspace fresh on each call so the
 * dev/e2e bootstrap doesn't need to pre-bind every store at boot.
 * Production should hand the handler an explicit per-workspace
 * db handle.
 */

import {
  brandId,
  type CollectionId,
  type DocumentId,
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
  SqliteAuditEventStore as SqliteAuditEventStoreImpl,
  SqliteDocumentStore as SqliteDocumentStoreImpl,
  SqliteVecStore as SqliteVecStoreImpl,
} from '@raghub/core';

import { WorkspacePool } from '../workspace-pool.js';
import { passVaultRef } from '../workspace-bootstrap.js';
import type { JobHandler } from '../job-worker.js';

export interface DocumentIngestHandlerDeps {
  readonly pool: WorkspacePool;
  readonly fileStorage: LocalFileStorage;
  readonly embedder: Embedder;
}

export const documentIngestHandler =
  (deps: DocumentIngestHandlerDeps): JobHandler =>
  async (job): Promise<void> => {
    const workspaceIdStr = String(job.workspaceId);
    const payload = job.payload as {
      documentId?: string;
      filename?: string;
      mimeType?: string;
    };
    if (!payload.documentId) throw new Error('document.ingest payload missing documentId');
    const documentId = brandId<DocumentId>(payload.documentId);
    const ownerId = brandId<UserId>(job.ownerId);
    const workspaceId = brandId<WorkspaceId>(workspaceIdStr);

    const passphrase = (await passVaultRef.value?.get(workspaceIdStr)) ?? '';
    const handle = await deps.pool.get({
      workspaceId,
      userId: ownerId,
      passphrase,
    });
    const documentStore: SqliteDocumentStore = new SqliteDocumentStoreImpl({
      db: handle.db as never,
    });
    const audit: SqliteAuditEventStore = new SqliteAuditEventStoreImpl({
      db: handle.db as never,
    });
    const vectorStore: VectorStore = new SqliteVecStoreImpl({
      db: handle.db as never,
      embeddingDim: 64,
    });

    const key = documentBytesKey(workspaceId, documentId);
    const bytes = await deps.fileStorage.get(key);
    if (bytes === null) {
      throw new Error(`document bytes not found at ${key}`);
    }
    const buffer = Buffer.isBuffer(bytes) ? bytes : Buffer.from(bytes);

    const doc = await documentStore.getById(workspaceId, documentId);
    if (!doc) throw new Error(`document row not found: ${documentId}`);

    const collectionIdStr =
      (doc.metadata['collection_id'] as string | undefined) ?? `col_${ownerId}`;
    const collectionId = brandId<CollectionId>(collectionIdStr);

    await documentStore.setStatus(documentId, workspaceId, DocumentLifecycleStatus.Indexing);

    try {
      const result = await ingest(
        {
          workspaceId,
          ownerId,
          collectionId,
          filename: payload.filename ?? doc.filename,
          mimeType: payload.mimeType ?? doc.mimeType,
          content: buffer,
          metadata: doc.metadata,
        },
        { embedder: deps.embedder, store: vectorStore },
      );

      await documentStore.setStatus(documentId, workspaceId, DocumentLifecycleStatus.Ready);

      await audit.record({
        kind: 'ingest.complete',
        workspaceId,
        actorId: ownerId,
        resourceId: documentId,
        detail: { chunks: result.chunks.length, alreadyExisted: result.alreadyExisted },
      });
    } catch (err) {
      await documentStore.setStatus(documentId, workspaceId, DocumentLifecycleStatus.Failed);
      await audit.record({
        kind: 'ingest.failure',
        workspaceId,
        actorId: ownerId,
        resourceId: documentId,
        detail: { error: err instanceof Error ? err.message : String(err) },
      });
      throw err;
    }
  };