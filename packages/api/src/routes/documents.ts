/**
 * Document routes — upload, list, delete, status.
 *
 * POST /v1/documents is multipart/form-data: a `file` part plus
 * optional `collection_id` + metadata form fields. The handler
 * persists the bytes to LocalFileStorage and enqueues a
 * `document.ingest` job on the per-workspace SqliteJobQueue. The
 * background JobWorker (started from `start()`) picks the job up,
 * loads the bytes back, runs ingest(), and flips the row's status
 * to ready/failed.
 *
 * The endpoint returns 202 with `{ documentId, status: 'pending' }`
 * — the chat UI polls /v1/documents until rows leave the pending
 * state.
 */

import { Hono } from 'hono';

import {
  brandId,
  type CollectionId,
  type Document,
  type DocumentId,
  type DocumentStore,
  type Embedder,
  type LocalFileStorage,
  type SessionStore,
  type SqliteJobQueue,
  type WorkspaceId,
  type User,
  type UserId,
  type VectorStore,
  documentBytesKey,
  hashDocument,
} from '@raghub/core';

import { getClaims } from '../middleware/auth.js';
import { workspaceContextFrom } from '../workspace-context.js';
import type { WorkspacePool } from '../workspace-pool.js';

export interface DocumentsRouteDeps {
  readonly pool: WorkspacePool;
  readonly userStore: { getById(workspaceId: WorkspaceId, id: UserId): Promise<User | null> } | null;
  readonly documentStore: DocumentStore | null;
  readonly sessionStore: SessionStore | null;
  readonly jobQueue: SqliteJobQueue | null;
  readonly fileStorage: LocalFileStorage | null;
  readonly vectorStore?: VectorStore | null;
  readonly embedder: Embedder;
}

interface UploadResponse {
  readonly documentId: DocumentId;
  readonly hash: string;
  readonly status: 'pending' | 'ready' | 'indexing' | 'failed';
  readonly byteSize: number;
  readonly alreadyExisted: boolean;
}

const defaultCollectionId = (userId: UserId): string => `col_${userId}`;

const readFile = async (form: FormData, field: string): Promise<File | null> => {
  const value = form.get(field);
  if (value instanceof File) return value;
  return null;
};

const readString = (form: FormData, field: string): string | null => {
  const v = form.get(field);
  return typeof v === 'string' ? v : null;
};

export const documentsRoutes = (deps: DocumentsRouteDeps): Hono => {
  const app = new Hono();

  app.post('/v1/documents', async (c) => {
    const claims = getClaims(c);
    const ctx = await workspaceContextFrom(c, {
      pool: deps.pool,
      embedder: deps.embedder,
      vectorStore: deps.vectorStore ?? (await import('@raghub/core')).SqliteVecStore as never,
    });
    const documentStore = ctx.documentStore;
    const jobQueue = ctx.jobQueue;
    const fileStorage = deps.fileStorage;
    if (!fileStorage) throw new Error('fileStorage missing');
    const embedder = deps.embedder;
    const vectorStore = ctx.vectorStore;
    const workspaceId = ctx.workspaceId;
    const userId = ctx.userId;
    const user = await ctx.userStore.getById(workspaceId, userId);
    if (!user) {
      return c.json({ error: { code: 'auth_error', message: 'user not found' } }, 401);
    }
    const form = await c.req.formData();
    const file = await readFile(form, 'file');
    if (!file) {
      return c.json({ error: { code: 'ingestion_error', message: 'file part required' } }, 400);
    }
    const collectionIdStr = readString(form, 'collection_id') ?? defaultCollectionId(userId);
    const collectionId = brandId<CollectionId>(collectionIdStr);
    const metadata: Record<string, string> = {};
    for (const [k, v] of form.entries()) {
      if (k !== 'file' && k !== 'collection_id' && typeof v === 'string') {
        metadata[k] = v;
      }
    }
    metadata['collection_id'] = collectionIdStr;

    const arrayBuf = await file.arrayBuffer();
    const buffer = Buffer.from(arrayBuf);
    const hash = hashDocument(buffer);

    const seen = await documentStore.getByHash(workspaceId, hash);
    if (seen) {
      const response: UploadResponse = {
        documentId: seen.id,
        hash,
        status: seen.status,
        byteSize: seen.byteSize,
        alreadyExisted: true,
      };
      return c.json(response, 200);
    }

    const document = await documentStore.upsert({
      workspaceId,
      ownerId: userId,
      filename: file.name || 'upload',
      mimeType: file.type || 'application/octet-stream',
      hash,
      byteSize: buffer.byteLength,
      metadata,
    });

    await fileStorage.put(documentBytesKey(workspaceId, document.id), buffer);

    await jobQueue.enqueue({
      workspaceId,
      ownerId: userId,
      kind: 'document.ingest',
      payload: {
        documentId: document.id,
        hash,
        byteSize: buffer.byteLength,
        filename: file.name || 'upload',
        mimeType: file.type || 'application/octet-stream',
      },
    });

    const response: UploadResponse = {
      documentId: document.id,
      hash,
      status: 'pending',
      byteSize: buffer.byteLength,
      alreadyExisted: false,
    };
    return c.json(response, 202);
  });

  app.get('/v1/documents', async (c) => {
    const ctx = await workspaceContextFrom(c, {
      pool: deps.pool,
      embedder: deps.embedder,
      vectorStore: deps.vectorStore ?? null,
    });
    const docs = await ctx.documentStore.listForUser(ctx.workspaceId, ctx.userId);
    if (process.env['RAGHUB_DEBUG_DOCS']) {
      // eslint-disable-next-line no-console
      console.log(`[docs.get] dbPath=${ctx.handle.path} wsId=${ctx.workspaceId} userId=${ctx.userId} count=${docs.length} statuses=${docs.map((d) => d.status).join(',')}`);
      /* Raw row inspection: what does better-sqlite3 actually see? */
      const rawRows = ctx.handle.db
        .prepare('SELECT id, status FROM documents WHERE workspace_id = ? AND owner_id = ?')
        .all(ctx.workspaceId, ctx.userId) as Array<{ id: string; status: string }>;
      // eslint-disable-next-line no-console
      console.log(`[docs.get] raw rows: ${JSON.stringify(rawRows)}`);
    }
    return c.json({ documents: docs.map((d: Document) => d.toJSON()) });
  });

  app.delete('/v1/documents/:id', async (c) => {
    const ctx = await workspaceContextFrom(c, {
      pool: deps.pool,
      embedder: deps.embedder,
      vectorStore: deps.vectorStore,
    });
    const documentStore = ctx.documentStore;
    const vectorStore = ctx.vectorStore;
    if (!vectorStore) {
      return c.json({ error: { code: 'configuration_error', message: 'vector store unavailable' } }, 503);
    }
    const workspaceId = ctx.workspaceId;
    const userId = ctx.userId;
    const id = brandId<DocumentId>(c.req.param('id'));
    const doc = await documentStore.getById(workspaceId, id);
    if (!doc) {
      return c.json({ error: { code: 'raghub_error', message: 'document not found' } }, 404);
    }
    if (doc.ownerId !== userId) {
      return c.json({ error: { code: 'authorization_error', message: 'not the owner' } }, 403);
    }
    if (!vectorStore) {
      return c.json({ error: { code: 'configuration_error', message: 'vector store unavailable' } }, 503);
    }
    await vectorStore.deleteByDocument(id, workspaceId);
    return c.json({ ok: true });
  });

  return app;
};