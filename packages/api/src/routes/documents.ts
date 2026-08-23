/**
 * Document routes — upload, list, delete, status.
 *
 * POST /v1/documents is multipart/form-data: a `file` part plus
 * optional `collection_id` + metadata form fields. The handler
 * runs `ingest()` synchronously (Phase 1 keeps the API simple;
 * Phase 2 will wire `SqliteJobQueue.enqueue()` and a worker).
 */

import { Hono } from 'hono';

import {
  brandId,
  type CollectionId,
  type Document,
  type DocumentId,
  type DocumentStore,
  type Embedder,
  type SessionStore,
  type SqliteJobQueue,
  type WorkspaceId,
  type User,
  type UserId,
  type VectorStore,
  ingest,
  hashDocument,
  DocumentLifecycleStatus,
} from '@raghub/core';

import { getClaims } from '../middleware/auth.js';

export interface DocumentsRouteDeps {
  readonly userStore: { getById(workspaceId: WorkspaceId, id: UserId): Promise<User | null> };
  readonly documentStore: DocumentStore;
  readonly sessionStore: SessionStore;
  readonly jobQueue: SqliteJobQueue;
  readonly embedder: Embedder;
  readonly vectorStore: VectorStore;
}

interface UploadResponse {
  readonly documentId: DocumentId;
  readonly hash: string;
  readonly chunks: number;
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
    const workspaceId = brandId<WorkspaceId>(claims.workspace_id);
    const userId = brandId<UserId>(claims.sub);
    const user = await deps.userStore.getById(workspaceId, userId);
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

    const arrayBuf = await file.arrayBuffer();
    const buffer = Buffer.from(arrayBuf);
    const hash = hashDocument(buffer);

    const document = await deps.documentStore.upsert({
      workspaceId,
      ownerId: userId,
      filename: file.name || 'upload',
      mimeType: file.type || 'application/octet-stream',
      hash,
      byteSize: buffer.byteLength,
      metadata,
    });

    const seen = await deps.documentStore.getByHash(workspaceId, hash);
    if (seen && seen.id !== document.id) {
      const response: UploadResponse = {
        documentId: seen.id,
        hash,
        chunks: 0,
        alreadyExisted: true,
      };
      return c.json(response, 200);
    }

    await deps.jobQueue.enqueue({
      workspaceId,
      ownerId: userId,
      kind: 'document.ingest',
      payload: { documentId: document.id, hash, byteSize: buffer.byteLength },
    });

    const result = await ingest(
      {
        workspaceId,
        ownerId: userId,
        collectionId,
        filename: file.name || 'upload',
        mimeType: file.type || 'application/octet-stream',
        content: buffer,
        metadata,
      },
      { embedder: deps.embedder, store: deps.vectorStore },
    );

    await deps.documentStore.setStatus(document.id, workspaceId, DocumentLifecycleStatus.Ready);

    const response: UploadResponse = {
      documentId: document.id,
      hash,
      chunks: result.chunks.length,
      alreadyExisted: result.alreadyExisted,
    };
    return c.json(response, 200);
  });

  app.get('/v1/documents', async (c) => {
    const claims = getClaims(c);
    const workspaceId = brandId<WorkspaceId>(claims.workspace_id);
    const userId = brandId<UserId>(claims.sub);
    const docs = await deps.documentStore.listForUser(workspaceId, userId);
    return c.json({ documents: docs.map((d: Document) => d.toJSON()) });
  });

  app.delete('/v1/documents/:id', async (c) => {
    const claims = getClaims(c);
    const workspaceId = brandId<WorkspaceId>(claims.workspace_id);
    const userId = brandId<UserId>(claims.sub);
    const id = brandId<DocumentId>(c.req.param('id'));
    const doc = await deps.documentStore.getById(workspaceId, id);
    if (!doc) {
      return c.json({ error: { code: 'raghub_error', message: 'document not found' } }, 404);
    }
    if (doc.ownerId !== userId) {
      return c.json({ error: { code: 'authorization_error', message: 'not the owner' } }, 403);
    }
    await deps.vectorStore.deleteByDocument(id, workspaceId);
    return c.json({ ok: true });
  });

  return app;
};