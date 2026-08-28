/**
 * WorkspaceContext — bundles every per-handle store the API needs.
 *
 * Each authenticated request lives behind one of these:
 *   1. `WorkspaceContext.from(c)` resolves the JWT claims + the
 *      passphrase cookie and asks the `WorkspacePool` for an
 *      unlocked `WorkspaceWithSettings`.
 *   2. The returned context owns fresh `Sqlite*Store` instances
 *      built from the shared `db` handle. The stores are cheap to
 *      create and discard per request — the heavy work is the
 *      scrypt-derived key inside the pool.
 *   3. `ctx.close()` is a no-op today (the pool owns the handle),
 *      but exists so call sites don't have to special-case the
 *      "no resources to release" path.
 *
 * Routes that touch only the registry (auth/login, auth/register)
 * don't go through here — they use the pool directly.
 */

import type { Context } from 'hono';

import {
  type ConversationStore,
  type DocumentId,
  type DocumentPrincipalStore,
  type DocumentStore,
  type Embedder,
  type SqliteAuditEventStore,
  type SqliteConversationStore,
  type SqliteDocumentPrincipalStore,
  type SqliteDocumentStore,
  type SqliteJobQueue,
  type SqliteSessionStore,
  type SqliteUserStore,
  type SqliteWorkspaceMemberStore,
  type SqliteWorkspaceMemoryStore,
  type SessionStore,
  type UserId,
  type VectorStore,
  type WorkspaceId,
  type WorkspaceMemberStore,
  type WorkspaceWithSettings,
  SqliteAuditEventStore as SqliteAuditEventStoreImpl,
  SqliteConversationStore as SqliteConversationStoreImpl,
  SqliteDocumentPrincipalStore as SqliteDocumentPrincipalStoreImpl,
  SqliteDocumentStore as SqliteDocumentStoreImpl,
  SqliteJobQueue as SqliteJobQueueImpl,
  SqliteSessionStore as SqliteSessionStoreImpl,
  SqliteUserStore as SqliteUserStoreImpl,
  SqliteWorkspaceMemberStore as SqliteWorkspaceMemberStoreImpl,
  SqliteWorkspaceMemoryStore as SqliteWorkspaceMemoryStoreImpl,
} from '@raghub/core';

import { getClaims, getPassphrase } from './middleware/auth.js';
import type { WorkspacePool } from './workspace-pool.js';

export interface WorkspaceContextDeps {
  readonly pool: WorkspacePool;
  readonly embedder: Embedder;
  readonly vectorStore: VectorStore;
}

export interface WorkspaceContext {
  readonly workspaceId: WorkspaceId;
  readonly userId: UserId;
  readonly handle: WorkspaceWithSettings;
  readonly userStore: SqliteUserStore;
  readonly documentStore: SqliteDocumentStore;
  readonly documentPrincipalStore: SqliteDocumentPrincipalStore;
  readonly memberStore: SqliteWorkspaceMemberStore;
  readonly sessionStore: SqliteSessionStore;
  readonly conversationStore: SqliteConversationStore;
  readonly jobQueue: SqliteJobQueue;
  readonly audit: SqliteAuditEventStore;
  readonly memory: SqliteWorkspaceMemoryStore;
  readonly embedder: Embedder;
  readonly vectorStore: VectorStore;
  close(): Promise<void>;
}

const requirePassphrase = (c: Context): string => {
  const passphrase = getPassphrase(c);
  if (!passphrase) {
    const err = new Error('passphrase missing');
    (err as Error & { status?: number }).status = 401;
    throw err;
  }
  return passphrase;
};

const idParam = (c: Context, name: string): string => {
  const value = c.req.param(name);
  if (!value) throw new Error(`missing path param: ${name}`);
  return value;
};

export class WorkspaceContextImpl implements WorkspaceContext {
  public readonly userStore: SqliteUserStore;
  public readonly documentStore: SqliteDocumentStore;
  public readonly documentPrincipalStore: SqliteDocumentPrincipalStore;
  public readonly memberStore: SqliteWorkspaceMemberStore;
  public readonly sessionStore: SqliteSessionStore;
  public readonly conversationStore: SqliteConversationStore;
  public readonly jobQueue: SqliteJobQueue;
  public readonly audit: SqliteAuditEventStore;
  public readonly memory: SqliteWorkspaceMemoryStore;

  constructor(
    public readonly workspaceId: WorkspaceId,
    public readonly userId: UserId,
    public readonly handle: WorkspaceWithSettings,
    public readonly embedder: Embedder,
    public readonly vectorStore: VectorStore,
  ) {
    const db = handle.db;
    this.userStore = new SqliteUserStoreImpl({ db });
    this.documentStore = new SqliteDocumentStoreImpl({ db });
    this.documentPrincipalStore = new SqliteDocumentPrincipalStoreImpl({ db });
    this.memberStore = new SqliteWorkspaceMemberStoreImpl({ db });
    this.sessionStore = new SqliteSessionStoreImpl({ db });
    this.conversationStore = new SqliteConversationStoreImpl({ db });
    this.jobQueue = new SqliteJobQueueImpl({ db });
    this.audit = new SqliteAuditEventStoreImpl({ db });
    this.memory = new SqliteWorkspaceMemoryStoreImpl({ db });
  }

  public async close(): Promise<void> {
    /* pool owns the handle */
  }
}

export const workspaceContextFrom = async (
  c: Context,
  deps: WorkspaceContextDeps,
): Promise<WorkspaceContext> => {
  const claims = getClaims(c);
  const passphrase = requirePassphrase(c);
  const workspaceId = claims.workspace_id as WorkspaceId;
  const userId = claims.sub as UserId;
  const handle = await deps.pool.get({ workspaceId, userId, passphrase });
  return new WorkspaceContextImpl(
    workspaceId,
    userId,
    handle,
    deps.embedder,
    deps.vectorStore,
  );
};

/**
 * Convenience for routes that need a document id from the path.
 * Centralised so the cast pattern lives in one place.
 */
export const documentIdFrom = (c: Context): DocumentId => {
  return idParam(c, 'id') as DocumentId;
};

/* Re-export store types so route code can import them from this
 * module if it prefers the bundle over individual imports. */
export type {
  ConversationStore,
  DocumentPrincipalStore,
  DocumentStore,
  SessionStore,
  WorkspaceMemberStore,
};