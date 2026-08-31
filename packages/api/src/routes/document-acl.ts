/**
 * Document ACL routes.
 *
 * POST   /v1/documents/:id/principals           grant a permission
 * DELETE /v1/documents/:id/principals           revoke
 * GET    /v1/documents/:id/principals           list grants
 *
 * Only document owners or workspace admins can mutate ACL.
 */

import { Hono } from 'hono';

import {
  brandId,
  type DocumentId,
  type Embedder,
  type DocumentPrincipalStore,
  type DocumentPrincipalType,
  type DocumentStore,
  type SqliteAuditEventStore,
  type UserId,
  type VectorStore,
  type WorkspaceId,
  canManageWorkspace,
  type WorkspaceMemberStore,
} from '@revex/core';

import { getClaims } from '../middleware/auth.js';
import { workspaceContextFrom } from '../workspace-context.js';

export interface DocumentAclRouteDeps {
  readonly pool: import('../workspace-pool.js').WorkspacePool;
  readonly principalStore: DocumentPrincipalStore | null;
  readonly memberStore: WorkspaceMemberStore | null;
  readonly documentStore: DocumentStore | null;
  readonly audit: SqliteAuditEventStore | null;
  readonly embedder: Embedder;
  readonly vectorStore: VectorStore | null;
}

const validTypes: readonly DocumentPrincipalType[] = ['user', 'role', 'group'];
const validPerms = ['read', 'admin'] as const;
type Perm = (typeof validPerms)[number];

const isType = (s: string): s is DocumentPrincipalType =>
  (validTypes as readonly string[]).includes(s);
const isPerm = (s: string): s is Perm => (validPerms as readonly string[]).includes(s);

export const documentAclRoutes = (deps: DocumentAclRouteDeps): Hono => {
  const app = new Hono();

  app.get('/v1/documents/:id/principals', async (c) => {
    const claims = getClaims(c);
    const ctx = await workspaceContextFrom(c, {
      pool: deps.pool,
      embedder: deps.embedder,
      vectorStore: deps.vectorStore,
    });
    const documentStore = ctx.documentStore;
    const memberStore = ctx.memberStore;
    const principalStore = ctx.documentPrincipalStore;
    const workspaceId = ctx.workspaceId;
    const userId = ctx.userId;
    const audit = deps.audit;
    const doc = await documentStore.getById(workspaceId, brandId<DocumentId>(c.req.param('id')));
    if (!doc) {
      return c.json({ error: { code: 'revex_error', message: 'document not found' } }, 404);
    }
    const me = await memberStore.get(workspaceId, userId);
    const isOwner = doc.ownerId === userId;
    const isAdmin = me ? canManageWorkspace(me.role) : false;
    if (!isOwner && !isAdmin) {
      return c.json({ error: { code: 'authorization_error', message: 'only owner or admin can read ACL' } }, 403);
    }
    const list = await principalStore.listByDocument(doc.id);
    return c.json({
      principals: list.map((p) => ({
        documentId: p.documentId,
        principalType: p.principalType,
        principalId: p.principalId,
        permission: p.permission,
        grantedBy: p.grantedBy,
        grantedAt: p.grantedAt.toISOString(),
      })),
    });
  });

  app.post('/v1/documents/:id/principals', async (c) => {
    const claims = getClaims(c);
    const ctx = await workspaceContextFrom(c, {
      pool: deps.pool,
      embedder: deps.embedder,
      vectorStore: deps.vectorStore,
    });
    const documentStore = ctx.documentStore;
    const memberStore = ctx.memberStore;
    const principalStore = ctx.documentPrincipalStore;
    const workspaceId = ctx.workspaceId;
    const userId = ctx.userId;
    const audit = deps.audit;
    const doc = await documentStore.getById(workspaceId, brandId<DocumentId>(c.req.param('id')));
    if (!doc) {
      return c.json({ error: { code: 'revex_error', message: 'document not found' } }, 404);
    }
    const me = await memberStore.get(workspaceId, userId);
    const isOwner = doc.ownerId === userId;
    const isAdmin = me ? canManageWorkspace(me.role) : false;
    if (!isOwner && !isAdmin) {
      return c.json({ error: { code: 'authorization_error', message: 'only owner or admin can grant' } }, 403);
    }
    const body = (await c.req.json().catch(() => ({}))) as {
      principalType?: string;
      principalId?: string;
      permission?: string;
    };
    if (!body.principalType || !isType(body.principalType) || !body.principalId || !body.permission || !isPerm(body.permission)) {
      return c.json({ error: { code: 'revex_error', message: 'principalType, principalId, permission required' } }, 400);
    }
    await principalStore.grant({
      documentId: doc.id,
      principalType: body.principalType,
      principalId: body.principalId,
      permission: body.permission,
      grantedBy: userId,
    });
    if (deps.audit) {
      await deps.audit.record({
        kind: 'document.acl.grant',
        workspaceId,
        actorId: userId,
        resourceId: doc.id,
        detail: {
          principalType: body.principalType,
          principalId: body.principalId,
          permission: body.permission,
        },
      });
    }
    return c.json({ ok: true });
  });

  app.delete('/v1/documents/:id/principals', async (c) => {
    const claims = getClaims(c);
    const ctx = await workspaceContextFrom(c, {
      pool: deps.pool,
      embedder: deps.embedder,
      vectorStore: deps.vectorStore,
    });
    const documentStore = ctx.documentStore;
    const memberStore = ctx.memberStore;
    const principalStore = ctx.documentPrincipalStore;
    const workspaceId = ctx.workspaceId;
    const userId = ctx.userId;
    const audit = deps.audit;
    const doc = await documentStore.getById(workspaceId, brandId<DocumentId>(c.req.param('id')));
    if (!doc) {
      return c.json({ error: { code: 'revex_error', message: 'document not found' } }, 404);
    }
    const me = await memberStore.get(workspaceId, userId);
    const isOwner = doc.ownerId === userId;
    const isAdmin = me ? canManageWorkspace(me.role) : false;
    if (!isOwner && !isAdmin) {
      return c.json({ error: { code: 'authorization_error', message: 'only owner or admin can revoke' } }, 403);
    }
    const body = (await c.req.json().catch(() => ({}))) as {
      principalType?: string;
      principalId?: string;
      permission?: string;
    };
    if (!body.principalType || !isType(body.principalType) || !body.principalId || !body.permission || !isPerm(body.permission)) {
      return c.json({ error: { code: 'revex_error', message: 'principalType, principalId, permission required' } }, 400);
    }
    await principalStore.revoke({
      documentId: doc.id,
      principalType: body.principalType,
      principalId: body.principalId,
      permission: body.permission,
    });
    if (deps.audit) {
      await deps.audit.record({
        kind: 'document.acl.revoke',
        workspaceId,
        actorId: userId,
        resourceId: doc.id,
        detail: {
          principalType: body.principalType,
          principalId: body.principalId,
          permission: body.permission,
        },
      });
    }
    return c.json({ ok: true });
  });

  return app;
};