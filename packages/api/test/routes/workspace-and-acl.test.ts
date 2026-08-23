import { describe, expect, it } from 'vitest';

import { Hono } from 'hono';
import { brandId, type DocumentPrincipalStore, type JwtClaims, type WorkspaceMemberStore } from '@raghub/core';
import { workspaceRoutes } from '../../src/routes/workspaces.js';
import { documentAclRoutes } from '../../src/routes/document-acl.js';

const buildClaims = (workspaceId: string, sub: string, isAdmin = false): JwtClaims => ({
  workspace_id: workspaceId,
  sub,
  is_admin: isAdmin,
  iat: 0,
  exp: 0,
});

const authedHono = (claims: JwtClaims): Hono => {
  const h = new Hono();
  h.use('*', async (c, next) => {
    c.set('claims' as never, claims as never);
    await next();
  });
  return h;
};

describe('workspaceRoutes', () => {
  it('lists members', async () => {
    const memberStore: WorkspaceMemberStore = {
      upsert: async () => ({}) as never,
      get: async () => null,
      list: async () => [
        {
          userId: brandId('usr_a'),
          role: 'owner' as const,
          joinedAt: new Date(),
        },
      ],
      remove: async () => undefined,
      close: async () => undefined,
    };
    const app = workspaceRoutes({ memberStore });
    const authed = authedHono(buildClaims('wsp_1', 'usr_x'));
    authed.route('/', app);
    const res = await authed.request('/v1/workspaces/members');
    const body = (await res.json()) as { members: readonly unknown[] };
    expect(body.members.length).toBe(1);
  });

  it('non-admin invite is rejected', async () => {
    const memberStore: WorkspaceMemberStore = {
      upsert: async () => ({}) as never,
      get: async () => ({ userId: brandId('usr_x'), role: 'member' as const, joinedAt: new Date() }),
      list: async () => [],
      remove: async () => undefined,
      close: async () => undefined,
    };
    const app = workspaceRoutes({ memberStore });
    const authed = authedHono(buildClaims('wsp_1', 'usr_x'));
    authed.route('/', app);
    const res = await authed.request('/v1/workspaces/members', {
      method: 'POST',
      headers: { 'content-type': 'application/json' },
      body: JSON.stringify({ email: 'a@b.c', role: 'admin' }),
    });
    expect(res.status).toBe(403);
  });
});

describe('documentAclRoutes', () => {
  it('admin can grant, owner can list', async () => {
    const memberStore: WorkspaceMemberStore = {
      upsert: async () => ({}) as never,
      get: async () => ({ userId: brandId('usr_admin'), role: 'admin' as const, joinedAt: new Date() }),
      list: async () => [],
      remove: async () => undefined,
      close: async () => undefined,
    };
    const principalStore: DocumentPrincipalStore = {
      grant: async () => undefined,
      revoke: async () => undefined,
      listByDocument: async () => [],
      listByPrincipal: async () => [],
      hasAccess: async () => true,
      applyDefaultAcl: async () => undefined,
      close: async () => undefined,
    };
    const documentStore = {
      getById: async () => ({
        id: brandId('doc_1'),
        workspaceId: brandId('wsp_1'),
        ownerId: brandId('usr_admin'),
        filename: 'a.txt',
        mimeType: 'text/plain',
        hash: 'h',
        byteSize: 1,
        metadata: {},
        status: 'ready' as const,
        createdAt: new Date(),
      }),
    } as never;
    const app = documentAclRoutes({ principalStore, memberStore, documentStore });
    const authed = authedHono(buildClaims('wsp_1', 'usr_admin'));
    authed.route('/', app);
    const grantRes = await authed.request('/v1/documents/doc_1/principals', {
      method: 'POST',
      headers: { 'content-type': 'application/json' },
      body: JSON.stringify({
        principalType: 'user',
        principalId: 'usr_bob',
        permission: 'read',
      }),
    });
    expect(grantRes.status).toBe(200);
    const listRes = await authed.request('/v1/documents/doc_1/principals');
    expect(listRes.status).toBe(200);
  });
});