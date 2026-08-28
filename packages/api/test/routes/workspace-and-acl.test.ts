import { afterEach, beforeEach, describe, expect, it } from 'vitest';
import { mkdtempSync, rmSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';

import { Hono } from 'hono';
import {
  type DocumentPrincipalStore,
  type JwtClaims,
  SqliteDocumentStore,
  SqliteWorkspaceMemberStore,
  type WorkspaceMemberStore,
  brandId,
  openEncryptedWorkspace,
  type WorkspaceId,
} from '@raghub/core';

import { workspaceRoutes } from '../../src/routes/workspaces.js';
import { documentAclRoutes } from '../../src/routes/document-acl.js';
import { WorkspacePool } from '../../src/workspace-pool.js';
import { openFileWorkspaceRegistry } from '@raghub/core';
import Database from 'better-sqlite3';
import { openEncryptedWorkspace as _oe } from '@raghub/core';

const PASSPHRASE = 'test-passphrase-1234';
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
    c.set('passphrase' as never, PASSPHRASE as never);
    await next();
  });
  return h;
};

interface TestWorkspace {
  home: string;
  pool: WorkspacePool;
  registry: Awaited<ReturnType<typeof openFileWorkspaceRegistry>>;
  workspaceId: WorkspaceId;
  memberStore: SqliteWorkspaceMemberStore;
  documentStore: SqliteDocumentStore;
  close: () => Promise<void>;
}

const setupWorkspace = async (): Promise<TestWorkspace> => {
  const home = mkdtempSync(join(tmpdir(), 'raghub-acl-'));
  const { mkdirSync } = await import('node:fs');
  mkdirSync(join(home, 'workspaces', 'wsp_test'), { recursive: true });
  const registry = await openFileWorkspaceRegistry(
    { registryPath: join(home, 'registry.db') },
    (p) => new Database(p) as never,
  );
  const wsDir = join(home, 'workspaces', 'wsp_test', 'workspace.db');
  const handle = await openEncryptedWorkspace({ path: wsDir, passphrase: PASSPHRASE });
  const memberStore = new SqliteWorkspaceMemberStore({ db: handle.db as never });
  const documentStore = new SqliteDocumentStore({ db: handle.db as never });
  await registry.register({
    workspaceId: brandId<WorkspaceId>('wsp_test'),
    path: wsDir,
    encryption: 'passphrase-aes-256-gcm',
  });
  const pool = new WorkspacePool({ registry });
  return {
    home,
    pool,
    registry,
    workspaceId: brandId<WorkspaceId>('wsp_test'),
    memberStore,
    documentStore,
    close: async () => {
      handle.close();
      await registry.close();
      rmSync(home, { recursive: true, force: true });
    },
  };
};

describe('workspaceRoutes', () => {
  let ctx: TestWorkspace;

  beforeEach(async () => {
    ctx = await setupWorkspace();
  });
  afterEach(async () => {
    await ctx.close();
  });

  it('lists members', async () => {
    await ctx.memberStore.upsert({
      workspaceId: ctx.workspaceId,
      userId: brandId('usr_a'),
      role: 'owner',
    });
    const app = workspaceRoutes({
      pool: ctx.pool,
      memberStore: null,
      audit: null,
      embedder: undefined as never,
      vectorStore: null,
    });
    const authed = authedHono(buildClaims(ctx.workspaceId, 'usr_x'));
    authed.route('/', app);
    const res = await authed.request('/v1/workspaces/members');
    expect(res.status).toBe(200);
    const body = (await res.json()) as { members: readonly unknown[] };
    expect(body.members.length).toBe(1);
  });
});

describe('documentAclRoutes', () => {
  let ctx: TestWorkspace;

  beforeEach(async () => {
    ctx = await setupWorkspace();
  });
  afterEach(async () => {
    await ctx.close();
  });

  it('admin can grant, owner can list', async () => {
    const doc = await ctx.documentStore.upsert({
      workspaceId: ctx.workspaceId,
      ownerId: brandId('usr_admin'),
      filename: 'a.txt',
      mimeType: 'text/plain',
      hash: 'h_admin_test',
      byteSize: 1,
      metadata: {},
    });
    const memberStore: WorkspaceMemberStore = ctx.memberStore;
    const principalStore: DocumentPrincipalStore = {
      grant: async () => undefined,
      revoke: async () => undefined,
      listByDocument: async () => [],
      listByPrincipal: async () => [],
      hasAccess: async () => true,
      applyDefaultAcl: async () => undefined,
      close: async () => undefined,
    };
    const documentStore = ctx.documentStore;
    const app = documentAclRoutes({
      pool: ctx.pool,
      principalStore,
      memberStore,
      documentStore,
      audit: null,
      embedder: undefined as never,
      vectorStore: null,
    });
    const authed = authedHono(buildClaims(ctx.workspaceId, 'usr_admin', true));
    authed.route('/', app);
    const grantRes = await authed.request(`/v1/documents/${doc.id}/principals`, {
      method: 'POST',
      headers: { 'content-type': 'application/json' },
      body: JSON.stringify({
        principalType: 'user',
        principalId: 'usr_bob',
        permission: 'read',
      }),
    });
    expect(grantRes.status).toBe(200);
  });
});