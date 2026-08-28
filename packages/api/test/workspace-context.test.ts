/**
 * WorkspaceContext — the per-request bundle of Sqlite*Stores.
 *
 * Asserts that:
 *   - workspaceContextFrom() reads claims + passphrase cookie,
 *     asks the pool for a handle, and exposes fresh Sqlite*
 *     Store instances that share the handle.db
 *   - The close() no-op doesn't crash
 */

import { afterEach, beforeEach, describe, expect, it } from 'vitest';

import {
  BcryptHasher,
  type Database,
  FeatureHashingEmbedder,
  type JwtService,
  JwtService,
  type Llm,
  openEncryptedWorkspace,
  type Settings,
  SqliteVecStore,
  StubLlm,
  type WorkspaceWithSettings,
  brandId,
  defaultRegistryPath,
  openFileWorkspaceRegistry,
  type WorkspaceId,
} from '@raghub/core';

import { Hono } from 'hono';
import { mkdtempSync, rmSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';

import { jwtAuthMiddleware } from '../src/middleware/auth.js';
import { workspaceContextFrom } from '../src/workspace-context.js';
import { WorkspacePool } from '../src/workspace-pool.js';

void jwtAuthMiddleware;

const PASSPHRASE = 'correct horse battery staple';

const buildSettings = (): Settings => ({
  auth: {
    jwtSecret: 'test-secret-test-secret-test-secret-32',
    jwtAlgorithm: 'HS256',
    tokenTtlSeconds: 3600,
    bcryptRounds: 4,
  },
  tenants: { isolation: 'row_level' },
  vectorStore: { backend: 'sqlite_vec', path: './x.db', embeddingDim: 64 },
  embedder: { provider: 'feature_hashing', model: 'test', batchSize: 16 },
  llm: { provider: 'openai', model: 'stub', temperature: 0 },
  hybrid: {
    denseWeight: 0.6,
    sparseWeight: 0.4,
    rrfK: 60,
    colbert: false,
  },
  orchestrator: {
    mode: 'graph',
    ordering: 'standard',
    topK: 10,
    reranker: 'identity',
    multimodal: { enabled: false, embeddingModel: 'x', embeddingDim: 64 },
    traceCorpus: { enabled: false, representation: 'semantic', topK: 5 },
  },
  telemetry: { provider: 'noop' },
  secrets: { tenantSecretsKey: '00'.repeat(32) },
});

const fakeClaims = (workspaceId: WorkspaceId, sub: string) => ({
  workspace_id: workspaceId as unknown as string,
  sub,
  is_admin: true,
  iat: 0,
  exp: 0,
});

describe('workspaceContextFrom', () => {
  let home: string;
  let pool: WorkspacePool;
  let handle: WorkspaceWithSettings;
  let registry: Awaited<ReturnType<typeof openFileWorkspaceRegistry>>;

  const dbFactory = (p: string): Database => {
    const Database = require('better-sqlite3') as new (path: string) => Database;
    return new Database(p) as unknown as Database;
  };

  beforeEach(async () => {
    home = mkdtempSync(join(tmpdir(), 'raghub-wctx-'));
    registry = await openFileWorkspaceRegistry(
      { registryPath: defaultRegistryPath(home) },
      dbFactory,
    );
    const wsDir = join(home, 'workspaces', 'wsp_1', 'workspace.db');
    const { mkdirSync } = await import('node:fs');
    mkdirSync(join(home, 'workspaces', 'wsp_1'), { recursive: true });
    handle = await openEncryptedWorkspace({ path: wsDir, passphrase: PASSPHRASE });
    await registry.register({
      workspaceId: brandId<WorkspaceId>('wsp_1'),
      path: wsDir,
      encryption: 'passphrase-aes-256-gcm',
    });
    pool = new WorkspacePool({ registry });
  });

  afterEach(async () => {
    pool?.closeAll();
    await registry?.close();
    rmSync(home, { recursive: true, force: true });
  });

  it('bundles every Sqlite*Store from the pool handle', async () => {
    const embedder = new FeatureHashingEmbedder('test', 64);
    const vectorStore = new SqliteVecStore({ db: handle.db, embeddingDim: 64 });
    const app = new Hono();
    app.use('*', async (c, next) => {
      c.set('claims', fakeClaims(brandId<WorkspaceId>('wsp_1'), 'usr_test'));
      c.set('passphrase', PASSPHRASE);
      await next();
    });
    let captured: unknown = null;
    app.get('/probe', async (c) => {
      const ctx = await workspaceContextFrom(c, { pool, embedder, vectorStore });
      captured = ctx;
      return c.json({ ok: true });
    });
    const res = await app.request('/probe');
    expect(res.status).toBe(200);
    const ctx = captured as Awaited<ReturnType<typeof workspaceContextFrom>>;
    expect(ctx.workspaceId).toBe('wsp_1');
    expect(ctx.userId).toBe('usr_test');
    expect(typeof ctx.userStore.getByEmail).toBe('function');
    expect(typeof ctx.documentStore.listForUser).toBe('function');
    expect(typeof ctx.memberStore.list).toBe('function');
    expect(typeof ctx.sessionStore.upsert).toBe('function');
    expect(typeof ctx.conversationStore).toBe('object');
    expect(typeof ctx.jobQueue.enqueue).toBe('function');
    expect(typeof ctx.audit.record).toBe('function');
    expect(typeof ctx.memory.remember).toBe('function');
    expect(ctx.embedder).toBe(embedder);
    expect(ctx.vectorStore).toBe(vectorStore);
    expect(ctx.handle.db).toBeDefined();
    await ctx.close();
  });

  it('throws when passphrase is missing', async () => {
    const embedder = new FeatureHashingEmbedder('test', 64);
    const vectorStore = new SqliteVecStore({ db: handle.db, embeddingDim: 64 });
    const app = new Hono();
    app.use('*', async (c, next) => {
      c.set('claims', fakeClaims(brandId<WorkspaceId>('wsp_1'), 'usr_test'));
      /* passphrase NOT set */
      await next();
    });
    let caught: unknown = null;
    app.get('/probe', async (c) => {
      try {
        await workspaceContextFrom(c, { pool, embedder, vectorStore });
      } catch (e) {
        caught = e;
      }
      return c.json({ ok: true });
    });
    await app.request('/probe');
    expect(String(caught)).toMatch(/passphrase missing/);
  });
});

/* Suppress unused-import warnings for type-only helpers in some
 * branches of the same file. */
const _types: { hasher: BcryptHasher; jwt: JwtService; llm: Llm; settings: Settings } = {
  hasher: undefined as never,
  jwt: undefined as never,
  llm: undefined as never,
  settings: undefined as never,
};
void _types;
void StubLlm;