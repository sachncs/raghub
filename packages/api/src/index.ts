/**
 * @raghub/api — server entrypoint.
 *
 * Boots a single Hono process that:
 *   - opens (or creates) the top-level workspace registry at
 *     $RAGHUB_WORKSPACE_HOME/registry.db
 *   - provisions one WorkspaceWithSettings per authenticated request
 *     via the WorkspacePool
 *   - listens on $RAGHUB_API_PORT (default 3000)
 *
 * Usage:
 *   $ pnpm --filter @raghub/api start
 * or with custom home:
 *   $ RAGHUB_WORKSPACE_HOME=/var/lib/raghub \
 *     RAGHUB_API_PORT=3000 \
 *     pnpm --filter @raghub/api start
 */

import {
  type BcryptHasher,
  type JwtService,
  type SessionStore,
  type UserStore,
  BcryptHasher as BcryptHasherImpl,
  JwtService as JwtServiceImpl,
  type VectorStore,
  type Embedder,
  type DocumentStore,
  type ConversationStore,
  type SqliteJobQueue,
  type WorkspaceMemberStore,
  type WorkspaceRegistry,
  type DocumentPrincipalStore,
  brandId,
  defaultRegistryPath,
  openFileWorkspaceRegistry,
} from '@raghub/core';
import { Orchestrator } from '@raghub/orchestrator';
import { serve } from '@hono/node-server';
import Database from 'better-sqlite3';

import { createApp } from './app.js';
import { WorkspacePool } from './workspace-pool.js';

const HOME = process.env['RAGHUB_WORKSPACE_HOME'] ?? `${process.env['HOME'] ?? '/tmp'}/.raghub`;
const PORT = Number(process.env['RAGHUB_API_PORT'] ?? 3000);
const VERSION = process.env['RAGHUB_VERSION'] ?? '0.1.0';

const openRegistry = async (path: string): Promise<WorkspaceRegistry> =>
  openFileWorkspaceRegistry({ registryPath: path }, (p: string) => {
    const db = new Database(p);
    return db as never;
  });

const ensureSingleton = async <T>(label: string, factory: () => T): Promise<T> => factory();

export interface BootOptions {
  readonly home?: string;
  readonly port?: number;
}

export const boot = async (opts: BootOptions = {}): Promise<{
  app: ReturnType<typeof createApp>;
  registry: WorkspaceRegistry;
  pool: WorkspacePool;
}> => {
  const home = opts.home ?? HOME;
  const port = opts.port ?? PORT;
  void port;

  const registry = await openRegistry(defaultRegistryPath(home));

  const pool = new WorkspacePool({ registry });

  const hasher: BcryptHasher = new BcryptHasherImpl(10);
  const jwtSecret = process.env['RAGHUB_JWT_SECRET'] ?? 'dev-secret-change-me-please-32-bytes-min';
  const jwt: JwtService = new JwtServiceImpl({ secret: jwtSecret, algorithm: 'HS256', ttlSeconds: 86_400 });

  const userStore: UserStore = {} as never;
  const documentStore: DocumentStore = {} as never;
  const documentPrincipalStore: DocumentPrincipalStore = {} as never;
  const memberStore: WorkspaceMemberStore = {} as never;
  const sessionStore: SessionStore = {} as never;
  const conversationStore: ConversationStore = {} as never;
  const jobQueue: SqliteJobQueue = {} as never;
  const embedder: Embedder = {} as never;
  const vectorStore: VectorStore = {} as never;

  const orchestrator = new Orchestrator({
    telemetry: {} as never,
    workspaceId: brandId('wsp_local'),
    agents: {} as never,
    tools: {} as never,
  });

  void ensureSingleton;

  const app = createApp({
    userStore,
    documentStore,
    documentPrincipalStore,
    memberStore,
    sessionStore,
    conversationStore,
    jobQueue,
    embedder,
    vectorStore,
    hasher,
    jwt,
    orchestrator,
    registry,
    pool,
    version: VERSION,
  });

  return { app, registry, pool };
};

export const start = async (): Promise<void> => {
  const { app, pool } = await boot();
  serve({ fetch: app.fetch, port: PORT }, (info) => {
    // eslint-disable-next-line no-console
    console.log(`raghub-api listening on http://localhost:${info.port}`);
  });
  const close = (): void => {
    pool.closeAll();
    process.exit(0);
  };
  process.on('SIGINT', close);
  process.on('SIGTERM', close);
};

const entry = process.argv[1] ?? '';
if (entry.endsWith('index.js') || entry.endsWith('index.ts')) {
  start().catch((err) => {
    // eslint-disable-next-line no-console
    console.error('failed to start:', err);
    process.exit(1);
  });
}