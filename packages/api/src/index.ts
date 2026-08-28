/**
 * @raghub/api — server entrypoint.
 *
 * Boots a single Hono process that:
 *   - opens (or creates) the top-level workspace registry at
 *     $RAGHUB_WORKSPACE_HOME/registry.db
 *   - provisions one WorkspaceWithSettings per authenticated request
 *     via the WorkspacePool
 *   - binds a real Embedder (FeatureHashing by default, OpenAI when
 *     RAGHUB_EMBEDDER_API_KEY is set) and a BcryptHasher + JwtService
 *   - resolves per-workspace stores (users, documents, members, ...)
 *     from the FIRST registered workspace for single-tenant mode;
 *     multi-workspace mode uses WorkspaceContext (see
 *     workspace-context.ts) on a per-request basis
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
  BcryptHasher as BcryptHasherImpl,
  type BcryptHasher,
  type ConversationStore,
  type DocumentPrincipalStore,
  type DocumentStore,
  type Embedder,
  FeatureHashingEmbedder,
  FsLocalFileStorage,
  type JwtService,
  JwtService as JwtServiceImpl,
  type LocalFileStorage,
  type SessionStore,
  SqliteAuditEventStore,
  SqliteConversationStore,
  type SqliteDocumentPrincipalStoreOptions,
  SqliteDocumentPrincipalStore,
  type SqliteDocumentStoreOptions,
  SqliteDocumentStore,
  type SqliteJobQueueOptions,
  SqliteJobQueue,
  type SqliteJobQueue as SqliteJobQueueType,
  type SqliteSessionStoreOptions,
  SqliteSessionStore,
  type SqliteUserStoreOptions,
  SqliteUserStore,
  SqliteVecStore,
  type SqliteVecStoreOptions,
  SqliteWorkspaceMemberStore,
  type SqliteWorkspaceMemberStore as SqliteWorkspaceMemberStoreType,
  type UserStore,
  type VectorStore,
  type WorkspaceMemberStore,
  type WorkspaceRegistry,
  defaultRegistryPath,
  openFileWorkspaceRegistry,
  openWorkspace,
  brandId,
} from '@raghub/core';
import { serve } from '@hono/node-server';
import Database from 'better-sqlite3';
import { openEncryptedWorkspace } from '@raghub/core';

import { passVaultRef, workspaceRegistry } from './workspace-vault.js';

import { createApp } from './app.js';
import { documentIngestHandler } from './handlers/document-ingest.js';
import { buildStubOrchestrator } from './orchestrator-stub.js';
import { WorkspacePool } from './workspace-pool.js';
import { WorkspaceWorkerSupervisor } from './workspace-supervisor.js';

const HOME = process.env['RAGHUB_WORKSPACE_HOME'] ?? `${process.env['HOME'] ?? '/tmp'}/.raghub`;
const PORT = Number(process.env['RAGHUB_API_PORT'] ?? 3000);
const VERSION = process.env['RAGHUB_VERSION'] ?? '0.1.0';

const DEV_JWT_SECRET = 'dev-secret-change-me-please-32-bytes-min';
const resolveJwtSecret = (): string => {
  const secret = process.env['RAGHUB_JWT_SECRET'];
  if (secret && secret.length > 0) return secret;
  if (process.env['NODE_ENV'] === 'production') {
    throw new Error(
      'RAGHUB_JWT_SECRET is required when NODE_ENV=production; refusing to start with a dev fallback.',
    );
  }
  return DEV_JWT_SECRET;
};

const openRegistry = async (path: string): Promise<WorkspaceRegistry> => {
  return openFileWorkspaceRegistry({ registryPath: path }, (p: string) => {
    const db = new Database(p);
    return db as never;
  });
};

const buildEmbedder = (): Embedder => {
  const apiKey = process.env['RAGHUB_EMBEDDER_API_KEY'] ?? process.env['OPENAI_API_KEY'];
  const model = process.env['RAGHUB_EMBEDDER_MODEL'] ?? 'text-embedding-3-large';
  const dim = Number(process.env['RAGHUB_VECTOR_EMBEDDING_DIM'] ?? 3072);
  if (!apiKey) {
    return new FeatureHashingEmbedder(model, dim);
  }
  /* Lazy import so the package builds without the SDK. */
  return new (require('@raghub/core').OpenAIEmbedder)(
    { model, apiKey, batchSize: 64 },
    dim,
  ) as Embedder;
};

export interface BootOptions {
  readonly home?: string;
  readonly port?: number;
}

export interface BootResult {
  app: ReturnType<typeof createApp>;
  registry: WorkspaceRegistry;
  pool: WorkspacePool;
  embedder: Embedder;
  jwt: JwtService;
  hasher: BcryptHasher;
  /**
   * Stores bound to the first registered workspace, when one exists.
   * For multi-workspace production, routes should resolve these via
   * WorkspaceContext.from(c) instead.
   */
  defaultWorkspace: {
    readonly workspaceId: string;
    readonly path: string;
    readonly userStore: UserStore;
    readonly documentStore: DocumentStore;
    readonly documentPrincipalStore: DocumentPrincipalStore;
    readonly memberStore: SqliteWorkspaceMemberStoreType;
    readonly sessionStore: SessionStore;
    readonly conversationStore: ConversationStore;
    readonly jobQueue: SqliteJobQueueType;
    readonly audit: SqliteAuditEventStore;
    readonly vectorStore: VectorStore;
  } | null;
  readonly fileStorage: LocalFileStorage;
  worker: undefined;
}

const wireFirstWorkspaceStores = async (
  registry: WorkspaceRegistry,
): Promise<BootResult['defaultWorkspace']> => {
  const list = await registry.list();
  const first = list[0];
  if (!first) return null;
  /* The first workspace is opened in plaintext mode here so the
   * server has at least one set of stores to bind to legacy routes.
   * The WorkspacePool + passphrase cookie are still the source of
   * truth for per-user decryption; this is a dev/single-tenant
   * fallback. */
  const handle = await openWorkspace({ path: first.path });
  const db = handle.db as unknown as SqliteUserStoreOptions['db'];
  const userStore = new SqliteUserStore({ db });
  const documentStore = new SqliteDocumentStore({ db });
  const documentPrincipalStore = new SqliteDocumentPrincipalStore({ db });
  const memberStore = new SqliteWorkspaceMemberStore({ db });
  const sessionStore = new SqliteSessionStore({ db });
  const conversationStore = new SqliteConversationStore({ db });
  const jobQueue = new SqliteJobQueue({ db });
  const audit = new SqliteAuditEventStore({ db });
  const vectorStore = new SqliteVecStore({
    db,
    embeddingDim: Number(process.env['RAGHUB_VECTOR_EMBEDDING_DIM'] ?? 3072),
  });
  return {
    workspaceId: first.workspaceId,
    path: first.path,
    userStore,
    documentStore,
    documentPrincipalStore,
    memberStore,
    sessionStore,
    conversationStore,
    jobQueue,
    audit,
    vectorStore,
  };
};

export const boot = async (opts: BootOptions = {}): Promise<BootResult> => {
  const home = opts.home ?? HOME;
  const port = opts.port ?? PORT;
  void port;

  const registry = await openRegistry(defaultRegistryPath(home));
  const pool = new WorkspacePool({ registry });

  const hasher = new BcryptHasherImpl(10);
  const jwt = new JwtServiceImpl({
    secret: resolveJwtSecret(),
    algorithm: 'HS256',
    ttlSeconds: 86_400,
  });

  const embedder = buildEmbedder();
  const defaultWorkspace = await wireFirstWorkspaceStores(registry);
  const fileStorage = new FsLocalFileStorage({ root: `${home}/files` });

  /* Orchestrator — see ./orchestrator-stub.ts for the dev-mode
   * wiring (NoOpTelemetry + stub AgentRegistry + StubLlm).
   * Multi-tenant builds swap this for a real RagAgent. */
  const orchestrator = await buildStubOrchestrator();

  const app = createApp({
    userStore: defaultWorkspace?.userStore ?? null,
    documentStore: defaultWorkspace?.documentStore ?? null,
    documentPrincipalStore: defaultWorkspace?.documentPrincipalStore ?? null,
    memberStore: defaultWorkspace?.memberStore ?? null,
    sessionStore: defaultWorkspace?.sessionStore ?? null,
    conversationStore: defaultWorkspace?.conversationStore ?? null,
    jobQueue: defaultWorkspace?.jobQueue ?? null,
    audit: defaultWorkspace?.audit ?? null,
    fileStorage,
    embedder,
    vectorStore: defaultWorkspace?.vectorStore ?? null,
    hasher,
    jwt,
    orchestrator,
    registry,
    pool,
    version: VERSION,
  });

  return { app, registry, pool, embedder, jwt, hasher, defaultWorkspace, fileStorage, worker: undefined };
};

export const start = async (): Promise<void> => {
  const { app, pool, registry: reg, defaultWorkspace, fileStorage, embedder } = await boot();
  /* Dev/e2e workspace supervisor — scans the registry every
   * pollMs and starts a JobWorker for any new registered
   * workspace. The passphrase vault is in-memory only; for
   * production this should be replaced with a KMS-backed
   * mechanism. */
  const vault = new Map<string, string>();
  const supervisor = new WorkspaceWorkerSupervisor({
    resolveDb: (workspaceId) => {
      const entryPromise = reg.resolve(workspaceId as never);
      const passphrase = vault.get(workspaceId);
      if (!passphrase) return null;
      return entryPromise.then(async (entry) => {
        if (!entry) return null;
        try {
          const handle = await openEncryptedWorkspace({ path: entry.path, passphrase });
          return handle.db as never;
        } catch {
          return null;
        }
      });
    },
    resolveHandler: (workspaceId) =>
      documentIngestHandler({
        pool,
        fileStorage,
        embedder,
      }),
    pollMs: 2_000,
  });
  if (defaultWorkspace) {
    vault.set(defaultWorkspace.workspaceId, '');
    workspaceRegistry.value.add(defaultWorkspace.workspaceId);
  }
  passVaultRef.value = vault;
  supervisor.start();
  // eslint-disable-next-line no-console
  console.log('raghub-api: WorkspaceWorkerSupervisor started');

  serve({ fetch: app.fetch, port: PORT }, (info: { port: number }) => {
    // eslint-disable-next-line no-console
    console.log(`raghub-api listening on http://localhost:${info.port}`);
  });
  const close = (): void => {
    void supervisor.stop();
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