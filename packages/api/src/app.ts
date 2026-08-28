/**
 * @raghub/api — public surface.
 *
 * Wires the Hono server with every middleware + route group, the
 * SSE streaming helpers, the workspace pool, and the auth/error
 * middleware exports.
 *
 * The server is a single-workspace-by-default process. The
 * `workspacePaths` resolver maps JWT claims → on-disk `.db` paths
 * via a top-level `WorkspaceRegistry`. Multi-workspace deployments
 * register multiple workspaces into the registry and the pool
 * handles the rest.
 */

import { Hono } from 'hono';

import {
  type BcryptHasher,
  type ConversationStore,
  type DocumentPrincipalStore,
  type DocumentStore,
  type Embedder,
  type JwtService,
  type LocalFileStorage,
  type SqliteAuditEventStore,
  type SqliteJobQueue,
  type SessionStore,
  type UserStore,
  type VectorStore,
  type WorkspaceMemberStore,
  type WorkspaceRegistry,
} from '@raghub/core';
import type { Orchestrator } from '@raghub/orchestrator';

import { jwtAuthMiddleware } from './middleware/auth.js';
import { errorMiddleware } from './middleware/error.js';
import { rateLimitMiddleware } from './middleware/rate-limit.js';
import { securityHeadersMiddleware } from './middleware/security.js';
import { authRoutes } from './routes/auth.js';
import { documentAclRoutes } from './routes/document-acl.js';
import { documentsRoutes } from './routes/documents.js';
import { logoutRoutes } from './routes/logout.js';
import { meRoutes } from './routes/me.js';
import { operationalRoutes } from './routes/operational.js';
import { queryRoutes } from './routes/query.js';
import { settingsRoutes } from './routes/settings.js';
import { workspaceRoutes } from './routes/workspaces.js';
import type { WorkspacePool } from './workspace-pool.js';

export interface AppDeps {
  readonly userStore: UserStore | null;
  readonly documentStore: DocumentStore | null;
  readonly documentPrincipalStore: DocumentPrincipalStore | null;
  readonly memberStore: WorkspaceMemberStore | null;
  readonly sessionStore: SessionStore | null;
  readonly conversationStore: ConversationStore | null;
  readonly jobQueue: SqliteJobQueue | null;
  readonly fileStorage: LocalFileStorage | null;
  readonly audit?: SqliteAuditEventStore | null;
  readonly embedder: Embedder;
  readonly vectorStore: VectorStore | null;
  readonly hasher: BcryptHasher;
  readonly jwt: JwtService;
  readonly orchestrator: Orchestrator;
  readonly registry: WorkspaceRegistry;
  readonly pool: WorkspacePool;
  readonly startTime?: number;
  readonly version?: string;
}

export const createApp = (deps: AppDeps): Hono => {
  const app = new Hono();
  app.use('*', securityHeadersMiddleware());
  app.use('*', errorMiddleware());
  app.use('*', rateLimitMiddleware({ bypassPaths: ['/health', '/readyz'] }));
  app.get('/health', (c) => c.json({ ok: true }));
  app.get('/readyz', async (c) => {
    try {
      const list = await deps.registry.list();
      return c.json({ ok: true, workspaces: list.length, poolSize: deps.pool.size() });
    } catch (e) {
      return c.json({ ok: false, error: e instanceof Error ? e.message : String(e) }, 503);
    }
  });

  app.route(
    '/',
    authRoutes({ userStore: deps.userStore, hasher: deps.hasher, jwt: deps.jwt, registry: deps.registry }),
  );

  const protectedApp = new Hono();
  protectedApp.use('*', jwtAuthMiddleware(deps.jwt));
  protectedApp.route(
    '/',
    meRoutes({ userStore: deps.userStore, sessionStore: deps.sessionStore, jwt: deps.jwt }),
  );
  protectedApp.route('/', queryRoutes({ orchestrator: deps.orchestrator }));
  protectedApp.route(
    '/',
    documentsRoutes({
      pool: deps.pool,
      userStore: deps.userStore,
      documentStore: deps.documentStore,
      sessionStore: deps.sessionStore,
      jobQueue: deps.jobQueue,
      fileStorage: deps.fileStorage,
      vectorStore: deps.vectorStore,
      embedder: deps.embedder,
    }),
  );
  protectedApp.route(
    '/',
    documentAclRoutes({
      principalStore: deps.documentPrincipalStore,
      memberStore: deps.memberStore,
      documentStore: deps.documentStore,
      audit: deps.audit ?? null,
    }),
  );
  protectedApp.route(
    '/',
    workspaceRoutes({ memberStore: deps.memberStore, audit: deps.audit ?? null }),
  );
  protectedApp.route(
    '/',
    logoutRoutes({ sessionStore: deps.sessionStore }),
  );
  protectedApp.route(
    '/',
    settingsRoutes({ pool: deps.pool, audit: deps.audit ?? null }),
  );
  protectedApp.route(
    '/',
    operationalRoutes({
      registry: deps.registry,
      pool: deps.pool,
      startTime: deps.startTime ?? Date.now(),
      version: deps.version ?? '0.0.0',
    }),
  );
  app.route('/', protectedApp);

  return app;
};

export { errorMiddleware, jwtAuthMiddleware, rateLimitMiddleware };