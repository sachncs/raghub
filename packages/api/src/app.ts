/**
 * @raghub/api — public surface.
 *
 * Wires the Hono server with every middleware + route group, the
 * SSE streaming helpers, and the auth/error middleware exports.
 */

import { Hono } from 'hono';

import {
  type BcryptHasher,
  type ConversationStore,
  type DocumentStore,
  type Embedder,
  type JwtService,
  type SqliteJobQueue,
  type SessionStore,
  type UserStore,
  type VectorStore,
} from '@raghub/core';
import type { Orchestrator } from '@raghub/orchestrator';

import { jwtAuthMiddleware } from './middleware/auth.js';
import { errorMiddleware } from './middleware/error.js';
import { authRoutes, type WorkspacePathResolver } from './routes/auth.js';
import { documentsRoutes } from './routes/documents.js';
import { meRoutes } from './routes/me.js';
import { queryRoutes } from './routes/query.js';

export interface AppDeps {
  readonly userStore: UserStore;
  readonly documentStore: DocumentStore;
  readonly sessionStore: SessionStore;
  readonly conversationStore: ConversationStore;
  readonly jobQueue: SqliteJobQueue;
  readonly embedder: Embedder;
  readonly vectorStore: VectorStore;
  readonly hasher: BcryptHasher;
  readonly jwt: JwtService;
  readonly orchestrator: Orchestrator;
  readonly workspacePaths: WorkspacePathResolver;
}

export const createApp = (deps: AppDeps): Hono => {
  const app = new Hono();
  app.use('*', errorMiddleware());
  app.get('/health', (c) => c.json({ ok: true }));

  app.route(
    '/',
    authRoutes({
      userStore: deps.userStore,
      hasher: deps.hasher,
      jwt: deps.jwt,
      paths: deps.workspacePaths,
    }),
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
      userStore: deps.userStore,
      documentStore: deps.documentStore,
      sessionStore: deps.sessionStore,
      jobQueue: deps.jobQueue,
      embedder: deps.embedder,
      vectorStore: deps.vectorStore,
    }),
  );
  app.route('/', protectedApp);

  return app;
};

export { errorMiddleware, jwtAuthMiddleware };