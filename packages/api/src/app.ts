/**
 * The Hono app factory.
 *
 * Wires every middleware and route group; the returned `app` can be
 * served by any Hono adapter (`@hono/node-server`, Bun, Cloudflare
 * Workers, Lambda, etc.).
 */

import { Hono } from 'hono';

import type { BcryptHasher, JwtService, UserStore } from '@raghub/core';
import type { Orchestrator } from '@raghub/orchestrator';

import { jwtAuthMiddleware } from './middleware/auth.js';
import { errorMiddleware } from './middleware/error.js';
import { authRoutes } from './routes/auth.js';
import { meRoutes } from './routes/me.js';
import { queryRoutes } from './routes/query.js';

export interface AppDeps {
  readonly userStore: UserStore;
  readonly hasher: BcryptHasher;
  readonly jwt: JwtService;
  readonly orchestrator: Orchestrator;
}

export const createApp = (deps: AppDeps): Hono => {
  const app = new Hono();
  app.use('*', errorMiddleware());
  app.get('/health', (c) => c.json({ ok: true }));

  app.route('/', authRoutes({ userStore: deps.userStore, hasher: deps.hasher, jwt: deps.jwt }));

  const protectedApp = new Hono();
  protectedApp.use('*', jwtAuthMiddleware(deps.jwt));
  protectedApp.route('/', meRoutes({ userStore: deps.userStore, jwt: deps.jwt }));
  protectedApp.route('/', queryRoutes({ orchestrator: deps.orchestrator }));
  app.route('/', protectedApp);

  return app;
};

export { errorMiddleware, jwtAuthMiddleware };