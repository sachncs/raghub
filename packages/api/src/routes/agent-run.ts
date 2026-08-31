/**
 * `POST /v1/agent/run` — non-streaming single-turn chat for tooling.
 *
 * Same contract as `POST /v1/query` but synchronous. Useful for
 * shell scripts and CI smoke tests.
 */

import { Hono } from 'hono';

import type { Orchestrator } from '@revex/orchestrator';

export interface AgentRunRouteDeps {
  readonly orchestrator: Orchestrator;
  readonly getClaims: (c: unknown) => { workspaceId: string; userId: string };
}

interface JwtLike {
  readonly workspace_id?: string;
  readonly sub?: string;
}

export const agentRunRoutes = (deps: AgentRunRouteDeps): Hono => {
  const app = new Hono();

  app.post('/v1/agent/run', async (c) => {
    const body = (await c.req.json().catch(() => ({}))) as {
      question?: string;
      sessionId?: string;
      strategy?: string;
    };
    if (!body.question) {
      return c.json({ error: { code: 'revex_error', message: 'question required' } }, 400);
    }
    const claims = deps.getClaims(c) as JwtLike;
    const result = await deps.orchestrator.run({
      question: body.question,
      user: { id: (claims.sub ?? '') as never, workspaceId: (claims.workspace_id ?? '') as never } as never,
      sessionId: body.sessionId ?? null,
      history: [],
    });
    return c.json({
      answer: result.answer,
      citations: result.citations,
      mode: result.mode,
    });
  });

  return app;
};