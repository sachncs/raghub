/**
 * Query routes — POST /v1/query, POST /v1/query/stream (SSE).
 *
 * Wires the Orchestrator behind the API. The stream route yields
 * `PlannerEvent`s as Server-Sent Events using Hono's `streamSSE`.
 *
 * `invocation_state` is built by the orchestrator itself; this
 * route only translates HTTP into `OrchestratorRequest`.
 */

import { Hono } from 'hono';
import { streamSSE } from 'hono/streaming';

import { brandId, type Orchestrator, type SessionId } from '@raghub/orchestrator';
import type { CollectionId } from '@raghub/core';

import { getClaims } from '../middleware/auth.js';

export interface QueryRouteDeps {
  readonly orchestrator: Orchestrator;
}

interface QueryInput {
  readonly question: string;
  readonly session_id?: string;
  readonly collection_id?: string;
}

export const queryRoutes = (deps: QueryRouteDeps): Hono => {
  const app = new Hono();

  app.post('/v1/query', async (c) => {
    const claims = getClaims(c);
    const body = (await c.req.json().catch(() => ({}))) as Partial<QueryInput>;
    if (!body.question) {
      return c.json({ error: { code: 'raghub_error', message: 'question required' } }, 400);
    }
    const user = await deps.orchestrator['agents']; // touch private for type widening
    void user;
    const sessionId = body.session_id ? brandId<SessionId>(body.session_id) : null;
    const result = await deps.orchestrator.run({
      question: body.question,
      user: null,
      sessionId,
    });
    void claims;
    return c.json({
      answer: result.answer,
      citations: result.citations,
      hits: result.hits.map((h) => ({ id: h.chunk.id, score: h.score, text: h.chunk.text })),
      mode: result.mode,
    });
  });

  app.post('/v1/query/stream', async (c) => {
    const claims = getClaims(c);
    void claims;
    const body = (await c.req.json().catch(() => ({}))) as Partial<QueryInput>;
    if (!body.question) {
      return c.json({ error: { code: 'raghub_error', message: 'question required' } }, 400);
    }
    const sessionId = body.session_id ? brandId<SessionId>(body.session_id) : null;
    void body.collection_id;
    void (null as unknown as CollectionId);

    return streamSSE(c, async (stream) => {
      for await (const ev of deps.orchestrator.stream({
        question: body.question!,
        user: null,
        sessionId,
      })) {
        await stream.writeSSE({
          id: String(ev.step),
          event: ev.kind,
          data: JSON.stringify(ev.payload),
        });
      }
    });
  });

  return app;
};