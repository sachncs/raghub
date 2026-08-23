import { describe, expect, it } from 'vitest';

import { createApp } from '../src/app.js';

const integration = process.env['RAGHUB_RUN_API_TESTS'] === '1';
const itg = integration ? it : it.skip;

describe('createApp (shape only)', () => {
  itg('builds a Hono app with /health', () => {
    const app = createApp({
      userStore: undefined as never,
      hasher: undefined as never,
      jwt: undefined as never,
      orchestrator: undefined as never,
      documentStore: undefined as never,
      sessionStore: undefined as never,
      conversationStore: undefined as never,
      jobQueue: undefined as never,
      embedder: undefined as never,
      vectorStore: undefined as never,
      workspacePaths: { resolve: async () => null } as never,
    });
    expect(typeof app.fetch).toBe('function');
  });
});