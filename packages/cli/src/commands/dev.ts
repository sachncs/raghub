/**
 * `raghub dev` — boot the API server in watch mode.
 *
 * Phase 1 imports the compiled @raghub/api app and hands it to
 * `@hono/node-server`. Phase 2 wires the dev-time orchestrator with
 * the FeatureHashingEmbedder + a fresh sqlite-vec store.
 */

import { loadSettings } from '@raghub/core';

import type { Command } from '../runner.js';

export const devCommand: Command = {
  name: 'dev',
  description: 'Run the API server in development mode.',
  usage: 'raghub dev [--port 3000]',
  async run({ flags, env }) {
    const port = Number(flags['port'] ?? 3000);
    try {
      loadSettings(env);
    } catch (e) {
      console.error('[raghub] invalid settings:', e instanceof Error ? e.message : String(e));
      return 1;
    }
    console.log(`[raghub] would start API on port ${port} (Hono adapter not yet wired — see Phase 1 follow-up commit)`);
    return 0;
  },
};