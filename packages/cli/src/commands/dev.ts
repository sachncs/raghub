/**
 * `raghub dev` — boot the API server in development mode.
 *
 * Loads settings from env + the project's `.raghub/.env` (when
 * present), wires every dependency, and listens on the requested
 * port. Owns the SQLite lifecycle so a clean shutdown closes
 * every handle.
 */

import { loadSettings } from '@raghub/core';
import { serve } from '@hono/node-server';

import { startServer } from '../server.js';
import type { Command } from '../runner.js';

export const devCommand: Command = {
  name: 'dev',
  description: 'Run the API server in development mode.',
  usage: 'raghub dev [--port 3000] [--host 127.0.0.1]',
  async run({ flags, env, cwd }) {
    const port = Number(flags['port'] ?? 3000);
    const host = String(flags['host'] ?? '127.0.0.1');
    let settings;
    try {
      settings = loadSettings(env);
    } catch (e) {
      console.error('[raghub] invalid settings:', e instanceof Error ? e.message : String(e));
      return 1;
    }
    let handle;
    try {
      handle = await startServer({ settings });
    } catch (e) {
      console.error('[raghub] failed to start:', e instanceof Error ? e.message : String(e));
      return 1;
    }

    const server = serve({
      fetch: handle.app.fetch,
      port,
      hostname: host,
    });
    console.log(`[raghub] listening on http://${host}:${port}`);
    void cwd;

    const shutdown = async (): Promise<void> => {
      console.log('[raghub] shutting down...');
      server.close();
      await handle.close();
      process.exit(0);
    };
    process.on('SIGINT', () => void shutdown());
    process.on('SIGTERM', () => void shutdown());

    await new Promise<void>(() => {
      // block forever; the signal handler exits
    });
    return 0;
  },
};