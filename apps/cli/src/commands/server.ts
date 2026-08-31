/**
 * `revex server` — start the Hono API server.
 *
 * Wires the same composition used by `pnpm --filter @revex/api dev`
 * but reads `revex.config.json` if present and respects its `server`
 * block.
 */

import chalk from 'chalk';
import { Command } from 'commander';

import { start } from '../runtime.js';

export function registerServerCommand(program: Command): void {
  program
    .command('server')
    .description('Start the Revex API server')
    .option('-p, --port <port>', 'Bind port', '3000')
    .option('--host <host>', 'Bind host', '127.0.0.1')
    .option('--no-watch', 'Disable job worker autostart')
    .action(async (opts: { port: string; host: string; watch: boolean }) => {
      const port = Number(opts.port);
      process.stdout.write(chalk.cyan(`revex-api listening on http://${opts.host}:${port}\n`));
      await start({ port, host: opts.host, autostartWorker: opts.watch });
    });
}