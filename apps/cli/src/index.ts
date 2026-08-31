#!/usr/bin/env node
/**
 * @revex/cli — public surface.
 *
 * Wires `commander` and registers every subcommand. Mirrors the
 * Python release's `cli.py` / `commands/` surface.
 */

import { Command } from 'commander';

import { registerInitCommand } from './commands/init.js';
import { registerServerCommand } from './commands/server.js';
import { registerIngestCommand } from './commands/ingest.js';
import { registerQueryCommand } from './commands/query.js';
import { registerConfigCommand } from './commands/config.js';
import { registerTenantCommand } from './commands/tenant.js';
import { registerBackupCommand } from './commands/backup.js';
import { registerQueueCommand } from './commands/queue.js';
import { registerMigrateCommand } from './commands/migrate.js';
import { registerFeedbackCommand } from './commands/feedback.js';
import { registerEvalCommand } from './commands/eval.js';

export * from './commands/init.js';
export * from './commands/server.js';
export * from './commands/ingest.js';
export * from './commands/query.js';
export * from './commands/config.js';
export * from './commands/tenant.js';
export * from './commands/backup.js';
export * from './commands/queue.js';
export * from './commands/migrate.js';
export * from './commands/feedback.js';
export * from './commands/eval.js';

export const VERSION = '1.1.0';

export function buildProgram(): Command {
  const program = new Command();
  program
    .name('revex')
    .description('Revex — hybrid retrieval for teams.')
    .version(VERSION);

  registerInitCommand(program);
  registerServerCommand(program);
  registerIngestCommand(program);
  registerQueryCommand(program);
  registerConfigCommand(program);
  registerTenantCommand(program);
  registerBackupCommand(program);
  registerQueueCommand(program);
  registerMigrateCommand(program);
  registerFeedbackCommand(program);
  registerEvalCommand(program);

  return program;
}

const program = buildProgram();
if (import.meta.url === `file://${process.argv[1]}`) {
  program.parseAsync(process.argv).catch((err: unknown) => {
    process.stderr.write(`error: ${err instanceof Error ? err.message : String(err)}\n`);
    process.exit(1);
  });
}