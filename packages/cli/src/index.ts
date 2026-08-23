#!/usr/bin/env node
/**
 * @raghub/cli — `raghub` binary entry.
 *
 * Registers every command and dispatches argv[0] to the right
 * runner. `runCommand()` exits with the runner's status code.
 */

import { runCommand } from './runner.js';
import { initCommand } from './commands/init.js';
import { pgvectorToSqliteCommand, sqliteImportCommand } from './commands/migrate.js';
import { devCommand } from './commands/dev.js';

const commands = [initCommand, pgvectorToSqliteCommand, sqliteImportCommand, devCommand];

const main = async (): Promise<void> => {
  const argv = process.argv.slice(2);
  const env = process.env as Readonly<Record<string, string | undefined>>;
  const cwd = process.cwd();
  const code = await runCommand(commands, argv, env, cwd);
  process.exit(code);
};

main().catch((e) => {
  console.error(e);
  process.exit(1);
});