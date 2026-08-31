/**
 * `revex migrate` — workspace migration helpers.
 *
 * `migrate tenant-split` — split a single workspace into two.
 * `migrate pgvector-to-sqlite` — pull chunks out of pgvector and
 *                                write them to a fresh sqlite-vec
 *                                workspace.
 *
 * These are placeholder entrypoints; the actual migration logic
 * lives in `@revex/api` so the server can hold the workspace
 * lock for the duration.
 */

import chalk from 'chalk';
import { Command } from 'commander';

export function registerMigrateCommand(program: Command): void {
  const cmd = new Command('migrate').description('Workspace migration helpers');

  cmd
    .command('tenant-split')
    .description('Split one workspace into two by document ownership')
    .requiredOption('--from <workspaceId>', 'Source workspace')
    .requiredOption('--to <workspaceId>', 'Target workspace')
    .requiredOption('--owner <userId>', 'Owner user id for the split subset')
    .action((opts: { from: string; to: string; owner: string }) => {
      process.stdout.write(
        chalk.green(
          `✓ would split ${opts.from} -> ${opts.to} (subset of ${opts.owner}'s docs)\n`,
        ),
      );
    });

  cmd
    .command('pgvector-to-sqlite')
    .description('Pull chunks out of a pgvector DB and into a sqlite-vec workspace')
    .requiredOption('--dsn <dsn>', 'pgvector connection string')
    .option('--vector-dim <dim>', 'Vector dim (default 3072)', '3072')
    .action((opts: { dsn: string; vectorDim: string }) => {
      process.stdout.write(
        chalk.green(`✓ would migrate pgvector (${opts.vectorDim}d) to sqlite-vec\n`),
      );
      void opts.dsn;
    });

  program.addCommand(cmd);
}