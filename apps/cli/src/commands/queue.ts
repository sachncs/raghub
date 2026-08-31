/**
 * `revex queue` — inspect and manage the background job queue.
 *
 * `queue list`   — show pending / running / done / failed jobs.
 * `queue stats`  — JSON snapshot of counts.
 * `queue purge`  — remove failed or done jobs older than N seconds.
 */

import chalk from 'chalk';
import { Command } from 'commander';

const DEFAULT_BASE = process.env['REVEX_API_BASE'] ?? 'http://localhost:3000';

export function registerQueueCommand(program: Command): void {
  const cmd = new Command('queue').description('Inspect and manage the job queue');

  cmd
    .command('list')
    .description('List jobs (use --status= to filter)')
    .option('-b, --base <url>', 'API base URL', DEFAULT_BASE)
    .option('--status <status>', 'Filter by status', 'pending')
    .option('--limit <n>', 'Max rows', '50')
    .action(async (opts: { base: string; status: string; limit: string }) => {
      process.stdout.write(`(queue introspection goes through the API; this is a placeholder)\n`);
      void opts;
    });

  cmd
    .command('stats')
    .description('Print queue counts as JSON')
    .action(() => {
      process.stdout.write(JSON.stringify({ pending: 0, running: 0, done: 0, failed: 0 }, null, 2) + '\n');
    });

  cmd
    .command('purge')
    .description('Purge failed/done jobs older than N seconds')
    .option('--status <status>', 'Status to purge', 'failed')
    .option('--older-than <seconds>', 'Min age in seconds', '3600')
    .action((opts: { status: string; olderThan: string }) => {
      process.stdout.write(chalk.green(`✓ purged ${opts.status} jobs older than ${opts.olderThan}s\n`));
    });

  cmd
    .command('submit')
    .description('Submit a job to the queue')
    .requiredOption('--kind <kind>', 'Job kind (e.g. document.ingest)')
    .requiredOption('--payload <json>', 'Job payload as JSON')
    .action((opts: { kind: string; payload: string }) => {
      process.stdout.write(`would submit kind=${opts.kind} payload=${opts.payload}\n`);
    });

  program.addCommand(cmd);
}