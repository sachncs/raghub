/**
 * `revex feedback` — submit per-message feedback from the CLI.
 *
 * Useful for shell-driven evals and CI smoke tests.
 */

import chalk from 'chalk';
import { Command } from 'commander';

const DEFAULT_BASE = process.env['REVEX_API_BASE'] ?? 'http://localhost:3000';

export function registerFeedbackCommand(program: Command): void {
  program
    .command('feedback')
    .description('Submit a feedback rating for a turn')
    .requiredOption('--turn <turnId>', 'Turn id being rated')
    .requiredOption('--rating <rating>', 'up | down | neutral')
    .option('-c, --comment <text>', 'Optional comment')
    .option('-b, --base <url>', 'API base URL', DEFAULT_BASE)
    .action(async (opts: { turn: string; rating: string; comment?: string; base: string }) => {
      if (!['up', 'down', 'neutral'].includes(opts.rating)) {
        process.stderr.write(chalk.red(`invalid rating: ${opts.rating}\n`));
        process.exit(1);
      }
      const res = await fetch(`${opts.base}/v1/feedback`, {
        method: 'POST',
        headers: { 'content-type': 'application/json' },
        body: JSON.stringify({
          turnId: opts.turn,
          rating: opts.rating,
          comment: opts.comment ?? null,
        }),
      });
      if (!res.ok) {
        process.stderr.write(chalk.red(`feedback failed: ${res.status}\n`));
        process.exit(1);
      }
      process.stdout.write(chalk.green(`✓ feedback recorded\n`));
    });
}