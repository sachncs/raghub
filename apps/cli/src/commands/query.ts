/**
 * `revex query` — issue a single non-streaming query.
 *
 * Bypasses the web UI and calls `/v1/query` directly. Useful for
 * shell scripts and smoke tests.
 */

import chalk from 'chalk';
import { Command } from 'commander';

const DEFAULT_BASE = process.env['REVEX_API_BASE'] ?? 'http://localhost:3000';

export function registerQueryCommand(program: Command): void {
  program
    .command('query')
    .description('Send a single query to the running API')
    .argument('<question>', 'The question to ask')
    .option('-b, --base <url>', 'API base URL', DEFAULT_BASE)
    .option('-s, --session <id>', 'Session id', `s_${Date.now().toString(36)}`)
    .action(async (question: string, opts: { base: string; session: string }) => {
      const res = await fetch(`${opts.base}/v1/query`, {
        method: 'POST',
        headers: { 'content-type': 'application/json' },
        body: JSON.stringify({ question, session_id: opts.session }),
      });
      if (!res.ok) {
        const text = await res.text();
        process.stderr.write(chalk.red(`query failed: ${res.status} ${text}\n`));
        process.exit(1);
      }
      const body = (await res.json()) as { answer?: string };
      process.stdout.write(`${body.answer ?? '(no answer)'}\n`);
    });
}