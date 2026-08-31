/**
 * `revex ingest` — upload + ingest a single document.
 *
 * Uses the same Hono `POST /v1/documents` endpoint the web UI
 * hits. Streams bytes + filename through fetch so the API can
 * persist to LocalFileStorage and enqueue the ingest job.
 */

import { readFile } from 'node:fs/promises';
import { basename } from 'node:path';

import chalk from 'chalk';
import { Command } from 'commander';

const DEFAULT_BASE = process.env['REVEX_API_BASE'] ?? 'http://localhost:3000';

export function registerIngestCommand(program: Command): void {
  program
    .command('ingest')
    .description('Ingest a document into the workspace')
    .argument('<path>', 'Path to the document to ingest')
    .option('-b, --base <url>', 'API base URL', DEFAULT_BASE)
    .option('-w, --workspace-id <id>', 'Target workspace id')
    .action(async (path: string, cmd: { base: string; workspaceId?: string }) => {
      const bytes = await readFile(path);
      const filename = basename(path);
      const form = new FormData();
      form.append('file', new Blob([bytes]), filename);
      const url = `${cmd.base}/v1/documents`;
      const res = await fetch(url, {
        method: 'POST',
        body: form,
        headers: cmd.workspaceId ? { 'x-revex-workspace': cmd.workspaceId } : {},
      });
      if (!res.ok) {
        const text = await res.text();
        process.stderr.write(chalk.red(`ingest failed: ${res.status} ${text}\n`));
        process.exit(1);
      }
      const body = (await res.json()) as { id?: string; status?: string };
      process.stdout.write(
        chalk.green(`✓ ingested ${filename}\n`) +
          `  id: ${body.id ?? 'unknown'}\n` +
          `  status: ${body.status ?? 'pending'}\n`,
      );
    });
}