/**
 * `revex init` — initialise a new workspace.
 *
 * Creates a workspace directory under $REVEX_WORKSPACE_HOME,
 * optionally writes a default config file, and prints a JSON
 * manifest describing what was created.
 */

import { mkdir, writeFile } from 'node:fs/promises';
import { resolve } from 'node:path';

import chalk from 'chalk';
import { Command } from 'commander';

const DEFAULT_HOME = `${process.env['HOME'] ?? '/tmp'}/.revex`;

export function registerInitCommand(program: Command): void {
  program
    .command('init')
    .description('Initialise a new Revex workspace')
    .requiredOption('-n, --name <name>', 'Workspace name')
    .option('-h, --home <path>', 'Workspace home directory', DEFAULT_HOME)
    .option('--config', 'Write a default revex.config.json')
    .action(async (opts: { name: string; home: string; config?: boolean }) => {
      const home = resolve(opts.home);
      const dir = resolve(home, `wsp_${slug(opts.name)}`);
      await mkdir(dir, { recursive: true });
      const manifest = {
        workspace: opts.name,
        path: dir,
        createdAt: new Date().toISOString(),
        encryption: 'passphrase-aes-256-gcm',
      };
      const manifestPath = resolve(dir, 'manifest.json');
      await writeFile(manifestPath, JSON.stringify(manifest, null, 2), 'utf8');
      if (opts.config) {
        await writeFile(
          resolve(dir, 'revex.config.json'),
          JSON.stringify(defaultConfig(opts.name), null, 2),
          'utf8',
        );
      }
      process.stdout.write(
        chalk.green(`✓ workspace initialised at ${dir}\n`) +
          `  manifest: ${manifestPath}\n` +
          `  next: run ${chalk.cyan('revex server')} from this directory\n`,
      );
    });
}

function slug(s: string): string {
  return s.toLowerCase().replace(/[^a-z0-9]+/g, '_').replace(/^_+|_+$/g, '').slice(0, 32) || 'workspace';
}

function defaultConfig(name: string): Record<string, unknown> {
  return {
    name,
    version: '1.0.0',
    server: { port: 3000, host: '127.0.0.1' },
    auth: { algorithm: 'HS256' },
    llm: { provider: 'openai', model: 'gpt-4.1', temperature: 0 },
    retrieval: {
      topK: 10,
      denseWeight: 0.6,
      sparseWeight: 0.4,
      rrfK: 60,
      fusion: 'rrf',
    },
    telemetry: { provider: 'noop' },
  };
}