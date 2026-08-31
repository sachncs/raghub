/**
 * `revex config` — read or print the active configuration.
 *
 * Reads `revex.config.json` from the current directory if present
 * and prints the merged result (env vars > config file > defaults).
 */

import { readFile } from 'node:fs/promises';
import { resolve } from 'node:path';
import { existsSync } from 'node:fs';

import chalk from 'chalk';
import { Command } from 'commander';

const KNOWN_KEYS = [
  'REVEX_WORKSPACE_HOME',
  'REVEX_API_PORT',
  'REVEX_API_BASE',
  'REVEX_LLM_API_KEY',
  'REVEX_EMBEDDER_API_KEY',
  'REVEX_TELEMETRY_PROVIDER',
  'REVEX_JWT_SECRET',
];

export function registerConfigCommand(program: Command): void {
  const cmd = new Command('config').description('Read or print configuration');

  cmd
    .command('show')
    .description('Print the merged configuration')
    .action(async () => {
      const config = await loadConfig();
      process.stdout.write(JSON.stringify(config, null, 2) + '\n');
    });

  cmd
    .command('get <key>')
    .description('Print a single value')
    .action((key: string) => {
      if (!KNOWN_KEYS.includes(key)) {
        process.stderr.write(chalk.yellow(`unknown env: ${key}\n`));
      }
      const v = process.env[key] ?? '';
      process.stdout.write(`${v}\n`);
    });

  program.addCommand(cmd);
}

async function loadConfig(): Promise<Record<string, unknown>> {
  const file = resolve(process.cwd(), 'revex.config.json');
  let cfg: Record<string, unknown> = {};
  if (existsSync(file)) {
    try {
      cfg = (JSON.parse(await readFile(file, 'utf8')) as Record<string, unknown>);
    } catch {
      process.stderr.write(chalk.yellow(`failed to parse ${file}\n`));
    }
  }
  const env: Record<string, string> = {};
  for (const k of KNOWN_KEYS) {
    const v = process.env[k];
    if (v) env[k] = v;
  }
  return { file: cfg, env };
}