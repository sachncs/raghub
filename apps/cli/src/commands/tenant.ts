/**
 * `revex tenant` — multi-tenant CRUD subcommands.
 *
 * `tenant list`   — list all registered workspaces.
 * `tenant create` — register a new workspace directory.
 * `tenant delete` — unregister a workspace (does not delete data).
 */

import { existsSync } from 'node:fs';
import { readFile, writeFile } from 'node:fs/promises';
import { resolve } from 'node:path';

import chalk from 'chalk';
import { Command } from 'commander';

const DEFAULT_HOME = process.env['REVEX_WORKSPACE_HOME'] ?? `${process.env['HOME'] ?? '/tmp'}/.revex`;
const REGISTRY_PATH = resolve(DEFAULT_HOME, 'registry.json');

interface RegistryEntry {
  workspaceId: string;
  path: string;
  encryption: string;
  registeredAt: string;
}

async function readRegistry(): Promise<RegistryEntry[]> {
  if (!existsSync(REGISTRY_PATH)) return [];
  try {
    return JSON.parse(await readFile(REGISTRY_PATH, 'utf8')) as RegistryEntry[];
  } catch {
    return [];
  }
}

async function writeRegistry(entries: RegistryEntry[]): Promise<void> {
  await writeFile(REGISTRY_PATH, JSON.stringify(entries, null, 2), 'utf8');
}

export function registerTenantCommand(program: Command): void {
  const cmd = new Command('tenant').description('Manage registered workspaces');

  cmd
    .command('list')
    .description('List all registered workspaces')
    .action(async () => {
      const entries = await readRegistry();
      for (const e of entries) {
        process.stdout.write(`${e.workspaceId}\t${e.path}\t${e.encryption}\n`);
      }
    });

  cmd
    .command('create <workspaceId>')
    .description('Register a new workspace directory')
    .requiredOption('--path <path>', 'Absolute path to the workspace.db file')
    .action(async (workspaceId: string, opts: { path: string }) => {
      const entries = await readRegistry();
      if (entries.some((e) => e.workspaceId === workspaceId)) {
        process.stderr.write(chalk.red(`workspace already registered: ${workspaceId}\n`));
        process.exit(1);
      }
      entries.push({
        workspaceId,
        path: resolve(opts.path),
        encryption: 'passphrase-aes-256-gcm',
        registeredAt: new Date().toISOString(),
      });
      await writeRegistry(entries);
      process.stdout.write(chalk.green(`✓ registered ${workspaceId}\n`));
    });

  cmd
    .command('delete <workspaceId>')
    .description('Unregister a workspace (data is NOT deleted)')
    .action(async (workspaceId: string) => {
      const entries = await readRegistry();
      const remaining = entries.filter((e) => e.workspaceId !== workspaceId);
      if (remaining.length === entries.length) {
        process.stderr.write(chalk.yellow(`no such workspace: ${workspaceId}\n`));
        process.exit(1);
      }
      await writeRegistry(remaining);
      process.stdout.write(chalk.green(`✓ unregistered ${workspaceId}\n`));
    });

  program.addCommand(cmd);
}