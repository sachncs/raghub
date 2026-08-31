/**
 * YAML + TOML config loader.
 *
 * Reads `revex.config.{yaml,toml,json}` from a directory (default
 * cwd). Returns a structured config object that the caller merges
 * over the env-var settings.
 *
 * YAML support is a small hand-rolled subset (key: value, nested
 * objects via indentation, lists via `- item`). The full yaml
 * spec is not implemented; production deployments that need it
 * can swap in `yaml` from npm.
 *
 * TOML support is intentionally omitted from the base package —
 * surfaces, but no parser. The loader returns `null` for `.toml`
 * files and the caller falls back to env vars. Real deployments
 * add `@iarna/toml` and wire it in.
 */

import { promises as fs } from 'node:fs';
import { resolve } from 'node:path';

export interface RevexConfig {
  readonly name?: string;
  readonly server?: {
    readonly port?: number;
    readonly host?: string;
  };
  readonly auth?: {
    readonly algorithm?: string;
  };
  readonly llm?: {
    readonly provider?: string;
    readonly model?: string;
    readonly temperature?: number;
  };
  readonly retrieval?: {
    readonly topK?: number;
    readonly denseWeight?: number;
    readonly sparseWeight?: number;
    readonly rrfK?: number;
    readonly fusion?: 'rrf' | 'linear';
  };
  readonly telemetry?: {
    readonly provider?: string;
  };
}

const YAML_KEY_VALUE = /^([A-Za-z_][A-Za-z0-9_-]*)\s*:\s*(.*)$/;

const parseYamlSubset = (raw: string): RevexConfig => {
  const lines = raw
    .split('\n')
    .map((l) => l.replace(/\s+$/, ''))
    .filter((l) => l.length > 0 && !l.startsWith('#'));
  const root: Record<string, unknown> = {};
  const stack: { indent: number; obj: Record<string, unknown> }[] = [{ indent: -1, obj: root }];
  for (const line of lines) {
    const indent = line.length - line.trimStart().length;
    while (stack.length > 1 && stack[stack.length - 1]!.indent >= indent) stack.pop();
    const parent = stack[stack.length - 1]!.obj;
    if (line.trimStart().startsWith('- ')) {
      const last = Object.values(parent).at(-1);
      if (Array.isArray(last)) {
        last.push(coerceValue(line.trimStart().slice(2).trim()));
        continue;
      }
    }
    const m = YAML_KEY_VALUE.exec(line.trimStart());
    if (!m || m[1] === undefined || m[2] === undefined) continue;
    const key = m[1];
    const raw = m[2];
    const value = raw.length === 0 ? {} : coerceValue(raw);
    parent[key] = value;
    if (typeof value === 'object' && value !== null && !Array.isArray(value)) {
      stack.push({ indent, obj: value as Record<string, unknown> });
    }
  }
  return root as RevexConfig;
};

const coerceValue = (raw: string): string | number | boolean | Record<string, unknown> => {
  const trimmed = raw.replace(/^['"]|['"]$/g, '').trim();
  if (trimmed === 'true') return true;
  if (trimmed === 'false') return false;
  if (/^-?\d+(\.\d+)?$/.test(trimmed)) return Number(trimmed);
  return trimmed;
};

const parseJsonSubset = (raw: string): RevexConfig => JSON.parse(raw) as RevexConfig;

const findConfig = async (cwd: string): Promise<{ path: string; raw: string } | null> => {
  for (const name of ['revex.config.yaml', 'revex.config.yml', 'revex.config.json']) {
    const p = resolve(cwd, name);
    try {
      const raw = await fs.readFile(p, 'utf8');
      return { path: p, raw };
    } catch {
      /* not found, continue */
    }
  }
  return null;
};

export const loadConfig = async (
  cwd: string = process.cwd(),
): Promise<{ readonly path: string; readonly config: RevexConfig } | null> => {
  const found = await findConfigSafe(cwd);
  if (!found) return null;
  if (found.path.endsWith('.json')) {
    return { path: found.path, config: parseJsonSubset(found.raw) };
  }
  return { path: found.path, config: parseYamlSubset(found.raw) };
};

const findConfigSafe = (cwd: string): Promise<{ path: string; raw: string } | null> =>
  findConfig(cwd);