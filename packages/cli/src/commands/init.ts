/**
 * `raghub init` — interactive wizard.
 *
 * Walks the user through every Phase 1 setting and writes
 * `.raghub/.env` so `raghub dev` boots without prompting. Falls
 * back to non-interactive defaults when stdin isn't a TTY (CI).
 */

import { mkdir, writeFile } from 'node:fs/promises';
import { join } from 'node:path';

import { generateKey, generateJwtSecret } from '../keys.js';
import type { Command } from '../runner.js';

const DEFAULT_EMBEDDER_MODEL = 'text-embedding-3-large';
const DEFAULT_LLM_MODEL = 'gpt-4.1';
const DEFAULT_PORT = '3000';

const prompt = async (label: string, fallback: string): Promise<string> => {
  if (!process.stdin.isTTY) return fallback;
  process.stdout.write(`${label} [${fallback}]: `);
  return new Promise<string>((resolve) => {
    let buffer = '';
    const onData = (chunk: Buffer): void => {
      buffer += chunk.toString('utf8');
      const newline = buffer.indexOf('\n');
      if (newline >= 0) {
        process.stdin.off('data', onData);
        process.stdin.off('end', onEnd);
        const answer = buffer.slice(0, newline).trim();
        resolve(answer.length > 0 ? answer : fallback);
      }
    };
    const onEnd = (): void => {
      process.stdin.off('data', onData);
      process.stdin.off('end', onEnd);
      const answer = buffer.trim();
      resolve(answer.length > 0 ? answer : fallback);
    };
    process.stdin.on('data', onData);
    process.stdin.on('end', onEnd);
  });
};

const buildEnv = (vars: Record<string, string>): string =>
  Object.entries(vars)
    .map(([k, v]) => `${k}=${v}`)
    .join('\n') + '\n';

export const initCommand: Command = {
  name: 'init',
  description: 'Initialise a new raghub project (writes .raghub/.env).',
  usage: 'raghub init [--dir <path>] [--non-interactive]',
  async run({ flags, cwd }) {
    const dirRaw = flags['dir'];
    const dir = typeof dirRaw === 'string' ? dirRaw : cwd;
    const target = join(dir, '.raghub');
    await mkdir(target, { recursive: true });

    const interactive = !flags['non-interactive'] && process.stdin.isTTY === true;

    const jwtSecret = await (interactive
      ? prompt('JWT secret (>=32 chars)', generateJwtSecret())
      : Promise.resolve(generateJwtSecret()));
    const tenantSecret = await (interactive
      ? prompt('Tenant secrets key (hex 64 chars)', generateKey(32))
      : Promise.resolve(generateKey(32)));
    const openaiKey = await (interactive
      ? prompt('OPENAI_API_KEY (leave empty to use feature-hashing)', '')
      : Promise.resolve(''));
    const embedderModel = await (interactive
      ? prompt('Embedder model', DEFAULT_EMBEDDER_MODEL)
      : Promise.resolve(DEFAULT_EMBEDDER_MODEL));
    const llmModel = await (interactive
      ? prompt('LLM model', DEFAULT_LLM_MODEL)
      : Promise.resolve(DEFAULT_LLM_MODEL));
    const port = await (interactive
      ? prompt('API port', DEFAULT_PORT)
      : Promise.resolve(DEFAULT_PORT));

    const env = {
      RAGHUB_JWT_SECRET: jwtSecret,
      RAGHUB_TENANT_SECRETS_KEY: tenantSecret,
      ...(openaiKey ? { OPENAI_API_KEY: openaiKey } : {}),
      RAGHUB_EMBEDDER_MODEL: embedderModel,
      RAGHUB_LLM_MODEL: llmModel,
      RAGHUB_VECTOR_PATH: './.raghub/raghub.db',
      RAGHUB_VECTOR_EMBEDDING_DIM: '3072',
      RAGHUB_ORCHESTRATOR_MODE: 'graph',
      RAGHUB_ORCHESTRATOR_TOP_K: '10',
      RAGHUB_HYBRID_DENSE_WEIGHT: '0.6',
      RAGHUB_HYBRID_SPARSE_WEIGHT: '0.4',
      RAGHUB_HYBRID_RRF_K: '60',
      RAGHUB_TELEMETRY_PROVIDER: 'noop',
    };

    await writeFile(join(target, '.env'), buildEnv(env), { mode: 0o600 });
    console.log(`✓ wrote ${join(target, '.env')}`);
    console.log(`✓ start with: PORT=${port} raghub dev`);
    return 0;
  },
};