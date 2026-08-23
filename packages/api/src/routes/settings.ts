/**
 * Workspace settings routes — LLM provider/model/apiKey reconfiguration.
 *
 * The passphrase is in the cookie; the workspace pool resolves the
 * handle, settings get updated through the encrypted
 * WorkspaceSettingsStore.
 */

import { Hono } from 'hono';

import {
  type Settings,
  type WorkspaceId,
  brandId,
} from '@raghub/core';

import { getClaims, getPassphrase } from '../middleware/auth.js';
import type { WorkspacePool } from '../workspace-pool.js';

export interface SettingsRouteDeps {
  readonly pool: WorkspacePool;
}

const validProviders = ['openai', 'minimax', 'litellm', 'anthropic', 'bedrock'] as const;
type Provider = (typeof validProviders)[number];

const isProvider = (s: string): s is Provider => (validProviders as readonly string[]).includes(s);

export const settingsRoutes = (deps: SettingsRouteDeps): Hono => {
  const app = new Hono();

  app.get('/v1/settings/llm', async (c) => {
    const claims = getClaims(c);
    const passphrase = getPassphrase(c);
    if (!passphrase) {
      return c.json({ error: { code: 'auth_error', message: 'passphrase missing' } }, 401);
    }
    const handle = await deps.pool.get({
      workspaceId: brandId<WorkspaceId>(claims.workspace_id),
      userId: claims.sub,
      passphrase,
    });
    const llm = await handle.settings.get<Settings['llm']>('llm');
    const redacted = llm ? { ...llm, apiKey: llm.apiKey ? '••••••••' : undefined } : null;
    return c.json({ llm: redacted });
  });

  app.put('/v1/settings/llm', async (c) => {
    const claims = getClaims(c);
    const passphrase = getPassphrase(c);
    if (!passphrase) {
      return c.json({ error: { code: 'auth_error', message: 'passphrase missing' } }, 401);
    }
    const body = (await c.req.json().catch(() => ({}))) as {
      provider?: string;
      model?: string;
      apiKey?: string;
      baseUrl?: string;
      temperature?: number;
    };
    if (!body.provider || !isProvider(body.provider) || !body.model) {
      return c.json({ error: { code: 'raghub_error', message: 'provider + model required' } }, 400);
    }
    const handle = await deps.pool.get({
      workspaceId: brandId<WorkspaceId>(claims.workspace_id),
      userId: claims.sub,
      passphrase,
    });
    const value: Settings['llm'] = {
      provider: body.provider,
      model: body.model,
      temperature: body.temperature ?? 0,
      ...(body.apiKey !== undefined ? { apiKey: body.apiKey } : {}),
      ...(body.baseUrl !== undefined ? { baseUrl: body.baseUrl } : {}),
    };
    await handle.settings.set('llm', value);
    return c.json({ ok: true });
  });

  return app;
};