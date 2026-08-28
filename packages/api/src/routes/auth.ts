/**
 * Auth routes — register, login.
 *
 * Phase 1 keeps these minimal: register creates a workspace + first
 * admin user + LLM config (encrypted with the workspace passphrase),
 * then mints a JWT. Login re-uses an existing workspace.
 *
 * The workspace passphrase is accepted on every login to unlock
 * `workspace_settings` server-side. The browser never sees the
 * decrypted settings directly — only the proxy reads them.
 */

import { Hono } from 'hono';
import { mkdirSync } from 'node:fs';
import { dirname } from 'node:path';

import {
  type BcryptHasher,
  type JwtService,
  type Settings,
  type UserStore,
  type WorkspaceId,
  type WorkspaceRegistry,
  type WorkspaceSettingsStore,
  AuthError,
  WorkspaceMemberRole,
  SqliteAuditEventStore,
  SqliteUserStore,
  SqliteWorkspaceMemberStore,
  brandId,
  openEncryptedWorkspace,
} from '@raghub/core';

import { requireStore } from '../guards.js';

export interface AuthRouteDeps {
  readonly userStore: UserStore | null;
  readonly hasher: BcryptHasher;
  readonly jwt: JwtService;
  readonly registry: WorkspaceRegistry;
}

interface LlmInput {
  readonly provider: Settings['llm']['provider'];
  readonly model: string;
  readonly apiKey?: string;
  readonly baseUrl?: string;
}

interface RegisterInput {
  readonly email: string;
  readonly password: string;
  readonly workspaceName: string;
  readonly passphrase: string;
  readonly llm: LlmInput;
}

interface LoginInput {
  readonly email: string;
  readonly password: string;
  readonly passphrase: string;
}

const newWorkspaceId = (): WorkspaceId =>
  brandId<WorkspaceId>(`wsp_${Math.random().toString(36).slice(2, 14)}`);

const workspaceHome = (): string =>
  process.env['RAGHUB_WORKSPACE_DIR'] ??
  (process.env['RAGHUB_WORKSPACE_HOME']
    ? `${process.env['RAGHUB_WORKSPACE_HOME']}/workspaces`
    : `${process.env['HOME'] ?? '/tmp'}/.raghub/workspaces`);

const workspaceDir = (workspaceId: WorkspaceId): string => `${workspaceHome()}/${workspaceId}/workspace.db`;

const writeLlmSettings = async (
  settings: WorkspaceSettingsStore,
  llm: LlmInput,
): Promise<void> => {
  const value = {
    provider: llm.provider,
    model: llm.model,
    ...(llm.apiKey !== undefined ? { apiKey: llm.apiKey } : {}),
    ...(llm.baseUrl !== undefined ? { baseUrl: llm.baseUrl } : {}),
    temperature: 0,
  };
  await settings.set('llm', value);
};

export const authRoutes = (deps: AuthRouteDeps): Hono => {
  const app = new Hono();

  app.post('/v1/auth/register', async (c) => {
    const body = (await c.req.json().catch(() => ({}))) as Partial<RegisterInput>;
    if (
      !body.email ||
      !body.password ||
      !body.workspaceName ||
      !body.passphrase ||
      !body.llm
    ) {
      return c.json(
        {
          error: {
            code: 'auth_error',
            message: 'email, password, workspaceName, passphrase, llm required',
          },
        },
        400,
      );
    }
    if (body.password.length < 8) {
      return c.json({ error: { code: 'auth_error', message: 'password must be at least 8 chars' } }, 400);
    }
    if (body.passphrase.length < 8) {
      return c.json({ error: { code: 'auth_error', message: 'passphrase must be at least 8 chars' } }, 400);
    }

    const workspaceId = newWorkspaceId();
    const path = workspaceDir(workspaceId);
    mkdirSync(dirname(path), { recursive: true });

    const handle = await openEncryptedWorkspace({ path, passphrase: body.passphrase });
    await deps.registry.register({ workspaceId, path, encryption: 'passphrase-aes-256-gcm' });
    try {
      /* Register creates a brand-new workspace, so the boot-bound
       * userStore (which is per-first-registered-workspace) is
       * intentionally absent here. Build a fresh userStore off
       * the new handle's db. */
      const userStore = new SqliteUserStore({ db: handle.db as never });
      const passwordHash = await deps.hasher.hash(body.password);
      const user = await userStore.create({
        workspaceId,
        email: body.email,
        passwordHash,
        role: 'Admin',
        allowedCompanies: [],
      });

      const members = new SqliteWorkspaceMemberStore({ db: handle.db });
      await members.upsert({
        workspaceId,
        userId: user.id,
        role: WorkspaceMemberRole.Owner,
      });
      await members.close();

      await writeLlmSettings(handle.settings, body.llm);

      const token = await deps.jwt.mint({
        subject: user.id,
        workspaceId: user.workspaceId,
        isAdmin: user.isAdmin,
      });
      const audit = new SqliteAuditEventStore({ db: handle.db });
      await audit.record({
        kind: 'auth.register',
        workspaceId,
        actorId: user.id,
        resourceId: null,
        detail: { email: body.email },
      });
      await audit.close();
      return c.json({ token, user: user.toJSON(), workspace: { id: workspaceId, name: body.workspaceName } });
    } finally {
      handle.close();
    }
  });

  app.post('/v1/auth/login', async (c) => {
    const body = (await c.req.json().catch(() => ({}))) as Partial<LoginInput>;
    if (!body.email || !body.password || !body.passphrase) {
      return c.json({ error: { code: 'auth_error', message: 'email, password, passphrase required' } }, 400);
    }
    const userStore = requireStore('userStore', deps.userStore);
    const found = await userStore.getByEmail(body.email);
    if (!found) {
      return c.json({ error: { code: 'auth_error', message: 'invalid credentials' } }, 401);
    }
    const ok = await deps.hasher.verify(body.password, found.passwordHash);
    if (!ok) {
      return c.json({ error: { code: 'auth_error', message: 'invalid credentials' } }, 401);
    }
    const entry = await deps.registry.resolve(found.user.workspaceId);
    if (!entry) {
      return c.json({ error: { code: 'auth_error', message: 'workspace not found' } }, 404);
    }
    try {
      await openEncryptedWorkspace({ path: entry.path, passphrase: body.passphrase });
    } catch (err) {
      if (err instanceof AuthError) {
        return c.json({ error: { code: 'auth_error', message: 'invalid passphrase' } }, 401);
      }
      throw err;
    }
    const token = await deps.jwt.mint({
      subject: found.user.id,
      workspaceId: found.user.workspaceId,
      isAdmin: found.user.isAdmin,
    });
    return c.json({ token, user: found.user.toJSON() });
  });

  return app;
};