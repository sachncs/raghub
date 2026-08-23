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

import {
  type BcryptHasher,
  type JwtService,
  type Settings,
  type UserStore,
  type WorkspaceId,
  type WorkspaceSettingsStore,
  AuthError,
  WorkspaceMemberRole,
  SqliteWorkspaceMemberStore,
  brandId,
  openEncryptedWorkspace,
} from '@raghub/core';

export interface WorkspacePathResolver {
  /**
   * Map a workspaceId to its on-disk path. Returned path is opened
   * with `openEncryptedWorkspace({ path, passphrase })`.
   */
  resolve(workspaceId: WorkspaceId): Promise<string | null>;
}

export interface AuthRouteDeps {
  readonly userStore: UserStore;
  readonly hasher: BcryptHasher;
  readonly jwt: JwtService;
  readonly paths: WorkspacePathResolver;
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
    const path = await deps.paths.resolve(workspaceId);
    if (path === null) {
      return c.json({ error: { code: 'auth_error', message: 'cannot provision workspace storage' } }, 500);
    }

    const handle = await openEncryptedWorkspace({ path, passphrase: body.passphrase });
    try {
      const passwordHash = await deps.hasher.hash(body.password);
      const user = await deps.userStore.create({
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
    const found = await deps.userStore.getByEmail(body.email);
    if (!found) {
      return c.json({ error: { code: 'auth_error', message: 'invalid credentials' } }, 401);
    }
    const ok = await deps.hasher.verify(body.password, found.passwordHash);
    if (!ok) {
      return c.json({ error: { code: 'auth_error', message: 'invalid credentials' } }, 401);
    }
    const path = await deps.paths.resolve(found.user.workspaceId);
    if (path === null) {
      return c.json({ error: { code: 'auth_error', message: 'workspace not found' } }, 404);
    }
    try {
      await openEncryptedWorkspace({ path, passphrase: body.passphrase });
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