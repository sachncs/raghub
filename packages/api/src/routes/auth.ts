/**
 * Auth routes — register, login.
 *
 * Phase 1 keeps these minimal: register creates a tenant + first
 * admin user; login mints a JWT. The full AuthService surface
 * (logout, refresh, password reset) lands in a later commit.
 */

import { Hono } from 'hono';

import {
  type BcryptHasher,
  type JwtService,
  type UserStore,
  brandId,
  AuthError,
} from '@raghub/core';

export interface AuthRouteDeps {
  readonly userStore: UserStore;
  readonly hasher: BcryptHasher;
  readonly jwt: JwtService;
}

interface RegisterInput {
  readonly email: string;
  readonly password: string;
  readonly tenantName: string;
}

interface LoginInput {
  readonly email: string;
  readonly password: string;
}

const newTenantId = () => brandId<'TenantId'>(`tnt_${Math.random().toString(36).slice(2, 14)}`);

export const authRoutes = (deps: AuthRouteDeps): Hono => {
  const app = new Hono();

  app.post('/v1/auth/register', async (c) => {
    const body = (await c.req.json().catch(() => ({}))) as Partial<RegisterInput>;
    if (!body.email || !body.password || !body.tenantName) {
      return c.json({ error: { code: 'auth_error', message: 'email, password, tenantName required' } }, 400);
    }
    if (body.password.length < 8) {
      return c.json({ error: { code: 'auth_error', message: 'password must be at least 8 chars' } }, 400);
    }
    const tenantId = newTenantId();
    await deps.userStore.upsertTenant({ id: tenantId, name: body.tenantName, plan: 'Free' });
    const passwordHash = await deps.hasher.hash(body.password);
    const user = await deps.userStore.create({
      tenantId,
      email: body.email,
      passwordHash,
      role: 'Admin',
      allowedCompanies: [],
    });
    const token = await deps.jwt.mint({ subject: user.id, tenantId: user.tenantId, isAdmin: user.isAdmin });
    return c.json({ token, user: user.toJSON() });
  });

  app.post('/v1/auth/login', async (c) => {
    const body = (await c.req.json().catch(() => ({}))) as Partial<LoginInput>;
    if (!body.email || !body.password) {
      return c.json({ error: { code: 'auth_error', message: 'email + password required' } }, 400);
    }
    const found = await deps.userStore.getByEmail(body.email);
    if (!found) {
      return c.json({ error: { code: 'auth_error', message: 'invalid credentials' } }, 401);
    }
    const ok = await deps.hasher.verify(body.password, found.passwordHash);
    if (!ok) {
      return c.json({ error: { code: 'auth_error', message: 'invalid credentials' } }, 401);
    }
    const token = await deps.jwt.mint({
      subject: found.user.id,
      tenantId: found.user.tenantId,
      isAdmin: found.user.isAdmin,
    });
    return c.json({ token, user: found.user.toJSON() });
  });

  return app;
};