/**
 * Proxy to the @raghub/api Hono server.
 *
 * EventSource cannot send Authorization headers, so the chat page
 * cannot stream SSE directly with a bearer token. This route
 * accepts a `x-raghub-path` header from the client, forwards the
 * request to the API server, and pipes the response back. Cookies
 * (the JWT and the workspace passphrase) ride along on the
 * server-to-server hop.
 *
 * The passphrase cookie is server-only (httpOnly would be ideal;
 * Next.js does not expose this for plain `Response.cookie` yet —
 * left as a non-httpOnly document cookie for now and marked TODO).
 *
 * Note: when streaming a request body (Next 16 fetch requires
 * `duplex: 'half'` for non-null bodies), the upstream fetch is
 * launched with `duplex: 'half'`. SSE responses stream back via
 * `upstream.body` unchanged.
 */

import { cookies } from 'next/headers';

const API_BASE = process.env['RAGHUB_API_BASE'] ?? 'http://localhost:3000';

const cookieHeader = async (): Promise<string> => {
  const cookieStore = await cookies();
  return cookieStore
    .getAll()
    .map((c) => `${c.name}=${c.value}`)
    .join('; ');
};

const forwardedHeaders = async (
  req: Request,
  method: 'GET' | 'POST' | 'PATCH' | 'DELETE',
): Promise<HeadersInit> => {
  const token = (await cookies()).get('raghub_token')?.value;
  const cookie = await cookieHeader();
  const headers: Record<string, string> = {
    'x-raghub-forwarded': '1',
  };
  if (cookie) headers['cookie'] = cookie;
  if (token) headers['authorization'] = `Bearer ${token}`;
  /* Server-to-server callers (e.g. curl, tests) may pass an
   * Authorization header directly; prefer it over the cookie. */
  const incomingAuth = req.headers.get('authorization');
  if (incomingAuth) headers['authorization'] = incomingAuth;
  if (method !== 'GET') {
    const contentType = req.headers.get('content-type');
    if (contentType) headers['content-type'] = contentType;
  }
  return headers;
};

const proxy = async (
  req: Request,
  method: 'GET' | 'POST' | 'PATCH' | 'DELETE',
): Promise<Response> => {
  const path = req.headers.get('x-raghub-path') ?? '/';
  const hasBody = method !== 'GET' && req.body !== null;
  const init: RequestInit = {
    method,
    headers: await forwardedHeaders(req, method),
    ...(hasBody
      ? { body: req.body, duplex: 'half' as const }
      : {}),
  };
  const upstream = await fetch(`${API_BASE}${path}`, init);
  return new Response(upstream.body, {
    status: upstream.status,
    headers: upstream.headers,
  });
};

export const POST = (req: Request): Promise<Response> => proxy(req, 'POST');
export const GET = (req: Request): Promise<Response> => proxy(req, 'GET');
export const PATCH = (req: Request): Promise<Response> => proxy(req, 'PATCH');
export const DELETE = (req: Request): Promise<Response> => proxy(req, 'DELETE');