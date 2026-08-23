/**
 * Proxy to the @raghub/api Hono server.
 *
 * EventSource cannot send Authorization headers, so the chat page
 * cannot stream SSE directly with a bearer token. This route
 * accepts a `x-raghub-path` header from the client, forwards the
 * request to the API server, and pipes the response back. Cookies
 * (the JWT) ride along on the server-to-server hop.
 */

import { cookies } from 'next/headers';

const API_BASE = process.env['RAGHUB_API_BASE'] ?? 'http://localhost:3000';

export async function POST(req: Request): Promise<Response> {
  const path = req.headers.get('x-raghub-path') ?? '/';
  const cookieStore = await cookies();
  const token = cookieStore.get('raghub_token')?.value;
  const init: RequestInit = {
    method: 'POST',
    headers: {
      'content-type': req.headers.get('content-type') ?? 'application/json',
      ...(token ? { authorization: `Bearer ${token}` } : {}),
    },
    body: req.body,
  };
  const upstream = await fetch(`${API_BASE}${path}`, init);
  return new Response(upstream.body, {
    status: upstream.status,
    headers: upstream.headers,
  });
}

export async function GET(req: Request): Promise<Response> {
  const path = req.headers.get('x-raghub-path') ?? '/';
  const cookieStore = await cookies();
  const token = cookieStore.get('raghub_token')?.value;
  const upstream = await fetch(`${API_BASE}${path}`, {
    method: 'GET',
    headers: token ? { authorization: `Bearer ${token}` } : {},
  });
  return new Response(upstream.body, {
    status: upstream.status,
    headers: upstream.headers,
  });
}

export async function PATCH(req: Request): Promise<Response> {
  const path = req.headers.get('x-raghub-path') ?? '/';
  const cookieStore = await cookies();
  const token = cookieStore.get('raghub_token')?.value;
  const upstream = await fetch(`${API_BASE}${path}`, {
    method: 'PATCH',
    headers: {
      'content-type': req.headers.get('content-type') ?? 'application/json',
      ...(token ? { authorization: `Bearer ${token}` } : {}),
    },
    body: req.body,
  });
  return new Response(upstream.body, {
    status: upstream.status,
    headers: upstream.headers,
  });
}