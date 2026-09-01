# Web console

`apps/web` is a Next.js 16 App Router application (React 19, Tailwind 4,
shadcn/ui `new-york`) served on port 3001.

## Route groups

Two route groups with distinct shells:

- `(marketing)` — light theme: `/`, `/sign-in`, `/privacy`, `/terms`.
- `(app)` — dark app shell: `/chat`, `/documents`, `/members`, `/memory`,
  `/onboarding`, `/settings`.

The `(app)/layout.tsx` is a server component that fetches `/v1/me` via
`REVEX_API_BASE` (default `http://localhost:3002`) with the `revex_session`
cookie and renders the `AppShell` + `ThemeProvider`.

## API proxy

`apps/web/src/app/api/proxy/route.ts` is a catch-all that forwards every
`GET`/`POST`/`PATCH`/`PUT`/`DELETE` to the Hono API at `REVEX_API_BASE`,
using the `x-revex-path` header and forwarding `revex_session` /
`revex_workspace_key` cookies + `Authorization: Bearer`. It streams SSE
responses back to the browser.

## Pages

| Route | Purpose | Backing API |
|---|---|---|
| `/chat` | Retrieval chat with mode toggle (graph / deep_research), streaming, trace panel, rating. | `POST /v1/query`, `POST /v1/query/stream`, `POST /v1/feedback` |
| `/documents` | Document grid + upload (SSE ingest log) + share dialog. | `GET/POST /v1/documents`, `POST /v1/documents/ingest-stream`, `*/v1/documents/:id/principals` |
| `/members` | Member cards, invite, role change/remove. | `*/v1/workspaces/members` |
| `/memory` | Workspace stats + vacuum. | `GET /v1/admin/stats`, `POST /v1/admin/vacuum` |
| `/onboarding` | 5-step wizard. | `POST /v1/auth/register` |
| `/settings` | Tabs (LLM / Account / Workspace / Security). | `GET`/`PUT /v1/settings/llm` |

## Key components

- `app-shell.tsx` — collapsible sidebar, top bar, theme toggle, user menu,
  Cmd+K palette.
- `lib/use-revex-stream.ts` — SSE parser for the streaming chat + trace panel.
- Marketing components (`hero`, `feature-grid`, `live-query`,
  `platform-preview`) on `/`.
- ~40 shadcn `ui/*` primitives.