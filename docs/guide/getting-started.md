# Getting started

## Prerequisites

- **Node.js 26+** and **pnpm** 9+.
- `corepack enable pnpm` if pnpm is not installed.

## Install

```bash
git clone https://github.com/sachncs/revex.git
cd revex
pnpm install
```

Run the test suite to confirm the toolchain works:

```bash
pnpm test
```

## Run the API + web console

Start two processes:

```bash
# Terminal 1 — API
pnpm --filter @revex/api dev           # http://localhost:3000

# Terminal 2 — web console
pnpm --filter @revex/web dev           # http://localhost:3001
```

The API listens on `REVEX_API_PORT` (default `3000`). The web console proxies
`/api/*` to `REVEX_API_BASE` (default `http://localhost:3002`) via
`apps/web/src/app/api/proxy/route.ts`.

## Onboarding

Open `http://localhost:3001/onboarding` and walk the 5-step wizard:

1. **Workspace** — name your workspace.
2. **Admin** — admin email + password (min 8 chars).
3. **LLM provider** — provider + model + optional API key.
4. **Passphrase** — a workspace passphrase (min 8 chars) that encrypts the DB.
5. **Confirm** — creates the workspace via `POST /v1/auth/register`.

The wizard sets two cookies: `revex_session` (JWT) and `revex_workspace_key`
(the encrypted passphrase). The LLM API key is never sent to the browser in
plaintext — it is encrypted with the workspace passphrase and stored at rest.

## Run the CLI

```bash
pnpm --filter @revex/cli dev -- init -n myworkspace
pnpm --filter @revex/cli dev -- server
pnpm --filter @revex/cli dev -- ingest ./doc.pdf
pnpm --filter @revex/cli dev -- query "what does the doc say?"
```

## What's next

- [Workspace model](workspace.md) — encryption and storage.
- [First query via the API](api.md).
- [Onboarding flow](onboarding.md).