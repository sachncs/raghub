# Onboarding

Onboarding is the 5-step wizard at `apps/web/src/app/(app)/onboarding/page.tsx`.
It calls `POST /v1/auth/register` to create a workspace, an admin user, and
session cookies in one round-trip.

## Steps

1. **Workspace** — the name, used to create the workspace + its SQLite file.
2. **Admin** — admin email and password (min 8 chars).
3. **LLM provider** — `provider`, `model`, optional `apiKey` + `baseUrl`.
4. **Passphrase** — the workspace passphrase (min 8 chars).
5. **Confirm** — submit.

## Request

`POST /v1/auth/register`

```json
{
  "email": "admin@acme.com",
  "password": "********",
  "workspaceName": "Acme",
  "passphrase": "********",
  "llm": {
    "provider": "openai",
    "model": "gpt-4.1",
    "apiKey": "sk-..."
  }
}
```

## Response

```json
{
  "token": "<jwt>",
  "user": { "id": "usr_...", "workspaceId": "wsp_...", "email": "...",
            "isAdmin": true, "role": "admin" },
  "workspace": { "id": "wsp_...", "name": "Acme" }
}
```

The browser stores `revex_session` (the JWT) and `revex_workspace_key` (the
encrypted passphrase) cookies. The API never returns the LLM API key in
plaintext.

## Validation

- All of `email`, `password`, `workspaceName`, `passphrase`, `llm` required
  (`400 auth_error`).
- `password` and `passphrase` must each be ≥ 8 characters.

## After onboarding

- `/chat` issues `POST /v1/query` (and `POST /v1/query/stream` for streaming).
- `/documents` uploads files via `POST /v1/documents` or
  `POST /v1/documents/ingest-stream` (SSE log).
- `/members` manages workspace membership.
- `/settings` configures the LLM via `GET`/`PUT /v1/settings/llm`.