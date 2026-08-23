# raghub — onboarding wizard

Five-step browser flow at `apps/web/src/app/onboarding/page.tsx`.

| step | collects | stored |
| --- | --- | --- |
| 1 — workspace | name | `workspace.name` row + JWT claim |
| 2 — admin | email + password | `users` row (bcrypt hash) |
| 3 — LLM | provider, model, apiKey, baseUrl | `workspace_settings.llm` (AES-GCM encrypted) |
| 4 — passphrase | workspace passphrase + confirm | scrypt salt + AES-GCM verifier in `workspace_keycheck` |
| 5 — confirm | review | — |

After step 4 the server:

1. Generates `wsp_<rand>`.
2. Opens a fresh `workspace.db` via `openEncryptedWorkspace({path, passphrase})`.
3. Creates the first user (owner role + workspace_member row).
4. Writes the LLM config to `workspace_settings` (encrypted at rest).
5. Mints a JWT and sets it as an HTTP-only cookie.

The browser never sees the passphrase-derived key. The passphrase is
POSTed once at register and again at every login; the API layer
forwards it server-side to unlock `workspace_settings`.

## Login flow

`POST /v1/auth/login` accepts `email`, `password`, and `passphrase`:

1. Verify bcrypt-hashed password.
2. `openEncryptedWorkspace({path, passphrase})` — throws on
   wrong passphrase (mapped to 401 `invalid_passphrase`).
3. Mint JWT, set cookie, return user.

## Providers supported out of the box

- `openai` — OpenAI direct
- `minimax` — MiniMax (https://platform.minimax.io/docs/guides/models-intro)
- `litellm` — local OpenAI-compatible proxy (default `http://localhost:4000/v1`)
- `anthropic` — throws `ConfigurationError` for now
- `bedrock` — throws `ConfigurationError` for now

MiniMax and any provider that wants a raw `Authorization` header can
flip `OpenAILlm({ authorizationPrefix: 'Raw' })`. Default is Bearer.