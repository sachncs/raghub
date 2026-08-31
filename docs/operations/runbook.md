> ⚠️ **ARCHIVED** — This document describes the **raghub 0.9.x Python
> release**, preserved for historical reference in
> [`archive/`](../../archive/). It does **not** describe the active
> **Revex** TypeScript codebase.
>
> For current documentation, see [`README.md`](../../README.md) and the
> TypeScript source under `packages/`.

# Runbook

This runbook covers the production stack launched with
`systemctl start raghub`.
Every command assumes Compose v2 and the production `.env` shipped
next to the compose file.

## Health

```bash
# Container health
systemctl status raghub

# API liveness (returns {"status":"ok"})
curl -fsS http://127.0.0.1:8000/health

service that fails its healthcheck 3× in a row is *not* killed
automatically; investigate `journalctl -u raghub` for the cause.

## Common failure modes

### `JWT_SECRET` is missing or shorter than 32 bytes

Symptom: the API container exits within a few seconds and the logs
end with `RuntimeError: JWT_SECRET must be configured`.

Fix: regenerate the secret with `openssl rand -base64 48`, update
`.env`, then `systemctl start raghub`.

Note: 0.4.0 dropped the legacy ``JwtAuthenticator``; the secret
signs opaque session tokens (UUIDs minted by
:class:`SqliteSessionStore`), not JWTs. The constraint is identical
(32 random bytes) for compatibility with future signature work.

`/documents/{id}/status` stays at `pending` for hours.

Fix: check `Batch` worker count in the API
logs (`max_workers=2` by default). Bump it by overriding the
container with an explicit `BACKGROUND_INGEST_WORKERS` env var (see
`scaling.md`), or throttle the upload rate.

### API cannot be reached from a frontend

Symptom: every request to the API fails with `ConnectionError`.

Fix: confirm the frontend has `REVEX_API_URL=http://api:8000` set
in the compose environment. Inside the compose network the API is
reachable on its service name `api`, not on `127.0.0.1`.

## Hard reset

If state corruption is suspected and a restore is not possible:

```bash
systemctl stop raghub && rm -rf /var/lib/raghub
rm -rf /var/lib/raghub
```

This deletes every named volume. Re-create the stack from a clean
backup afterwards.
