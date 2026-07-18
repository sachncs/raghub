# Runbook

This runbook covers the production stack launched with
`docker compose -f docker-compose.yml --profile production up -d`.
Every command assumes Compose v2 and the production `.env` shipped
next to the compose file.

## Health

```bash
# Container health
docker compose -f docker-compose.yml --profile production ps

# API liveness (returns {"status":"ok"})
curl -fsS http://127.0.0.1:8000/health

# UI readiness
curl -fsS http://127.0.0.1:8501/_stcore/health

# Qdrant
curl -fsS http://127.0.0.1:6333/healthz
```

The healthcheck intervals are 30s with 3 retries and a 20–30s start
period; the stack reports `healthy` only after each service has
finished initialising.

## Logs

```bash
# Tail every service, follow mode.
docker compose -f docker-compose.yml --profile production logs -f

# A single service, last 200 lines.
docker compose -f docker-compose.yml --profile production logs \
    --tail=200 api
```

Log rotation is configured at the container level (10 MiB × 5
files). Stale log files live in `/var/lib/docker/containers/*/*.log`
on the host.

## Restarting a single service

```bash
docker compose -f docker-compose.yml --profile production restart api
docker compose -f docker-compose.yml --profile production restart ui
docker compose -f docker-compose.yml --profile production restart qdrant
```

Restart is `unless-stopped`, so transient failures self-heal. A
service that fails its healthcheck 3× in a row is *not* killed
automatically; investigate `docker compose logs` for the cause.

## Common failure modes

### `JWT_SECRET` is missing or shorter than 32 bytes

Symptom: the API container exits within a few seconds and the logs
end with `RuntimeError: JWT_SECRET must be configured`.

Fix: regenerate the secret with `openssl rand -base64 48`, update
`.env`, then `docker compose up -d api`.

Note: 0.4.0 dropped the legacy ``JwtAuthenticator``; the secret
signs opaque session tokens (UUIDs minted by
:class:`SqliteSessionStore`), not JWTs. The constraint is identical
(32 random bytes) for compatibility with future signature work.

### Qdrant is unreachable from the API

Symptom: API logs contain `qdrant_client.http.exceptions.UnexpectedResponse`
or a `connection refused` against `qdrant:6333`.

Fix: confirm `docker compose ps` shows `raghub-qdrant` as `healthy`
(its healthcheck is `GET /healthz` on port 6333). If the volume is
corrupt, restore from the most recent backup (see `backup.md`).

### Disk pressure on the Qdrant volume

Symptom: `raghub-qdrant` restarts frequently and `df -h` shows the
`raghub_qdrant_data` volume above 85% full.

Fix:

```bash
# Inspect segment count and per-collection size.
curl -fsS http://127.0.0.1:6333/collections/raghub | jq

# Run an optimizer pass.
curl -fsS -X POST http://127.0.0.1:6333/collections/raghub/optimizer

# Drop the oldest snapshot.
curl -fsS -X DELETE \
    http://127.0.0.1:6333/collections/raghub/snapshots/<name>
```

### Ingestion backlog

Symptom: `/documents/upload` returns 202 but
`/documents/{id}/status` stays at `pending` for hours.

Fix: check `BackgroundIngestionService` worker count in the API
logs (`max_workers=2` by default). Bump it by overriding the
container with an explicit `BACKGROUND_INGEST_WORKERS` env var (see
`scaling.md`), or throttle the upload rate.

### Streamlit cannot reach the API

Symptom: UI renders but every query fails with `ConnectionError`.

Fix: confirm the UI service has `RAGHUB_API_URL=http://api:8000` set
in the compose environment. Inside the compose network the API is
reachable on its service name `api`, not on `127.0.0.1`.

## Hard reset

If state corruption is suspected and a restore is not possible:

```bash
docker compose -f docker-compose.yml --profile production down -v
docker volume rm raghub_qdrant_data raghub_qdrant_snapshots raghub_data
```

This deletes every named volume. Re-create the stack from a clean
backup afterwards.
