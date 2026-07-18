# Security Policy

## Reporting a Vulnerability

To report a security vulnerability, please email the maintainers
directly. Do not file a public issue. We will acknowledge receipt
within 48 hours and provide a timeline for resolution.

## Supported Versions

Security patches are released for the latest tag on the `main`
branch. The release workflow uses OIDC trusted publishing; there
is no long-lived PyPI API token in CI.

| Branch | Status |
|---|---|
| `main` | Supported — current release line |
| Older tags | Best effort |

## Hardening checklist (production)

* `JWT_SECRET` is set, unique, and ≥ 32 bytes
  (`openssl rand -base64 48`).
* `RAG_ALLOW_PASSWORDLESS=false`.
* `RAG_PROFILE=production` in `.env`.
* `CORS_ORIGINS` is the real frontend origin, not the wildcard.
* Compose stack started with
  `docker compose -f docker-compose.yml --profile production up -d`
  (read-only root, `cap_drop: [ALL]`, `no-new-privileges`,
  log rotation, named volumes, healthchecks).
* `pip-audit` is clean on every CI run; `bandit` is clean on
  every CI run.
* The OpenAPI schema is validated in CI; breaking schema changes
  are caught at PR time.

## Disclosure

We follow a coordinated disclosure model. We aim to ship a fix
within 14 days for critical issues and 60 days for non-critical
issues, with public disclosure once a fix is available.
