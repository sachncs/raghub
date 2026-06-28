# API Reference

All endpoints except `/health` require `Authorization: Bearer <token>` header.

## Authentication

### POST /auth/login

Authenticate with email and password.

```json
{"email": "alice@example.com", "password": "secret"}
```

Response:

```json
{"session_token": "...", "user_email": "alice@example.com", "allowed_companies": ["Apple"]}
```

### POST /auth/logout

Invalidate the current session token. Returns `{"status": "logged_out"}`.

## Documents

### POST /documents/upload

Upload and index a document. Multipart form with `file` and optional `company`.

Returns 202 with `document_id`, `version`, `status`, `company`, `filename`.

### GET /documents

List accessible documents for the authenticated user.

Returns `{"documents": [...]}`.

### GET /documents/{id}/status

Get document processing status.

### DELETE /documents/{id}

Delete a document. Admin only. Returns 204 (no content).

## Query

### POST /query

Answer a question using RAG. Auth via `Authorization` header.

```json
{"question": "What was the total revenue in 2023?"}
```

Response:

```json
{"answer": "...", "citations": [...], "source_chunks": [...]}
```

## Session

### GET /session/history

Get conversation history for the current session.

Returns `{"history": [...]}`.

### DELETE /session/history

Clear conversation history. Returns 204 (no content).

## Ingestion

### POST /ingest/async

Queue a document for background ingestion. Multipart form with `file` and optional `company`.

Returns `{"job_id": "..."}`.

## Health

### GET /health

Service health check. No auth required.

## Admin

Routes mounted at `/admin` prefix. Requires admin role.

## Metrics

### GET /metrics

Prometheus metrics endpoint (if enabled).
