# API

## Endpoints

- `POST /auth/login`
- `POST /auth/logout`
- `GET /session/history`
- `DELETE /session/history`
- `POST /documents/upload`
- `GET /documents/{document_id}/status`
- `GET /documents`
- `DELETE /documents/{document_id}`
- `POST /query`
- `GET /health`

## Contract Notes

- Authorization is enforced before retrieval.
- Chunk metadata is filtered before vector search.
- Conversation history stores turns only. Retrieved chunks are not stored in memory.

