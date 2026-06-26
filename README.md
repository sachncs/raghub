# retrieval-augmented-generation

Python 3.12+ multi-user RAG application with:

- FastAPI backend.
- Streamlit reference UI.
- SQLite metadata and conversation storage.
- Alibaba Zvec for vector search.
- LlamaIndex used only for PDF loading, parsing, and chunking.
- ChatNVIDIA with `minimaxai/minimax-m3` behind a small adapter.

The code is organized as a reusable library under `app/` so the backend,
ingestion pipeline, and UI all share the same services.

## Project Layout

```text
app/
  api/
  embeddings/
  ingestion/
  llm/
  models/
  services/
  storage/
  ui/
database/
documents/
requirements.txt
README.md
```

## Setup

Create a virtual environment and install dependencies:

```bash
pip install -e .
pip install -r requirements.txt
```

The editable install makes the `app` package importable from the repository
root.

## Environment Variables

The following environment variables are supported:

- `RAG_CHUNK_SIZE`
- `RAG_OVERLAP`
- `RAG_EMBEDDING_MODEL`
- `RAG_LLM_MODEL`
- `RAG_TOP_K`
- `RAG_TEMPERATURE`
- `RAG_DATA_DIR`
- `RAG_DOCUMENTS_DIR`
- `RAG_USERS_PATH`
- `RAG_SQLITE_PATH`
- `RAG_ZVEC_DIR`
- `RAG_MAX_UPLOAD_BYTES`
- `NVIDIA_API_KEY`

The default LLM model is `minimaxai/minimax-m3`.

## Initialize SQLite

The repository includes a schema file and an initialization helper.

```bash
python -m database.init_db
```

This creates the `documents`, `chunks`, and `conversations` tables in
`database/rag.db`.

## Sample Users

`app/users.json` contains the simulated login identities:

```json
{
  "alice@email.com": { "companies": ["A"] },
  "bob@email.com": { "companies": ["B", "C"] },
  "charlie@email.com": { "companies": ["D", "E"] }
}
```

Access rules:

- Alice can only query Company A.
- Bob can query Company B and Company C.
- Charlie can query Company D and Company E.

## Sample Documents

Five sample earnings PDFs are included in `documents/`:

- `A_earnings_q4_2024.pdf`
- `B_earnings_q4_2024.pdf`
- `C_earnings_q4_2024.pdf`
- `D_earnings_q4_2024.pdf`
- `E_earnings_q4_2024.pdf`

If you want to regenerate them:

```bash
python -m app.ingestion.run_ingestion --generate-samples
```

## Ingestion

The ingestion pipeline is offline and can be executed independently.

```bash
python -m app.ingestion.run_ingestion --source documents
```

Optional sample generation:

```bash
python -m app.ingestion.run_ingestion --source documents --generate-samples
```

Pipeline:

1. PDF loader.
2. Parser.
3. Chunker.
4. Embedder.
5. SQLite metadata write.
6. Zvec vector write.

## Run FastAPI

Start the backend server:

```bash
uvicorn app.main:app --reload
```

Available endpoints:

- `POST /login`
- `POST /chat`
- `GET /history`
- `POST /logout`

### Example Requests

Login:

```bash
curl -X POST http://127.0.0.1:8000/login \
  -H "Content-Type: application/json" \
  -d '{"email":"alice@email.com"}'
```

Chat:

```bash
curl -X POST http://127.0.0.1:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"session":"<session-token>","question":"What did Company A report?"}'
```

History:

```bash
curl "http://127.0.0.1:8000/history?session=<session-token>"
```

Logout:

```bash
curl -X POST "http://127.0.0.1:8000/logout?session=<session-token>"
```

## Run Streamlit

Start the reference UI:

```bash
streamlit run app/ui/streamlit_app.py
```

The UI has two pages:

- Login
- Chat

Set the API base URL if the backend is not on localhost:

```bash
export RAG_API_BASE_URL=http://127.0.0.1:8000
```

## Sample Queries

Use these after logging in:

- Alice: "What was Company A's revenue guidance?"
- Bob: "What did Company B say about cash flow?"
- Bob: "What did Company C report for operating income?"
- Charlie: "What was Company D's subscription growth?"
- Charlie: "What did Company E say about margin improvement?"

Queries outside a user's allowed companies return no authorized results.

## Notes on Dependencies

- LlamaIndex is used only in the ingestion loader.
- Zvec stores embeddings keyed by `chunk_id`.
- SQLite stores document metadata and conversation history.
- ChatNVIDIA is isolated behind `app.llm.nvidia.NvidiaLLM`.

## Quality Checks

Run linting and type checking:

```bash
ruff check app database
mypy app
```

Run tests:

```bash
pytest
```

## Runtime Behavior

- Retrieval always filters by the user's allowed companies before search.
- Conversation history is session-scoped and stored in SQLite.
- No LangChain retrieval abstractions are used.
- No JWT, OAuth, Redis, Kafka, Celery, or Docker are required.

