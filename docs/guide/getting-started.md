# Getting Started

## Prerequisites

- Python 3.11+
- `NVIDIA_API_KEY` environment variable (for LLM and embeddings)

## Installation

```bash
git clone <repo>
cd raghub
./setup.sh                # creates .venv, installs all dependencies
source .venv/bin/activate
```

## Quick Start

Start the API server:

```bash
uvicorn raghub.api.app:app --reload
```

In another terminal, use the CLI:

```bash
# Login (creates a session token)
python -m raghub login admin@example.com secret

# Health check
python -m raghub health
```

To create a user, use the `evaluate_financebench.py` script as a reference — user creation is done programmatically via `user_store.create_user()`.

## API Usage

```bash
# Login and get a token
curl -X POST http://localhost:8000/auth/login \
  -H "Content-Type: application/json" \
  -d '{"email": "admin@example.com", "password": "secret"}'

# Upload a document
curl -X POST http://localhost:8000/documents/upload \
  -H "Authorization: Bearer <token>" \
  -F "file=@report.pdf" \
  -F "company=Acme"

# Query
curl -X POST http://localhost:8000/query \
  -H "Authorization: Bearer <token>" \
  -H "Content-Type: application/json" \
  -d '{"question": "What is the revenue?"}'
```

## Streamlit UI

```bash
streamlit run streamlit_app.py
```

## Run Tests

```bash
python -m pytest tests/ -v
```
