# Getting Started

## Prerequisites

- Python 3.12+
- (Optional) API keys for LiteLLM, Langfuse, Marker, Chonkie,
  Qdrant, and Instructor. The framework works offline without any
  of them by falling back to in-process defaults.

## Installation

```bash
git clone <repo>
cd raghub
pip install -e ".[api,ui,dev]"
```

## Quick Start

The smallest working RAG app is fewer than 10 lines:

```python
from raghub import RAG

rag = RAG()
rag.ingest(b"Revenue grew 12% YoY in Q3 2024.")
print(rag.query("revenue").answer)
```

Or via the CLI:

```bash
# Ingest a file
python -m raghub.cli ingest ./path/to/doc.txt

# Ask a question
python -m raghub.cli query "What was the revenue growth?"

# Health
python -m raghub.cli health
```

Or with a YAML / TOML config file:

```bash
python -m raghub.cli ingest --config raghub.yaml ./documents
python -m raghub.cli query --config raghub.yaml "What is the revenue guidance?"
```

## Configuration

Configuration precedence (highest first):

1. Constructor arguments
2. Environment variables (``RAG_*``, ``JWT_SECRET``, ``NVIDIA_API_KEY``, …)
3. TOML config (``config/<profile>.toml``)
4. YAML config (``config/<profile>.yaml``)
5. Built-in defaults

See `docs/reference/configuration.md` for the full list of keys.

## API Usage

The package ships a FastAPI surface under `raghub.api`:

```bash
uvicorn raghub.api.app:app --reload
```

Use any HTTP client to call `/auth/login`, `/documents/upload`,
`/query`, etc. See `docs/reference/api.md` for the full schema.

## Streaming

```python
async for chunk in rag.astream("What was the revenue?"):
    print(chunk, end="", flush=True)
```

## Structured Output

```python
from pydantic import BaseModel

class Answer(BaseModel):
    revenue: float
    growth_pct: float

result = rag.query("Q3 revenue and growth?", response_model=Answer)
```

## Observability

Set the Langfuse credentials and the facade emits traces,
spans, latency, and token-usage automatically:

```bash
export LANGFUSE_PUBLIC_KEY=...
export LANGFUSE_SECRET_KEY=...
```

The default telemetry provider scrubs secrets from every log
message before forwarding to Langfuse.
