# Getting Started

The recommended way to use RAGHub is through the
[`raghub.RAG`](../../raghub/api/rag.py) facade. It is a single import,
works offline by default, and lets you replace any spec component
through the constructor or the plugin registry.

## Prerequisites

- Python 3.12+
- (Optional) An API key for any of: LiteLLM-backed LLM/embedding
  provider, Langfuse, Marker (PDF conversion), Instructor (typed
  outputs). The framework **works offline without any of them** by
  falling back to in-process deterministic defaults.

## Installation

```bash
git clone https://github.com/sachn-cs/raghub.git
cd raghub
pip install -e ".[api,ui,dev]"
```

The `api` extra installs FastAPI + uvicorn; `ui` installs Streamlit;
`dev` installs pytest, ruff, mypy, and the type stubs. The spec
libraries (Marker, Chonkie, LiteLLM, Instructor, Qdrant-client,
Langfuse, datasets) are core dependencies.

## The five-minute path

The smallest working RAG app:

```python
from raghub import RAG

rag = RAG()
rag.ingest(b"Revenue grew 12% YoY in Q3 2024.")
print(rag.query("revenue").answer)
```

That's the whole story for the simplest case. The next sections show
how to wire multi-user RBAC, conversation memory, streaming, typed
structured output, and configuration.

## CLI

The framework ships a `raghub` console script (powered by
`python -m raghub.cli`):

```bash
# Emit a starter YAML config
raghub init -o raghub.yaml

# Ingest a file or directory
raghub ingest ./documents

# Ask a question
raghub query "What was the revenue guidance?"

# Liveness probe (calls RAG.health())
raghub health

# Print the package version
raghub version
```

## Multi-user RBAC

The retrieval layer is filtered to the user's `allowed_companies`.
Admins see every company; non-admins see only chunks whose
`company` is in their allow-list; users with empty allow-lists see
**nothing**. The LLM is given only the authorised context — there
is no path by which unauthorised content can leak into the prompt.

```python
from raghub import RAG
from raghub.models import UserPrincipal

rag = RAG()

alice = UserPrincipal(user_id="alice", email="alice@acme.com",
                      allowed_companies=["Apple"])
bob   = UserPrincipal(user_id="bob",   email="bob@acme.com",
                      allowed_companies=["Microsoft"])
admin = UserPrincipal(user_id="admin", email="admin@acme.com",
                      is_admin=True)

rag.query("revenue", user=alice)   # sees Apple chunks only
rag.query("revenue", user=bob)     # sees Microsoft chunks only
rag.query("revenue", user=admin)   # sees everything
```

`user.user_id` (or `user.email`) is also used to namespace
conversation history, so two users who supply the same `session_id`
still get isolated sessions.

## Conversational memory

Pass `session_id=` to `query`, `aquery`, or `astream` and the
in-process `InMemoryConversationStore` keeps a per-session history
of the most recent turns. Follow-up questions are answered with
that history prepended to the prompt:

```python
import asyncio
from raghub import RAG

rag = RAG()

async def main() -> None:
    await rag.aingest(b"Q3 revenue was $10B. Growth was 12% YoY.")
    print(await rag.aquery("What was the revenue?", session_id="alice"))
    print(await rag.aquery("And the growth?", session_id="alice"))

asyncio.run(main())
```

Plug in your own store via the `ConversationStore` protocol:

```python
from raghub.conversation.memory import ConversationStore

class RedisConversationStore:
    def __init__(self, redis): self._redis = redis
    def append(self, session_id, turn): ...
    def load(self, session_id, limit=20): ...
    def clear(self, session_id): ...

rag = RAG()
rag.conversation_store = RedisConversationStore(...)
rag.query_pipeline.conversation_store = rag.conversation_store
```

## Streaming

`RAG.astream` is a real token stream — it routes through
`QueryPipeline.stream` → `DefaultGenerator.astream` →
`LiteLLMProvider.astream`, so the first byte reaches the caller
without waiting for the full answer:

```python
async for chunk in rag.astream("What was the revenue?"):
    print(chunk, end="", flush=True)
```

The `LiteLLMProvider` is constructed with
`stream_options={"include_usage": True}` so usage counters are
populated on every stream.

## Structured output

`RAG.query(..., response_model=MyModel)` returns a typed
`BaseModel` in `Response.structured`:

```python
from pydantic import BaseModel
from raghub import RAG

class Answer(BaseModel):
    revenue_b: float
    growth_pct: float

rag = RAG()
response = rag.query(
    "Q3 revenue and growth?",
    response_model=Answer,
)
print(response.structured)  # Answer(revenue_b=10.0, growth_pct=12.0)
```

The typed path uses `Instructor`'s
`from_provider("litellm/<model>")` (which requires
`litellm` + an LLM credential); otherwise `structured` is `None`
and the call returns the regular free-form answer.

## Configuration

`AppSettings` precedence (highest first):

1. Constructor arguments to `RAG(...)`.
2. Environment variables (`RAG_*`, `JWT_SECRET`, `NVIDIA_API_KEY`,
   `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `LANGFUSE_PUBLIC_KEY`,
   `LANGFUSE_SECRET_KEY`, `QDRANT_URL`).
3. TOML config (`config/<profile>.toml`).
4. YAML config (`config/<profile>.yaml`).
5. Built-in defaults.

Two ways to start a facade from a config file:

```python
from raghub import RAG

# From a YAML/TOML profile
rag = RAG.from_config("raghub.yaml")

# Programmatically with overrides
from raghub.config.settings import load_settings
settings = load_settings(profile="development")
settings = settings.override(chunk_size_words=400, top_k=8)
rag = RAG(settings=settings)
```

See [`reference/configuration.md`](../reference/configuration.md) for
the full list of keys and the production invariants (JWT secret
length, passwordless login, structured-log level).

## API surface

The package ships a FastAPI server bound to the legacy
`DynamicRagApplication` (the multi-tenant JWT-auth surface). The new
`RAG` facade is consumed directly or via the Streamlit UI; for an API
wrapper of the facade, instantiate it in your own FastAPI routes.

```bash
# Run the legacy FastAPI server (port 8000)
uvicorn raghub.api.app:app --reload

# Run the Streamlit UI (port 8501)
streamlit run streamlit_app.py
```

See [`reference/api.md`](../reference/api.md) for the full FastAPI
schema and [`plugins.md`](../plugins.md) for replacing components.

## Observability

Default telemetry is **Langfuse** (when `LANGFUSE_PUBLIC_KEY` and
`LANGFUSE_SECRET_KEY` are set); otherwise `NoOpTelemetry`. Every
default is wrapped in `RedactingTelemetry`, which scrubs kwargs
whose keys match `password|secret|api_key|token|jwt|authorization`
(case-insensitive, recursive into nested dicts) before forwarding
to the underlying provider.

```bash
export LANGFUSE_PUBLIC_KEY=...
export LANGFUSE_SECRET_KEY=...
```

Health summary:

```python
print(rag.health())
# {
#   "status": "ok",
#   "vector_store": "InMemoryVectorStore",
#   "embedder": "HashingEmbeddingProvider",
#   "llm": "HeuristicLLMProvider",
#   "chunker": "WordWindowChunker",
#   "converter": "PlainTextConverter",
#   "telemetry": "RedactingTelemetry",
#   "structured": "NoneType",
#   "reranker": "IdentityReranker",
# }
```

See [`operations/monitoring.md`](../operations/monitoring.md) for
the full list of emitted spans and metrics.
