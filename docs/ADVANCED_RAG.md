# Advanced RAG

This document covers everything beyond the basic "embed → retrieve → generate" flow. Every feature listed here ships in the standard install; no extra packages are required to use them, though some have optional dependencies for higher quality.

---

## Table of contents

1. [Query transforms](#query-transforms)
2. [Hybrid retrieval (dense + sparse + ColBERT)](#hybrid-retrieval-dense--sparse--colbert)
3. [Rerankers](#rerankers)
4. [Long-context second-pass rerank](#long-context-second-pass-rerank)
5. [RAPTOR recursive summaries](#raptor-recursive-summaries)
6. [GraphRAG entity / community graph](#graphrag-entity--community-graph)
7. [Agentic retrieval](#agentic-retrieval)
8. [Per-user tool preferences](#per-user-tool-preferences)
9. [Streaming events](#streaming-events)
10. [Performance & latency notes](#performance--latency-notes)
11. [Configuration reference](#configuration-reference)

---

## Query transforms

Pre-retrieval rewrites that improve recall on hard queries. All transforms share one surface: a list of `QueryVariant` objects (`text`, `kind`, `weight`) that the `RetrievalPipeline` searches in parallel and fuses by max-normalised linear combination.

### HyDE — Hypothetical Document Embeddings

```python
from raghub.retrieval import Hyde

hyde = Hyde(llm, n=1)
variants = await hyde.transform(question="What drove Q3 SaaS bookings?", history=[])
```

Generates a short LLM-authored paragraph that *would* answer the question; embeds that paragraph instead of the question. Literature default is `n=1`.

### Multi-query

```python
from raghub.retrieval import MultiQuery

mq = MultiQuery(llm, n=4)
variants = await mq.transform(question="revenue growth", history=[])
```

LLM rewrites the question into four alternative phrasings; each is embedded and searched independently. The original question is added with `weight=1.5`.

### Step-back

```python
from raghub.retrieval import StepBack

sb = StepBack(llm)
variants = await sb.transform(question="Why did operating margin expand?", history=[])
```

LLM produces the abstract higher-level question (e.g. *"What economic forces drive SaaS margin?"*) and both are searched.

### Decomposition

```python
from raghub.retrieval import Decompose

dc = Decompose(llm)
variants = await dc.transform(question="Compare Acme and Globex Q3 revenue", history=[])
```

LLM splits multi-hop questions into independent sub-questions.

### Composition

```python
from raghub.retrieval import Compose

composer = Compose([hyde, mq, sb, dc])
variants = await composer.transform(question="...", history=[])
```

The original question (weight 1.5) is always prepended. Each failing transform is silently skipped — the composer never raises.

---

## Hybrid retrieval (dense + sparse + ColBERT)

`RetrievalPipeline.retrieve_hybrid` fuses two channels (dense + BM25 keyword) with **reciprocal rank fusion** (k=60, the literature default). Switch to the legacy linear-combination path with `fusion="linear"`.

### BM25 with TF fallback

`MemoryStore.keyword_search` uses `rank_bm25.BM25Okapi` when the optional dependency is installed; otherwise it falls back to a naive TF / chunk-length score.

```python
# Add the optional dependency for production-quality keyword scoring:
#   pip install rank_bm25
```

### ColBERT late interaction

```python
# Optional dependency:
#   pip install raghub[colbert]
# (uses the ragatouille package to load the standard 70M ColBERTv2 model)
```

Set `settings.hybrid.colbert_enabled = True` and the `RetrievalPipeline.retrieve_hybrid_v2` path will add ColBERT as a third channel. When ragatouille isn't installed the channel is silently dropped (the operator gets a logged telemetry event, not a crash).

---

## Rerankers

Four implementations ship under `raghub.retrieval`:

| Provider | Class | Dependencies | When to use |
| --- | --- | --- | --- |
| `none` | `Identity` | none | default; no reranking |
| `cohere` | `Cohere` | `pip install 'raghub[rerank]'` (cohere) | API, needs key |
| `llm` | `LlmJudge` | none (uses existing LLM) | Zero-deps; slow |
| `cascade` | `Cascade` | both cohere and an LLM | Cheap then expensive |

Configure via `Settings.reranker.provider`. Each reranker reports its wall-clock to Prometheus under `raghub_rerank_latency_seconds{provider=...}`.

---

## Long-context second-pass rerank

A second LLM call reranks the top-K candidates from the first-pass retrieval. Only runs when:

* `LongContextConfig.enabled = True`, **and**
* the configured LLM's `model_name` is in `LongContextConfig.allowlist_models`.

The default allowlist is:

```python
[
    "claude-3-5-sonnet", "claude-3-7-sonnet",
    "gemini-1.5-pro", "gemini-2.0-flash",
    "command-r-plus", "gpt-4.1",
]
```

Operators add their own model names by overriding the field.

Failure modes degrade silently:

* Unknown model → pass is a no-op (the first-pass order wins).
* LLM raises → original order preserved.
* Unparseable JSON → original order preserved.
* Schema-violating scores → original order preserved.

Counter: `raghub_long_context_pass_used_total{outcome=ran|skipped|bad_json|error}`.

---

## RAPTOR recursive summaries

RAPTOR builds a tree of summaries over the chunk embeddings:

1. Cluster the leaf chunks (KMeans with the corpus-derived k).
2. Summarise each cluster with the LLM.
3. Embed the summaries.
4. Recurse up to `Raptor(depth=...)` levels.

`RetrievalPipeline.search` then matches the query against every level in one flat pass — high-level questions ("what is this corpus about?") retrieve the abstract summaries, specific questions ("what was Q3 revenue?") retrieve the leaf chunks.

```python
from raghub.knowledge.structures.raptor import Raptor

raptor = Raptor(llm=llm, embedder=embedder, depth=2, cluster_size=5)
raptor.add_chunks(chunks, vectors)  # called per ingest
hits = raptor.search("Q3 SaaS bookings", top_k=5)
```

If sklearn isn't installed, RAPTOR falls back to a windowed chunk partition — quality degrades gracefully, the index still works.

---

## GraphRAG entity / community graph

Three-step pipeline built at ingest time:

1. **Extract** entities and `(subject, predicate, object)` triples via the LLM.
2. **Partition** the graph into communities. Uses `leidenalg` when available (better communities); falls back to networkx-style connected components.
3. **Summarise** each community with the LLM.

Two query modes:

* `search_local(query, top_k)` — anchor on entities mentioned in the query, expand to their k-hop neighbourhood.
* `search_global(query, top_k)` — Map-Reduce over community summaries; useful for "what is the corpus about?" questions.
* `search(query, top_k)` — combined, deduplicated.

```python
from raghub.knowledge.structures.graphrag import GraphIndex

graph = GraphIndex(llm=llm, embedder=embedder, hop_limit=2)
graph.add_chunks(chunks, vectors)
hits = graph.search_local("Acme Q3 revenue", top_k=5)
```

Optional dependencies (install when you want higher-quality community detection):

```bash
pip install python-igraph leidenalg
```

---

## Agentic retrieval

The ReAct agent loop drives tool selection per query. The agent runs when:

* `AgentConfig.enabled = True`, **or**
* any of `WebSearchConfig.enabled`, `summary_search_enabled`, `graph_search_enabled` is on, **or**
* the per-request `tools_enabled` flag is non-empty.

### The loop

Each turn the agent:

1. Renders a ReAct system prompt with the active tool catalog.
2. Sends the prompt + running history to the LLM.
3. Parses the LLM output as either a tool call or a final answer.
4. If a tool: dispatches via `BaseTool.run`, captures the observation.
5. If final: yields an `answer_chunk` event and returns.
6. On budget breach (steps / wall-clock / tool-calls): raises `AgentBudgetExceeded`.

### Configurable budgets

```python
AgentConfig(
    max_steps=8,            # hard cap on planner steps
    max_tool_calls=10,      # hard cap on tool invocations
    max_wall_seconds=30.0,  # wall-clock ceiling
    planner_model=None,     # optional LLM override
)
```

### Built-in tools

| Tool | Purpose | Optional dependency |
| --- | --- | --- |
| `vector_search` | top-K dense retrieval | none |
| `keyword_search` | BM25 keyword | `rank_bm25` |
| `hybrid_search` | RRF fusion of the two | `rank_bm25` |
| `summary_search` | RAPTOR summaries | none |
| `graph_search` | GraphRAG local / global | none |
| `web_search` | DuckDuckGo | `pip install 'raghub[agent]'` |
| `date_today` | current UTC date | none |

### Writing your own tool

Subclass `BaseTool`, declare `name`, `description`, `json_schema`, and override `execute(context, **kwargs)`. The framework handles error wrapping, timing, and the JSON-Schema prompt — your tool just does the work.

```python
from raghub.agent.tools.base import BaseTool, ToolContext, ToolResult

class MyTool(BaseTool):
    name = "my_tool"
    description = "Does a thing."
    json_schema = {"type": "object", "properties": {}, "additionalProperties": False}

    def __init__(self, collaborator):
        self._collaborator = collaborator

    async def execute(self, context: ToolContext, **_: Any) -> ToolResult:
        result = await self._collaborator.do_thing()
        return ToolResult(content=str(result))
```

Register:

```python
from raghub.agent.tools.registry import ToolRegistry
registry = ToolRegistry()
registry.register(MyTool(my_collaborator))
```

---

## Per-user tool preferences

Three layers of override:

1. **Per-request** — `await rag.aquery(question, agent=True, tools_enabled=[...])`.
2. **Per-session** — `conversation_store.set_overrides(scoped_session_id, {"agent_enabled": True})`.
3. **Per-user** — persisted in the `user_preferences` table:
   - API: `PATCH /v1/users/me/preferences` with `{"prefs": {"tool_settings": {"agent_enabled": true}}}`.
   - CLI: `raghub config tools set --email alice@acme.com --json '{"agent_enabled": true}'`.
   - Streamlit sidebar: the "Tools" panel.

Resolution order: **request > session > user > global default**.

---

## Streaming events

The agent and the legacy query path both stream through one shape: `PlannerEvent(kind=..., step=..., payload=...)`. Five kinds:

| Kind | Payload | Meaning |
| --- | --- | --- |
| `thought` | `{thought: str}` or `{error: str}` | LLM reasoning, or a parse failure |
| `tool_call` | `{name: str, args: dict}` | Planner invoked a tool |
| `tool_result` | `{name, ok, content, error, latency_ms}` | Tool observation |
| `answer_chunk` | `{text: str}` | Token from the final answer |
| `final` | `{answer: str}` | Loop terminated |

The FastAPI surface wraps these as Server-Sent Events at `/v1/query/stream`:

```
: raghub-query-stream
event: thought
data: {"kind":"thought","step":0,"payload":{"thought":"..."}}

event: tool_call
data: {"kind":"tool_call","step":0,"payload":{"name":"vector_search","args":{"query":"..."}}}

event: tool_result
data: {"kind":"tool_result","step":0,"payload":{"name":"vector_search","ok":true,"content":"...","latency_ms":12.4}}

event: final
data: {"kind":"final","step":1,"payload":{"answer":"..."}}
```

Or fetch the full response in one shot via `/v1/agent/run`.

---

## Performance & latency notes

A few design choices worth knowing:

* **Embeddings are computed once per query.** The `RAG` facade caches the embedder instance; the pipeline never re-embeds the question.
* **RAPTOR clustering is eager.** The summary tree is rebuilt on every `add_chunks` call, but the cached embedding store means each ingest only re-touches the new chunks, not the full corpus.
* **Long-context pass is skipped on ineligible models.** Zero LLM calls when the configured model isn't in the allowlist.
* **Agent budget is enforced before every LLM call.** An over-eager tool cannot exhaust the wall-clock budget mid-loop.
* **Tool errors are isolated.** A tool raising is captured as `ToolResult(ok=False, ...)`; the agent sees the failure and adapts, never crashes.
* **Streaming is end-to-end.** Token-level output via `astream`, event-level output via `astream_agent`, SSE framing via `/v1/query/stream`.

Benchmarks: a fresh `RAG(...)` instance on the heuristic LLM answers a single-vector query in under 50 ms on a modern laptop. End-to-end with a real LLM is dominated by model latency; the framework overhead is microseconds.

---

## Configuration reference

Every advanced-RAG feature is opt-in. The full `AppSettings` reference:

```python
from raghub.config.settings import (
    AppSettings, AgentConfig, WebSearchConfig,
    RerankerConfig, LongContextConfig,
    HybridConfig, QueryTransformsConfig,
)

AppSettings(
    agent=AgentConfig(
        enabled=False,           # master agent switch
        max_steps=8,
        max_tool_calls=10,
        max_wall_seconds=30.0,
    ),
    web_search=WebSearchConfig(
        enabled=False,
        max_results=5,
        timeout_seconds=10.0,
        safe_search="moderate",
    ),
    graph_search_enabled=False,
    summary_search_enabled=False,
    reranker=RerankerConfig(
        provider="none",         # none|cohere|bge|llm|cascade
        top_k=20,
        cascade_threshold=0.05,
    ),
    long_context_pass=LongContextConfig(
        enabled=False,
        candidate_k=20,
        allowlist_models=[...],  # see "Long-context second-pass rerank"
    ),
    hybrid=HybridConfig(
        fusion="rrf",             # rrf|linear
        rrf_k=60,
        keyword_weight=0.3,
        vector_weight=0.7,
        colbert_enabled=False,
    ),
    query_transforms=QueryTransformsConfig(
        enabled=[],              # hyde|multi_query|step_back|decompose
        hyde_n=1,
        multi_query_n=4,
    ),
)
```

All fields are environment-variable-overridable (see `RAG_*` names in `raghub/config/settings.py`).