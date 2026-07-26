# Atomic Changes: Agentic + Hybrid + Knowledge-Structured RAGHub

Each change is a single commit-shaped unit: a file action (new/edit/delete), a
code sketch, the deps it pulls in, and the test that proves it. Order is
dependency-driven; within a phase, top-to-bottom is the merge order.

Locked choices (from the design session):
- **RAPTOR + GraphRAG** (both, toggle at ingest).
- **Agent merged into `aquery`/`astream`** — `agent=False` and no tools /
  transforms / reranker keeps the fast path byte-equivalent to today's
  `QueryPipeline.run`.
- **Ship all three rerankers** (Cohere / BGE / LLM-judge); default
  `IdentityReranker`.
- **Per-user config in a separate `user_preferences` table**.

Conventions used below:
- `[N]` = new file, `[E]` = edit, `[D]` = delete, `[M]` = migration.
- `★` marks a change that ships first inside its phase (merge blocker).
- Every change ends with a `Test:` line.

---

## Phase 1 — Foundations (plumbing)

### 1.1 ★[N] `raghub/agent/__init__.py`

Empty package marker. Re-exports `Tool`, `ToolResult`, `ToolRegistry`,
`Agent`, `PlannerEvent`.

```python
"""Agent loop, tool registry, planner events."""
```

### 1.2 ★[N] `raghub/agent/tools/base.py`

```python
class ToolResult(BaseModel):
    ok: bool = True
    content: str = ""
    data: dict[str, Any] = Field(default_factory=dict)
    error: str | None = None
    latency_ms: float = 0.0
    source_url: str | None = None

class Tool(Protocol):
    name: str
    description: str
    json_schema: dict[str, Any]
    async def run(self, args: dict[str, Any]) -> ToolResult: ...
```

Test: `tests/agent/test_tool_protocol.py::test_protocol_shape`.

### 1.3 ★[N] `raghub/agent/tools/registry.py`

```python
class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}
    def register(self, tool: Tool) -> None: ...
    def get(self, name: str) -> Tool: ...
    def schemas(self) -> list[dict[str, Any]]: ...
    def names(self) -> list[str]: ...
```

Test: `tests/agent/test_registry.py`.

### 1.4 ★[N] `raghub/agent/events.py`

```python
class PlannerEvent(BaseModel):
    kind: Literal["thought", "tool_call", "tool_result", "answer_chunk", "final"]
    step: int = 0
    payload: dict[str, Any] = Field(default_factory=dict)
```

Test: `tests/agent/test_events.py`.

### 1.5 [E] `raghub/exceptions/__init__.py`

Add:

```python
class ToolError(RagHubError): ...
class AgentBudgetExceeded(RagHubError): ...
class WebSearchError(RagHubError): ...
class RerankerError(RagHubError): ...
class GraphUnavailableError(RagHubError): ...
class TransformError(RagHubError): ...
```

Test: `tests/test_exceptions.py` gains cases for each.

### 1.6 [E] `raghub/config/settings.py`

Add nested blocks:

```python
class AgentConfig(BaseModel):
    enabled: bool = False
    max_steps: int = 8
    max_tool_calls: int = 10
    max_wall_seconds: float = 30.0
    planner_model: str | None = None
    enable_streaming: bool = True

class WebSearchConfig(BaseModel):
    enabled: bool = False
    max_results: int = 5
    timeout_seconds: float = 10.0
    safe_search: str = "moderate"

class RerankerConfig(BaseModel):
    provider: Literal["none", "cohere", "bge", "llm", "cascade"] = "none"
    top_k: int = 20
    cascade_threshold: float = 0.05

class LongContextConfig(BaseModel):
    enabled: bool = False
    candidate_k: int = 20
    allowlist_models: list[str] = [
        "claude-3-5-sonnet", "claude-3-7-sonnet",
        "gemini-1.5-pro", "gemini-2.0-flash",
        "command-r-plus", "gpt-4.1",
    ]

class HybridConfig(BaseModel):
    fusion: Literal["rrf", "linear"] = "rrf"
    rrf_k: int = 60
    keyword_weight: float = 0.3
    vector_weight: float = 0.7
    colbert_enabled: bool = False

class QueryTransformsConfig(BaseModel):
    enabled: list[Literal["hyde", "multi_query", "step_back", "decompose"]] = []
    hyde_n: int = 1
    multi_query_n: int = 4
```

Extend `AppSettings`:

```python
agent: AgentConfig = Field(default_factory=AgentConfig)
web_search: WebSearchConfig = Field(default_factory=WebSearchConfig)
graph_search_enabled: bool = False
summary_search_enabled: bool = False
reranker: RerankerConfig = Field(default_factory=RerankerConfig)
long_context_pass: LongContextConfig = Field(default_factory=LongContextConfig)
hybrid: HybridConfig = Field(default_factory=HybridConfig)
query_transforms: QueryTransformsConfig = Field(default_factory=QueryTransformsConfig)
```

Add env-var plumbing in `load_settings`:
`RAG_AGENT_ENABLED`, `RAG_WEB_ENABLED`, `RAG_RERANKER_PROVIDER`,
`RAG_LONG_CONTEXT_ENABLED`, `RAG_HYBRID_FUSION`, `RAG_TRANSFORMS_ENABLED`.

Test: `tests/config/test_settings_advanced.py`.

### 1.7 [E] `raghub/models/api.py`

Add fields to `QueryRequest` (all optional, default to None/unset):

```python
class QueryRequest(BaseModel):
    question: str = Field(min_length=1)
    tools_enabled: list[str] | None = None
    agent: bool | None = None
    web: bool | None = None
    graph: bool | None = None
    summaries: bool | None = None
    reranker: str | None = None      # "none|cohere|bge|llm|cascade"
    long_context_pass: bool | None = None
    query_transforms: list[str] | None = None
    max_steps: int | None = None
    top_k: int | None = None
```

Extend `QueryResponse` with:

```python
    planner_trace: list[dict] | None = None
    tools_invoked: list[str] = []
    transforms_applied: list[str] = []
```

Test: `tests/test_api_schemas.py::test_query_request_new_fields`.

### 1.8 ★[N] `raghub/agent/resolver.py`

```python
@dataclass(frozen=True)
class ResolvedConfig:
    agent_enabled: bool
    tools_enabled: set[str]
    reranker: str
    long_context_pass: bool
    query_transforms: list[str]
    max_steps: int

def resolve(
    *, request_overrides: dict[str, Any],
    session_overrides: dict[str, Any] | None,
    user_prefs: dict[str, Any] | None,
    settings: AppSettings,
) -> ResolvedConfig:
    """Order: request > session > user > global default."""
```

Test: `tests/agent/test_resolver.py::test_precedence_request_over_user`.

### 1.9 ★[M] `raghub/auth/migrations/001_user_preferences.sql`

```sql
CREATE TABLE IF NOT EXISTS user_preferences (
    user_id TEXT NOT NULL,
    key TEXT NOT NULL,
    value JSON NOT NULL,
    updated_at TEXT NOT NULL DEFAULT (datetime('now')),
    PRIMARY KEY (user_id, key),
    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
);
CREATE INDEX IF NOT EXISTS idx_user_prefs_user ON user_preferences(user_id);
```

Hook: `auth_service.ensure_schema()` runs this on startup if missing.

Test: `tests/auth/test_user_preferences_migration.py`.

### 1.10 [E] `raghub/auth/user_store.py`

Add:

```python
def get_prefs(self, user_id: str) -> dict[str, Any]: ...
def set_pref(self, user_id: str, key: str, value: Any) -> None: ...
def delete_pref(self, user_id: str, key: str) -> None: ...
```

Backed by `user_preferences` table (1.9).

Test: `tests/auth/test_user_prefs_crud.py`.

### 1.11 [E] `raghub/auth/rbac.py` (or wherever UserPrincipal is defined)

Extend `UserPrincipal` (no DB change — purely additive Pydantic field):

```python
class UserPrincipal(BaseModel):
    ...existing...
    tool_settings: dict[str, Any] = Field(default_factory=dict)
```

`auth_service.load_user` populates it by merging `user_prefs["tool_settings"]`.

Test: `tests/auth/test_user_principal_tool_settings.py`.

### 1.12 [E] `raghub/conversation/memory.py`

Extend the in-memory conversation store to carry a session-level override dict:

```python
class ConversationSession(BaseModel):
    turns: list[ConversationTurn] = []
    overrides: dict[str, Any] = {}    # tool/agent toggles for this session
```

Persistent equivalent gets the same column.

Test: `tests/conversation/test_session_overrides.py`.

---

## Phase 2 — Query transforms (pre-retrieval)

### 2.1 ★[N] `raghub/retrieval/transforms/__init__.py` + `base.py`

```python
class QueryVariant(BaseModel):
    text: str
    kind: Literal["original", "hyde", "multi_query", "step_back", "sub"]
    weight: float = 1.0

class QueryTransformer(Protocol):
    name: str
    async def transform(self, *, question: str, history: list) -> list[QueryVariant]: ...
```

Test: `tests/retrieval/transforms/test_base.py`.

### 2.2 [N] `raghub/retrieval/transforms/hyde.py`

```python
class HydeTransformer:
    name = "hyde"
    def __init__(self, llm): self.llm = llm
    async def transform(self, *, question, history):
        hyp = await self.llm.complete(
            f"Write a short paragraph that would answer: {question}\n"
            "Only the paragraph, no preamble."
        )
        return [QueryVariant(text=hyp, kind="hyde", weight=1.0)]
```

Test: `tests/retrieval/transforms/test_hyde.py` (uses `HeuristicLLMProvider`).

### 2.3 [N] `raghub/retrieval/transforms/multi_query.py`

LLM produces N rephrasings as a JSON array. Each becomes a variant. Original
question is also kept (weight 1.5).

Test: `tests/retrieval/transforms/test_multi_query.py`.

### 2.4 [N] `raghub/retrieval/transforms/step_back.py`

LLM produces the abstract higher-level question; both abstract (weight 1.2)
and original (weight 1.0) returned.

Test: `tests/retrieval/transforms/test_step_back.py`.

### 2.5 [N] `raghub/retrieval/transforms/decompose.py`

LLM produces sub-questions as JSON; each is a `kind="sub"` variant.

Test: `tests/retrieval/transforms/test_decompose.py`.

### 2.6 [N] `raghub/retrieval/transforms/compose.py`

```python
class ComposeTransformer:
    def __init__(self, transformers: list[QueryTransformer]): ...
    async def transform(self, *, question, history):
        out: list[QueryVariant] = [QueryVariant(text=question, kind="original", weight=1.5)]
        for t in self.transformers:
            out.extend(await t.transform(question=question, history=history))
        return out
```

Test: `tests/retrieval/transforms/test_compose.py`.

### 2.7 [E] `raghub/api/defaults.py`

Add `default_transforms(settings) -> ComposeTransformer` — builds the
configured chain.

Test: `tests/api/test_defaults.py::test_default_transforms`.

### 2.8 [E] `raghub/api/rag.py`

`RAG.__init__` gains `transformer: QueryTransformer | None = None`,
defaulting to `default_transforms(self.settings)`.

Test: `tests/test_rag_facade.py::test_rag_accepts_transformer`.

---

## Phase 3 — Hybrid retrieval v2

### 3.1 [E] `raghub/vectorstore/memory.py`

Replace naive TF in `keyword_search` with `rank_bm25.BM25Okapi` when
`rank_bm25` is importable. Fall back to TF when not installed.

```python
try:
    from rank_bm25 import BM25Okapi
except ImportError:
    BM25Okapi = None

def keyword_search(self, query, top_k):
    if BM25Okapi is None: return self._tf_keyword_search(query, top_k)
    ...
```

Test: `tests/vectorstore/test_memory_bm25.py` (skipped if dep missing).

### 3.2 ★[N] `raghub/retrieval/fusion.py`

```python
def rrf(rankings: list[list[str]], k: int = 60) -> list[tuple[str, float]]:
    scores: dict[str, float] = {}
    for ranking in rankings:
        for rank, item in enumerate(ranking, start=1):
            scores[item] = scores.get(item, 0.0) + 1.0 / (k + rank)
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)

def linear_combine(channel_scores: dict[str, dict[str, float]],
                   weights: dict[str, float]) -> list[tuple[str, float]]: ...
```

Test: `tests/retrieval/test_fusion.py`.

### 3.3 [E] `raghub/retrieval/pipeline.py`

`retrieve_hybrid` selects fusion by `settings.hybrid.fusion`:

```python
if self.settings.hybrid.fusion == "rrf":
    return self._rrf(query, vector_results, k=self.settings.hybrid.rrf_k)
return self._linear(query, vector_results, ...)
```

Default switches to RRF. Linear kept verbatim under the old method name for
back-compat callers.

Test: `tests/retrieval/test_pipeline_rrf.py` + regression of linear path.

### 3.4 ★[N] `raghub/retrieval/colbert.py`

```python
class ColbertLateInteraction:
    def __init__(self, settings): self._enabled = settings.hybrid.colbert_enabled
    def is_available(self) -> bool:
        try:
            import ragatouille  # noqa
            return self._enabled
        except ImportError:
            return False
    def score(self, query: str, doc_texts: list[str]) -> list[float]:
        # lazy import ragatouille; raise GraphUnavailableError if absent
        ...
```

Test: `tests/retrieval/test_colbert_optional.py` (skip if not installed).

### 3.5 [E] `raghub/retrieval/pipeline.py`

Add `retrieve_hybrid_v2` that runs dense + sparse + colbert and fuses via RRF
when colbert is available. Wired through `RetrievalPipeline.retrieve` only
when `settings.hybrid.colbert_enabled` is true (so the v1 path stays
byte-stable for the fast-path regression test).

Test: `tests/retrieval/test_pipeline_colbert.py`.

### 3.6 [E] `raghub/retrieval/search.py`

No semantic change; just expose `ColbertLateInteraction` for plumbing.

---

## Phase 4 — Rerankers

### 4.1 ★[N] `raghub/retrieval/rerankers/__init__.py`

Re-exports all rerankers.

### 4.2 [N] `raghub/retrieval/rerankers/cohere.py`

```python
class CohereReranker:
    def __init__(self, api_key: SecretStr, model: str = "rerank-english-v3.0",
                 top_k: int = 20): ...
    def rerank(self, *, question, hits):
        try:
            import cohere
        except ImportError as e:
            raise RerankerError("cohere not installed") from e
        ...
```

Test: `tests/retrieval/rerankers/test_cohere.py` (monkeypatched HTTP).

### 4.3 [N] `raghub/retrieval/rerankers/bge.py`

```python
class BgeReranker:
    def __init__(self, model: str = "BAAI/bge-reranker-v2-m3",
                 top_k: int = 20): ...
    def rerank(self, *, question, hits):
        try:
            from sentence_transformers import CrossEncoder
        except ImportError as e:
            raise RerankerError("sentence-transformers not installed") from e
        ...
```

Test: `tests/retrieval/rerankers/test_bge.py` (mocked model).

### 4.4 [N] `raghub/retrieval/rerankers/llm.py`

Listwise via existing litellm: ask the LLM to return `[{"id":..., "score":...}]`
JSON. Pairwise fallback for `len(hits) > 20`.

Test: `tests/retrieval/rerankers/test_llm_judge.py`.

### 4.5 [N] `raghub/retrieval/rerankers/cascade.py`

```python
class CascadeReranker:
    def __init__(self, cheap: Reranker, expensive: Reranker,
                 spread_threshold: float = 0.05): ...
    def rerank(self, *, question, hits):
        ranked = cheap.rerank(question=question, hits=hits)
        if spread(ranked) < self.spread_threshold:
            ranked = expensive.rerank(question=question, hits=ranked)
        return ranked
```

Test: `tests/retrieval/rerankers/test_cascade.py`.

### 4.6 ★[N] `raghub/retrieval/rerankers/factory.py`

```python
def build_reranker(settings: AppSettings) -> Reranker:
    p = settings.reranker.provider
    if p == "none":   return IdentityReranker()
    if p == "cohere": return CohereReranker(...)
    if p == "bge":    return BgeReranker(...)
    if p == "llm":    return LLMReranker(llm=..., top_k=settings.reranker.top_k)
    if p == "cascade":return CascadeReranker(BgeReranker(...), CohereReranker(...))
    raise RerankerError(p)
```

Test: `tests/retrieval/rerankers/test_factory.py`.

### 4.7 [E] `raghub/api/rag.py`

`RAG.__init__` gains `reranker: Any = None`. Resolution:

```python
self.reranker = reranker or build_reranker(self.settings)
```

Test: `tests/test_rag_facade.py::test_reranker_factory`.

### 4.8 [E] `raghub/observability/metrics.py`

Add counter + histogram:

```python
rerank_latency_seconds = Histogram("raghub_rerank_latency_seconds",
                                    ["provider"])
```

Wire in each reranker's `rerank` method.

Test: `tests/observability/test_rerank_metrics.py`.

---

## Phase 5 — Long-context second pass

### 5.1 ★[N] `raghub/retrieval/long_context.py`

```python
class LongContextRerankPass:
    def __init__(self, llm, settings: AppSettings): ...
    def is_eligible(self) -> bool:
        return (self.settings.long_context_pass.enabled
                and self.llm.model_name in self.settings.long_context_pass.allowlist_models)
    async def rerank(self, *, question, hits) -> list[RetrievalHit]:
        if not self.is_eligible():
            return hits  # graceful no-op
        prompt = self._build_prompt(question, hits)
        refined = await self.llm.astructured(prompt, response_model=RankedList)
        return self._reorder(hits, refined)
```

Test: `tests/retrieval/test_long_context_pass.py` (eligible + ineligible).

### 5.2 [N] `raghub/models/long_context.py`

```python
class RankedItem(BaseModel):
    chunk_id: str
    score: float
    rationale: str

class RankedList(BaseModel):
    items: list[RankedItem]
```

Test: `tests/models/test_long_context_models.py`.

### 5.3 [E] `raghub/pipelines/rag.py`

`QueryPipeline.run` gains a step after rerank:

```python
if self.long_context_pass is not None:
    hits = await self.long_context_pass.rerank(question=question, hits=hits)
```

Constructor param `long_context_pass: LongContextRerankPass | None = None`.

Test: `tests/pipelines/test_query_pipeline_long_context.py`.

### 5.4 [E] `raghub/observability/metrics.py`

```python
long_context_pass_used_total = Counter("raghub_long_context_pass_used_total",
                                       ["outcome"])  # "ran|skipped"
```

Test: `tests/observability/test_long_context_metrics.py`.

---

## Phase 6 — Knowledge structures (RAPTOR + GraphRAG)

### 6.1 [N] `raghub/knowledge/structures/__init__.py` + `base.py`

```python
class KnowledgeIndex(Protocol):
    def add_chunks(self, chunks: list[Chunk], vectors: list[list[float]]) -> None: ...
    def search(self, query: str, top_k: int) -> list[RetrievalHit]: ...
```

Test: `tests/knowledge/structures/test_base.py`.

### 6.2 ★[N] `raghub/knowledge/structures/raptor.py`

```python
class RaptorIndex:
    def __init__(self, *, llm, embedder, depth: int = 2,
                 cluster_per_level: int = 8): ...
    def build(self, chunks, vectors): ...   # synchronous (ingest-time)
    def search(self, query, top_k): ...
```

Clustering: `sklearn.cluster.KMeans` (already pulled by litellm/sentence-transformers
transitively; if missing, lazy import and raise `GraphUnavailableError`).

Summarisation: existing `llm` interface.

Test: `tests/knowledge/structures/test_raptor.py`.

### 6.3 [N] `raghub/knowledge/structures/graphrag.py`

```python
class GraphRagIndex:
    def __init__(self, *, llm, embedder): ...
    def build(self, chunks, vectors):
        triples = self._extract_triples(chunks)         # Instructor + Pydantic
        self._graph = self._build_graph(triples)         # networkx
        self._communities = self._leiden_partition()      # igraph/leidenalg
        self._summaries = self._summarise_communities()  # Map-Reduce
    def search_local(self, query, top_k): ...           # entity expansion
    def search_global(self, query, top_k): ...          # Map-Reduce over summaries
```

Test: `tests/knowledge/structures/test_graphrag.py` (skipped if deps missing).

### 6.4 [N] `raghub/agent/tools/summary_search.py` (Phase 7 wiring)

Wraps `RaptorIndex.search` as a tool.

Test: `tests/agent/test_tools_summary_search.py`.

### 6.5 [N] `raghub/agent/tools/graph_search.py` (Phase 7 wiring)

Wraps `GraphRagIndex.search_local` (and optional `search_global`).

Test: `tests/agent/test_tools_graph_search.py`.

### 6.6 [E] `raghub/pipelines/rag.py`

`IngestPipeline.__init__` gains `raptor: KnowledgeIndex | None = None`,
`graph: KnowledgeIndex | None = None`. After `vector_store.upsert`:

```python
if self.raptor: self.raptor.add_chunks(chunks, vectors)
if self.graph:  self.graph.add_chunks(chunks, vectors)
```

Test: `tests/pipelines/test_ingest_pipeline_knowledge_structures.py`.

### 6.7 [E] `raghub/api/rag.py`

`RAG.__init__` gains `raptor`, `graph` kwargs. Manifest hooks: deleting a
document purges its entries from both indexes.

Test: `tests/test_rag_facade_knowledge_structures.py`.

### 6.8 [E] `raghub/manifest.py` (or `raghub/knowledge/manifest.py`)

Track per-document id presence in raptor/graph indexes; delete method
walks both.

Test: `tests/knowledge/test_manifest_invalidation.py`.

---

## Phase 7 — Agentic planner

### 7.1 ★[N] `raghub/agent/prompts.py`

```python
REACT_SYSTEM = """You are a planner with tools. Each turn respond with JSON:
{"thought": str, "action": {"name": str, "args": dict} | null,
 "final_answer": str | null}"""
```

Test: `tests/agent/test_prompts.py`.

### 7.2 ★[N] `raghub/agent/agent.py`

```python
class Agent:
    def __init__(self, *, llm, tool_registry: ToolRegistry,
                 settings: AppSettings,
                 telemetry: TelemetryProvider | None = None): ...
    async def run(self, *, question: str, history: list,
                  tools_enabled: set[str]) -> AgentTrace: ...
    async def astream(self, *, question: str, history: list,
                      tools_enabled: set[str]) -> AsyncIterator[PlannerEvent]: ...
    def _within_budget(self, started: float, steps: int) -> bool: ...
```

Loop body:
1. LLM call with system + history + tool schemas.
2. Parse `PlannerAction | PlannerFinal`.
3. If final → return.
4. Else execute tool (within `tools_enabled`), append observation, repeat.
5. On budget breach → raise `AgentBudgetExceeded` with partial trace.

Test: `tests/agent/test_agent_loop.py`, `test_agent_budget.py`.

### 7.3 [N] `raghub/agent/tools/vector_search.py`

```python
class VectorSearchTool:
    name = "vector_search"
    json_schema = {"type": "object", "properties": {"query": {"type": "string"},
                      "top_k": {"type": "integer", "default": 10}}, "required": ["query"]}
    def __init__(self, retrieval_pipeline: RetrievalPipeline): ...
    async def run(self, args): ...
```

Test: `tests/agent/tools/test_vector_search.py`.

### 7.4 [N] `raghub/agent/tools/keyword_search.py`

Wraps `RetrievalPipeline.retrieve_keyword`.

Test: `tests/agent/tools/test_keyword_search.py`.

### 7.5 [N] `raghub/agent/tools/hybrid_search.py`

Wraps `RetrievalPipeline.retrieve_hybrid` (RRF or linear).

Test: `tests/agent/tools/test_hybrid_search.py`.

### 7.6 ★[N] `raghub/agent/tools/web_search.py`

```python
class WebSearchTool:
    name = "web_search"
    json_schema = {"type": "object", "properties": {"query": {"type": "string"},
                      "max_results": {"type": "integer", "default": 5}}, ...}
    async def run(self, args):
        try:
            from duckduckgo_search import DDGS
        except ImportError as e:
            raise WebSearchError("duckduckgo-search not installed") from e
        with DDGS() as ddgs:
            results = list(ddgs.text(args["query"], max_results=args.get("max_results", 5)))
        return ToolResult(content="\n\n".join(r["body"] for r in results),
                          data={"results": results})
```

Test: `tests/agent/tools/test_web_search.py` (monkeypatched DDGS).

### 7.7 [N] `raghub/agent/tools/date_today.py`

No deps. Returns current UTC ISO date.

Test: `tests/agent/tools/test_date_today.py`.

### 7.8 ★[N] `raghub/agent/builder.py`

```python
def build_tool_registry(*, retrieval_pipeline, settings,
                        raptor=None, graph=None) -> ToolRegistry:
    reg = ToolRegistry()
    reg.register(VectorSearchTool(retrieval_pipeline))
    reg.register(KeywordSearchTool(retrieval_pipeline))
    reg.register(HybridSearchTool(retrieval_pipeline))
    if settings.web_search.enabled:    reg.register(WebSearchTool())
    if settings.summary_search_enabled and raptor is not None:
        reg.register(SummarySearchTool(raptor))
    if settings.graph_search_enabled and graph is not None:
        reg.register(GraphSearchTool(graph))
    reg.register(DateTodayTool())
    return reg
```

Test: `tests/agent/test_builder.py`.

### 7.9 ★[N] `raghub/pipelines/agentic.py`

```python
class AgenticQueryPipeline:
    name = "query_agent"
    def __init__(self, *, agent: Agent, embedder, vector_store, generator,
                 reranker=None, structured=None, telemetry=None,
                 conversation_store=None, cache=None): ...
    async def run(self, context, **inputs) -> PipelineResult: ...
    async def stream(self, context, **inputs) -> AsyncIterator[str | PlannerEvent]: ...
```

The agent's `final_answer` is fed to the generator as the synthesised
response (with citation enrichment from observed tool results).

Test: `tests/pipelines/test_agentic_pipeline.py`.

### 7.10 [E] `raghub/pipelines/rag.py`

`QueryPipeline` becomes the **fast-path only**. `run` checks the resolved
config; if `agent_enabled` or any tool is requested, it forwards to a
configured `AgenticQueryPipeline`:

```python
if self.agentic_pipeline is not None and (
    resolved.agent_enabled or resolved.tools_enabled
):
    return await self.agentic_pipeline.run(context, **inputs)
# ... existing fast path unchanged ...
```

Constructor param `agentic_pipeline: AgenticQueryPipeline | None = None`.

Test: `tests/pipelines/test_query_pipeline_dispatch.py`.

### 7.11 [E] `raghub/api/rag.py`

`RAG.__init__` gains `agent: Agent | None = None`,
`tool_registry: ToolRegistry | None = None`. Both default via builders.

Test: `tests/test_rag_facade.py::test_agent_wiring`.

### 7.12 [E] `raghub/observability/metrics.py`

```python
planner_steps_total = Counter("raghub_planner_steps_total")
tool_calls_total = Counter("raghub_tool_calls_total", ["tool"])
web_search_hits_total = Counter("raghub_web_search_hits_total")
```

Test: `tests/observability/test_agent_metrics.py`.

---

## Phase 8 — User-configurable toggles (UI + API)

### 8.1 [N] `raghub/api/preferences.py`

```python
router = APIRouter()

@router.get("/users/me/preferences")
async def get_preferences(...): return auth_service.get_prefs(user.id)

@router.patch("/users/me/preferences")
async def patch_preferences(payload: dict, ...):
    for k, v in payload.items(): auth_service.set_pref(user.id, k, v)
    return auth_service.get_prefs(user.id)
```

Test: `tests/api/test_preferences_endpoints.py`.

### 8.2 [E] `raghub/api/app.py`

Mount the new router at `/v1`.

### 8.3 [E] `raghub/services/application.py`

`DynamicRagApplication.query` accepts the new `QueryRequest` fields and
forwards them into the resolved config.

### 8.4 [E] `raghub/cli/main.py`

Add subcommands:

```python
@cli.command("config tools list")
@cli.command("config tools set")
@cli.command("config tools unset")
```

Test: `tests/cli/test_config_tools.py`.

### 8.5 [E] `streamlit_app.py`

Add a sidebar panel rendered inside `_render_sidebar`:

```python
def _render_tools_panel(state): ...
    agent = st.toggle("Agent mode", value=...)
    web   = st.toggle("Web search", value=...)
    graph = st.toggle("Graph search", value=...)
    reranker = st.selectbox("Reranker", ["none","bge","cohere","llm","cascade"])
    lcp   = st.toggle("Long-context rerank", value=...)
    transforms = st.multiselect("Query transforms",
                                ["hyde","multi_query","step_back","decompose"])
    max_steps = st.slider("Max planner steps", 1, 16, 8)
    if st.button("Save"):
        httpx.patch(f"{API}/v1/users/me/preferences",
                    json={"tool_settings": {...}}, headers=auth)
        st.success("Saved")
```

Test: `tests/streamlit/test_tools_panel.py` (snapshot of generated widgets).

### 8.6 [E] `raghub/conversation/memory.py` + `sliding_window.py`

Both stores carry the `overrides` dict (1.12). UI writes via
`conversation_store.set_overrides(session_id, dict)`.

Test: `tests/conversation/test_session_overrides_crud.py`.

### 8.7 [E] `raghub/api/rag.py`

`RAG.aquery` / `RAG.astream` accept the new kwargs and pass them through
the resolver:

```python
async def aquery(self, question, *, user=None, session_id=None,
                 tools_enabled=None, agent=None, web=None, graph=None,
                 summaries=None, reranker=None, long_context_pass=None,
                 query_transforms=None, max_steps=None, top_k=5,
                 metadata_filter=None, response_model=None) -> Response: ...
```

Test: `tests/test_rag_facade_aquery_new_kwargs.py`.

---

## Phase 9 — Evaluation + observability

### 9.1 [N] `raghub/evaluation/agent_evals.py`

```python
class HotpotQASubsetEval(Evaluator): ...
class FreshQASubsetEval(Evaluator): ...
class BeirRerankerEval(Evaluator): ...
```

Test: `tests/evaluation/test_agent_evals.py` (uses canned fixtures).

### 9.2 [E] `raghub/evaluation/financebench.py`

`FinanceBenchEvaluator` records the resolved config snapshot alongside each
result so reports can attribute gains to specific tools/transforms.

Test: `tests/evaluation/test_financebench_attribution.py`.

### 9.3 [E] `raghub/observability/metrics.py`

Already added counters/histograms in 4.8, 5.4, 7.12. This change adds the
**dashboard** helpers in `raghub/observability/dashboard.py` (Prometheus
query helpers, JSON snapshot for `/health`).

Test: `tests/observability/test_dashboard.py`.

### 9.4 [E] `raghub/observability/tracing.py`

Add spans for each tool and each transform; ensure existing query span
nests them.

Test: `tests/observability/test_trace_tree.py`.

### 9.5 [E] `raghub/api/app.py`

Extend `/health` to include the new counters as a JSON block.

Test: `tests/api/test_health_extended.py`.

---

## Phase 10 — API + UI plumbing

### 10.1 [N] `raghub/api/streaming.py`

```python
def sse_format(event: PlannerEvent) -> bytes:
    return f"event: {event.kind}\ndata: {event.model_dump_json()}\n\n".encode()
```

Test: `tests/api/test_sse_format.py`.

### 10.2 [E] `raghub/api/app.py`

```python
@router.post("/query/stream")
async def query_stream(payload: QueryRequest, ...):
    async def gen():
        async for ev in rag.astream_agent(payload.question,
                                          user=..., **kwargs):
            yield sse_format(ev)
    return StreamingResponse(gen(), media_type="text/event-stream")
```

Test: `tests/api/test_query_stream_endpoint.py`.

### 10.3 [N] `raghub/api/agent_endpoint.py`

```python
@router.post("/agent/run")
async def agent_run(payload: QueryRequest, ...) -> QueryResponse:
    return await app_service.run_agent(token=..., **payload.model_dump())
```

Test: `tests/api/test_agent_endpoint.py`.

### 10.4 [E] `streamlit_app.py`

Replace `_render_chat`'s body with an agent-aware renderer:

```python
async for ev in rag.astream(..., return_events=True):
    if ev.kind == "thought":
        with st.expander(f"Thought (step {ev.step})"):
            st.write(ev.payload["thought"])
    elif ev.kind == "tool_call":
        with st.status(f"Calling {ev.payload['name']}…", expanded=False): ...
    elif ev.kind == "tool_result":
        ...
    elif ev.kind == "answer_chunk":
        placeholder.markdown(ev.payload["text"])
```

Test: `tests/streamlit/test_agent_renderer.py`.

### 10.5 [E] `raghub/api/rag.py`

`RAG.astream` returns `AsyncIterator[str]` today. Add `RAG.astream_agent`
that returns `AsyncIterator[PlannerEvent]` for the merged path:

```python
async def astream_agent(self, question, *, user=None, session_id=None,
                        **flags) -> AsyncIterator[PlannerEvent]: ...
```

Test: `tests/test_rag_facade_astream_agent.py`.

### 10.6 ★[N] `tests/regression/test_fast_path_unchanged.py`

The single most important regression: with `agent=False`, no tools,
no transforms, no reranker, no long-context pass, the pipeline's calls
into `vector_store.search`, `embedder.embed_text`, `reranker.rerank`,
and `generator.generate` must be byte-equivalent in args to today's
`QueryPipeline.run`. Captured by spying on those four collaborators.

Test: this *is* the test.

---

## Phase 11 — Docs

### 11.1 [N] `docs/ADVANCED_RAG.md`

Sections: query transforms, hybrid v2 (RRF + ColBERT), rerankers,
long-context pass, RAPTOR, GraphRAG, agent loop, tool/user config.

Test: `mkdocs build --strict` succeeds (CI hook).

### 11.2 [E] `README.md`

Update Features table; add "Advanced RAG" section linking to 11.1.

Test: `markdown-link-check README.md` passes in CI.

### 11.3 [N] `examples/agentic_rag_tour.ipynb`

Jupyter notebook walking through:
1. Basic `RAG.aquery`
2. With `agent=True, web=True`
3. With `query_transforms=["hyde","multi_query"]`
4. With `reranker="cascade"`
5. With `build_raptor=True` at ingest, then a summary search
6. With `build_graph=True` at ingest, then a graph query

Test: `pytest --nbval-lax examples/agentic_rag_tour.ipynb`.

### 11.4 [E] `CHANGELOG.md`

Add `0.5.0` entry summarising the new surface.

### 11.5 [E] `docs/index.md`

Update sidebar nav to include `ADVANCED_RAG.md`.

---

## Dependencies (optional extras — `pyproject.toml`)

### D.1 [E] `[project.optional-dependencies]`

```toml
agent  = ["duckduckgo-search>=6", "httpx>=0.27", "rank_bm25>=0.2"]
rerank = ["cohere>=5", "sentence-transformers>=3"]
colbert = ["ragatouille>=0.0.10"]
graph  = ["networkx>=3", "python-igraph>=0.11", "leidenalg>=0.10",
          "umap-learn>=0.5", "scikit-learn>=1.4"]
all-advanced = ["raghub[agent,rerank,colbert,graph]"]
```

Test: `pip install -e ".[agent,rerank,colbert,graph,all-advanced]"` succeeds
in CI; `pip check` clean.

### D.2 [E] `[project.entry-points."raghub.plugins"]`

Add entry-points for opt-in registration of the new rerankers and tools so
external packages can ship their own.

```toml
[project.entry-points."raghub.rerankers"]
cohere = "raghub.retrieval.rerankers.cohere:CohereReranker"
bge    = "raghub.retrieval.rerankers.bge:BgeReranker"
llm    = "raghub.retrieval.rerankers.llm:LLMReranker"

[project.entry-points."raghub.tools"]
web_search      = "raghub.agent.tools.web_search:WebSearchTool"
summary_search  = "raghub.agent.tools.summary_search:SummarySearchTool"
graph_search    = "raghub.agent.tools.graph_search:GraphSearchTool"
```

Test: `tests/plugins/test_entry_points.py`.

---

## End-to-end smoke

### E.1 [N] `tests/e2e/test_advanced_rag_smoke.py`

Covers in one run (gated by env var `RAGHUB_RUN_ADVANCED_E2E=1`):
- `RAG.ingest` with `build_raptor=True, build_graph=True`
- `RAG.aquery(question, agent=True, web=True, query_transforms=["hyde","multi_query"],
              reranker="cascade")` → returns answer + `planner_trace`
- `RAG.aquery(question)` (no flags) → fast path, zero tool calls

Test: itself.

### E.2 [N] `tests/e2e/test_user_preferences_flow.py`

```python
login -> set_pref("tool_settings", {...}) -> aquery with no request flags ->
assert resolved config matches prefs -> patch prefs ->
assert next aquery reflects new prefs
```

Test: itself.

---

## Merge order (one PR per phase is fine; smaller PRs inside a phase are better)

1. **Phase 1** (1.1 → 1.12) — unlocks everything; ship as PR #1.
2. **Phase 4** (4.1 → 4.8) — independent; ships early because it's a clean
   module under `retrieval/rerankers/`.
3. **Phase 8** (8.1 → 8.7) — depends only on Phase 1; gives users the
   toggles so they can experiment before the agent lands.
4. **Phase 2** (2.1 → 2.8) — depends only on Phase 1.
5. **Phase 3** (3.1 → 3.6) — independent of Phases 2/4.
6. **Phase 5** (5.1 → 5.4) — depends on Phase 4 reranker wiring in
   `QueryPipeline`.
7. **Phase 6** (6.1 → 6.8) — independent module, longest build time;
   land early so the rest can wire to it.
8. **Phase 7** (7.1 → 7.12) — depends on Phases 1, 2, 3, 6; biggest PR.
9. **Phase 9** (9.1 → 9.5) — wraps up metrics.
10. **Phase 10** (10.1 → 10.6) — final surface; **10.6 is the
    regression test** that locks the fast-path invariance.
11. **Phase 11** (11.1 → 11.5) — docs last.

---

## Risk ledger

- **Ponytail limits honoured**: every step above is the smallest viable diff.
  No new abstraction is introduced unless two callers need it (e.g. the
  Tool protocol is needed by both the planner and the registry).
- **Fast-path invariance** is gated by 10.6; that test must land before
  any other Phase 7 PR merges.
- **Optional deps everywhere**: nothing in Phases 2–7 is required for the
  default `pip install raghub`. Lazy imports + `try/except ImportError`
  are used at the seam.
- **No silent fallback to web search**: `WebSearchTool` raises
  `WebSearchError` if the dep is missing; the agent surfaces that
  cleanly via the trace, never silently no-ops.
- **Per-user prefs migration** is additive (`CREATE TABLE IF NOT EXISTS`);
  no destructive schema change.
- **RAPTOR/GraphRAG are off by default** at ingest (6.6, 6.7); they only
  run when the caller opts in.
- **Long-context pass** is gated by an allowlist (5.1); even with
  `enabled=True`, an unknown model produces a graceful no-op with a
  telemetry event.