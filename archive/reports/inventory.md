# raghub Inventory

generated: `2026-07-31T07:47:13+00:00`

## Summary

- Files scanned: **40**
- Class definitions: **264**
- Functions: **177**
- Enum candidates: **0**
- Private (`_`-prefixed) candidates: **5**
- Class collisions: **9**

## Class collisions (resolve before Phase 1.7)

### `Auth`
- `raghub/helper/auth.py:66`
- `raghub/services/__init__.py:618`

### `Chunk`
- `raghub/domain.py:32`
- `raghub/models.py:436`

### `ConversationStore`
- `raghub/conv.py:331`
- `raghub/models.py:1178`

### `Document`
- `raghub/domain.py:77`
- `raghub/models.py:428`
- `raghub/services/__init__.py:134`

### `DurationTimer`
- `raghub/pipeline.py:86`
- `raghub/utils.py:239`

### `Metrics`
- `raghub/eval/__init__.py:40`
- `raghub/models.py:923`

### `PromptBuilder`
- `raghub/models.py:1094`
- `raghub/prompts.py:117`

### `Query`
- `raghub/models.py:492`
- `raghub/services/__init__.py:308`

### `UnitOfWork`
- `raghub/domain.py:323`
- `raghub/repos.py:497`

## Private candidates (resolve in Phase 1.5)

- `_resolve_config_dir` — raghub/config.py:386
- `_env_int` — raghub/config.py:474
- `_env_float` — raghub/config.py:501
- `_evaluate` — raghub/eval/__init__.py:621
- `_is_aiosqlite_row` — raghub/stores/__init__.py:52

## Enum candidates

None detected.

## Modules with `__all__`

### `raghub.__init__`

- `RAG`
- `MissingDep`
- `RagHubError`
- `Settings`

### `raghub.agent`

- `Agent`
- `AgentTrace`
- `PlannerEvent`
- `build_tool_registry`
- `resolve`

### `raghub.auth`

- `AuthService`
- `SqliteUsers`
- `UserRecord`

### `raghub.config`

- `AgentConfig`
- `HybridConfig`
- `LongContextConfig`
- `QueryTransformsConfig`
- `RerankerConfig`
- `Settings`
- `WebSearchConfig`

### `raghub.conv`

- `Memory`

### `raghub.embedder`

- `Embedder`
- `Hasher`
- `LiteLLMEmbedder`
- `build_embedder`

### `raghub.errors`

- `AgentBudgetExceeded`
- `AuthenticationError`
- `AuthorizationError`
- `CacheMiss`
- `ConfigurationError`
- `ConversionError`
- `EmbeddingError`
- `EvaluationError`
- `GenerationError`
- `GraphUnavailableError`
- `IngestionError`
- `KnowledgeError`
- `MissingDep`
- `PipelineError`
- `PipelineFailed`
- `RagHubError`
- `RerankerError`
- `RetrievalError`
- `StreamingFormatError`
- `TelemetryError`
- `TokenBudgetExceeded`
- `ToolError`
- `TransformError`
- `VectorStoreError`
- `WebSearchError`

### `raghub.eval.__init__`

- `Finance`
- `Frames`
- `Gate`
- `Judge`
- `Metrics`
- `Scoring`
- `average`
- `compare`
- `parse`
- `run`

### `raghub.eval.ragas.__init__`

- `RagasAdapter`
- `import_ragas`
- `load_metric`

### `raghub.eval.synthetic`

- `SyntheticDataset`
- `chunk_id`
- `chunk_text`
- `clean_response`

### `raghub.gen`

- `DefaultGenerator`
- `Instructor`

### `raghub.helper.auth`

- `App`
- `Auth`
- `Bearer`

### `raghub.helper.cli`

- `CliConfig`
- `IngestCommand`
- `InitCommand`
- `QueryCommand`
- `ServerCommand`
- `ToolConfig`

### `raghub.helper.rate_limit`

- `RateLimiterMiddleware`
- `TokenBucket`

### `raghub.helper.response`

- `Redaction`
- `ResponseBuilder`

### `raghub.helper.search`

- `date`
- `graph`
- `hybrid`
- `keyword`
- `summary`
- `vector`
- `web`

### `raghub.helper.sse`

- `Sse`

### `raghub.ingest`

- `Batch`
- `Chonkie`
- `Ingestor`
- `Resumable`
- `WordChunker`
- `build_chonkie_chunker`

### `raghub.knowledge`

- `GraphIndex`
- `Manifest`
- `MemoryRepo`
- `Raptor`
- `sha256_bytes`

### `raghub.lifecycle.__init__`

- `ChunkingPlan`
- `Lifecycle`
- `Marker`
- `PlainTextConverter`
- `Section`
- `build_chunk_records`
- `build_marker_converter`
- `chunk_words`
- `convert_path`
- `datetime_now_utc`
- `detect_mime_type`
- `extract_pdf_metadata`
- `extract_pdf_pages`
- `extract_pdf_text`
- `extract_text`
- `looks_like_pdf`
- `md_to_blocks`
- `new_version`
- `normalise_markdown`
- `normalize_text`
- `pick_converter`
- `validate_upload`

### `raghub.llm`

- `LLM_API_KEY_ENV_VARS`
- `Generator`
- `HeuristicProvider`
- `LiteLLM`
- `any_llm_api_key_present`
- `build_llm`

### `raghub.models`

- `AuthLoginRequest`
- `AuthLoginResponse`
- `Bundle`
- `Chunk`
- `ChunkRecord`
- `Citation`
- `Classification`
- `ConversationTurn`
- `Document`
- `DocumentLifecycleStatus`
- `DocumentRecord`
- `DocumentUploadResponse`
- `Embedding`
- `Hit`
- `PipelineCtx`
- `PipelineResult`
- `Query`
- `QueryRequest`
- `QueryResponse`
- `RankedItem`
- `RankedList`
- `Response`
- `Result`
- `SearchRequest`
- `SearchResponse`
- `SearchResult`
- `SessionRecord`
- `User`
- `Visibility`
- `deterministic_id`

### `raghub.parsers`

- `HTML`
- `Catalog`
- `ChunkingPlan`
- `Csv`
- `Image`
- `Lifecycle`
- `Marker`
- `Office`
- `ParsedSection`
- `Pdf`
- `Txt`
- `chunk_words`
- `normalize_text`
- `parse`
- `validate_upload`

### `raghub.pipeline`

- `AgentPipeline`
- `IngestPipeline`
- `QueryCache`
- `QueryPipeline`

### `raghub.rag`

- `LLM_API_KEY_ENV_VARS`
- `RAG`
- `has_llm_api_key`

### `raghub.repos`

- `ChunkStore`
- `DocStore`
- `UnitOfWork`

### `raghub.retrieval.__init__`

- `Cascade`
- `Cohere`
- `Colbert`
- `Compose`
- `Context`
- `Decompose`
- `Fusion`
- `Hyde`
- `Identity`
- `LlmJudge`
- `MultiQuery`
- `Rerank`
- `RerankerFactory`
- `Retrieval`
- `Search`
- `SearchFilters`
- `StepBack`
- `Transformer`
- `Variant`
- `areranker`
- `build_filter`
- `build_reranker`
- `context_prompt`
- `decompose_prompt`
- `extract_json_array`
- `extract_json_object`
- `extract_string_array`
- `hyde_prompt`
- `linear_combine`
- `merge_with_rrf`
- `multi_query_prompt`
- `record_context_latency`
- `reorder_candidates`
- `rerank_latency`
- `reranker`
- `rrf`
- `step_back_prompt`
- `transform`

### `raghub.services.__init__`

- `Auth`
- `Document`
- `Facade`
- `Facade`
- `Health`
- `MemoryQueue`
- `Mixin`
- `Preference`
- `Query`
- `RagContainer`
- `Shutdown`
- `Synchronous`
- `ThreadPool`
- `aggregate_status`
- `build_container`
- `get_doc`
- `list_records`
- `missing_doc`
- `parse_users`
- `probe_embedder`
- `probe_vector_store`
- `seed_blocked`
- `seed_demo_users`
- `upload_record`

### `raghub.store`

- `MemoryStore`
- `SqliteStore`
- `Store`
- `build_store`

### `raghub.stores.__init__`

- `Database`
- `Documents`
- `ImageStore`
- `JsonSessions`
- `Sessions`
- `Snapshot`
- `migrate_from_json`

### `raghub.telemetry`

- `DEFAULT_METRICS_REGISTRY`
- `LangfuseTelemetryProvider`
- `LoguruTelemetryProvider`
- `MetricsRegistry`
- `NoOpTelemetry`
- `PrometheusMetrics`
- `RedactingTelemetry`
- `build_logger`

### `raghub.tools.__init__`

- `DateToday`
- `GraphSearch`
- `HybridSearch`
- `KeywordSearch`
- `SummarySearch`
- `Tool`
- `ToolContext`
- `ToolProtocol`
- `ToolRegistry`
- `ToolResult`
- `VectorSearch`
- `WebSearch`
- `as_admin_user`
