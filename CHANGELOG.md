# Changelog

All notable changes to RAGHub are documented in this file. The
format follows [Keep a Changelog](https://keepachangelog.com/) and
the project adheres to [Semantic Versioning](https://semver.org/).

## [Unreleased]

### Added

- **Module-level docstrings** for all 10 legacy domain / repository
  modules (``raghub.domain.*``, ``raghub.repositories.*``).
  Every public function, class, and method has an explicit
  docstring.
- **Docstrings for CLI and retrieval hot-paths**:
  ``handle_health``, ``SearchFilter.build_filter_string``,
  ``FacetedSearchEngine``, ``IdentityReranker.rerank``,
  ``KnowledgeManifest.__contains__`` / ``__getitem__``,
  ``marker_converter_instance``, ``ensure_loaded_examples``,
  ``safe_histogram`` / ``safe_counter``.
- **Lazy import wrapper** for ``migrate_from_json`` in
  ``raghub.storage`` to break the pre-existing circular import
  cycle between ``storage`` / ``repositories`` / ``domain``.

### Changed

- **Public naming conventions**: every ``_``-prefixed name across
  the entire codebase has been renamed to public (e.g.
  ``_MARKER_AVAILABLE`` → ``MARKER_AVAILABLE``,
  ``self._llm`` → ``self.llm``, ``_ensure_examples`` →
  ``ensure_examples``). ~90 names updated across 30+ files.
- **``setup.sh``** now uses ``pip install -e ".[dev,api,ui,zvec]"``
  instead of ``requirements/*.txt``, aligning with
  ``pyproject.toml`` as the single source of truth for
  dependencies.
- **``raghub/cli/_common.py``** renamed to ``raghub/cli/common.py``
  (no underscore prefix). All 5 import sites updated.
- **``QdrantVectorStore``** merged private ``_upsert`` and
  ``_search`` methods into the public ``upsert`` / ``search``
  methods. ``insert`` now calls ``self.upsert``.
- **``InstructorStructuredOutputProvider.astream``** uses a nested
  async generator instead of ``yield`` directly, fixing the mypy
  override mismatch without suppression.
- **``vectorstore.BaseVectorStore.search``** and
  ``hybrid_search`` widened ``metadata_filter`` parameter from
  ``str`` to ``str | dict`` to match both in-memory and Qdrant
  backends.

### Removed

- **All 29 ``# type: ignore`` suppression comments** — underlying
  type issues resolved (optional-dep type annotations via ``Any``,
  ``sys.version_info`` branches for ``tomllib`` compat, aligned
  protocol signatures, ``cast()`` for ``asyncio.run``, etc.).
- **All 12 ``# noqa`` suppression comments** — late imports
  restructured via ``TYPE_CHECKING`` or inlined; ``bench/benchmark.py``
  uses ``find_spec`` instead of probe-import; ``per-file-ignores``
  in ``pyproject.toml`` for the standard ``importorskip`` pattern.
- **Dead fields** from ``AppSettings``: ``retrieval_mode``
  and ``worker_backend`` (unused outside of settings loading).
- **Dead code**: ``merge_mapping`` utility, unused
  ``delete_chunks_by_id`` call in ``QdrantVectorStore.delete``,
  unused ``query_filter`` / ``query`` params from
  ``QdrantVectorStore._search``, unused ``_device`` attribute from
  ``MarkerConverter``, dead ``if False else True`` in
  ``test_security.py``.
- **Stale ``pytest.ini`` warning filter**
  ``error::jwt.warnings.InsecureKeyLengthWarning`` (PyJWT API
  removed the warning class; the filter blocked all test
  execution).

### Fixed

- **All 34 mypy errors** across 18 source files — type annotations,
  incompatible types, missing ``delete_version`` in
  ``QdrantVectorStore``, dict type mismatches, async generator
  override signatures, abstract method conformance.
- **All 34 ruff errors** — unused imports, unused variables,
  module-level import violations (E402).
- **Pre-existing test failures** uncovered by the pytest.ini fix:
  ``JsonDocumentRegistry.get_specific_version`` (renamed from
  ``get_version``), ``_IncludedRouter.path`` attribute error in
  ``test_legacy_services.py``, mock-module setup in litellm tests,
  ``session_id_or_user`` attribute in security test,
  ``marker_converter`` skip (now runnable with installed dep),
  ``LiteLLMProvider`` streaming test mock setup.
- **Missing ``delete_version``** in ``QdrantVectorStore`` — now
  delegates to the existing delete-by-collection implementation.
- **PyJWT test blocker**: removed stale
  ``error::jwt.warnings.InsecureKeyLengthWarning`` filter from
  ``pytest.ini`` that prevented any test from running.

### Security

- ``JWT_SECRET`` must be ≥ 32 bytes in production (PyJWT's
  ``InsecureKeyLengthWarning`` is now an error in CI).
- ``RedactingTelemetry`` scrubs secret-looking kwargs before
  forwarding to the telemetry backend.
- Per-user retrieval isolation: unauthorised users receive empty
  result sets, not 403 errors.

---

## [v0.2.0-beta] - 2026-06-30

`8fbdbcd` — Beta release of the RAGHub platform rewrite.

### Added

- **Public ``RAG`` facade** (`raghub.api.rag.RAG`) is the single
  recommended entry point. Every spec component — Marker,
  Chonkie, LiteLLM, Instructor, Qdrant, Langfuse — is wired as a
  default but can be replaced through the constructor.
- **Multi-user RBAC** at the retrieval layer via
  ``allowed_company_filter``. Admins see every company;
  non-admins see only chunks whose ``company`` is in their
  ``allowed_companies``. The LLM only ever receives authorised
  context. Unauthorised users with empty allow-lists see no
  documents.
- **Conversational RAG** with session-scoped history. Every
  query accepts ``session_id=``; the ``InMemoryConversationStore``
  prepends the most recent turns so the LLM can answer
  follow-up questions.
- **Streamlit UI** (`streamlit_app.py`) with ``st.chat_message``
  and ``st.chat_input``, citation rendering per turn, per-user
  document upload scoped to the user's company, and persistent
  sign-in. Pre-seeds five demo users.
- **Token-usage tracking**: ``LiteLLMProvider.generate`` and
  ``astream`` populate ``self.last_usage``; ``DefaultGenerator``
  exposes ``record_tokens()``; the ``QueryPipeline`` forwards
  tokens to the telemetry provider on every call.
- **Real streaming**: ``RAG.astream`` routes through
  ``QueryPipeline.stream`` → ``DefaultGenerator.astream`` →
  ``LiteLLMProvider.astream`` (with
  ``stream_options={"include_usage": True}``).
- **Incremental indexing** by SHA-256 content hash.
  ``IngestPipeline`` short-circuits when the file hasn't changed.
- **Resumable background ingestion** via
  ``ResumableBackgroundIngestionService`` (persistent job
  ledger in SQLite).
- **Retrieval-quality metrics** (``raghub.evaluation.metrics``):
  ``recall_at_k``, ``precision_at_k``, ``mean_reciprocal_rank``,
  ``context_recall``, ``context_precision``, ``faithfulness``,
  ``answer_correctness``. Integrated into ``FinanceBenchEvaluator``.
- **Performance benchmark** (``bench/benchmark.py``) measures
  startup time, ingestion throughput, query latency (p50/p95),
  queries-per-second under concurrency, and peak RSS.
- **10 spec libraries** wired as defaults: Marker (PDF → MD),
  Chonkie (chunking), LiteLLM (LLM + embeddings), Instructor
  (structured output), Qdrant (vector store), Langfuse
  (telemetry), OKF (knowledge), datasets (FinanceBench), pypdf
  (legacy), Tika-style parser fallback.
- **Open Knowledge Format (OKF)** is the canonical persisted
  representation. ``KnowledgeBundle`` round-trips through
  ``to_okf`` / ``from_okf``.
- **Plugin registry** with entry-point discovery
  (``[project.entry-points."raghub.plugins"]``).
- **Structured output** via Instructor; ``RAG.query(..., response_model=MyModel)``
  returns a typed ``BaseModel`` in ``Response.structured``.
- **Comprehensive test suite**: 315+ tests across 30+ test
  files, including 10-user concurrent multi-user tests, security
  tests, end-to-end tests, retrieval-metric tests, and benchmark
  tests.

### Changed

- **Public API surface** is now the new ``RAG`` facade. The legacy
  ``DynamicRagApplication`` and ``Build_application`` entry points
  remain available for backwards compatibility but are no longer
  the recommended path.
- **Configuration precedence**: env > TOML > YAML > defaults.
  ``AppSettings.override(**changes)`` provides runtime
  overrides.
- **Embedders and LLMs** fall back to in-process
  (``HashingEmbeddingProvider``, ``HeuristicLLMProvider``) when no
  API key is configured.
- **Default chunker** uses Chonkie when available, falls back to
  ``WordWindowChunker``.

### Security

- ``JWT_SECRET`` must be ≥ 32 bytes in production (PyJWT's
  ``InsecureKeyLengthWarning`` is now an error in CI).
- ``RedactingTelemetry`` scrubs secret-looking kwargs before
  forwarding to the telemetry backend.
- Per-user retrieval isolation: unauthorised users receive empty
  result sets, not 403 errors.

### Fixed

- ``RAG.astream`` is now a real token stream (previously
  materialised the full answer before yielding).
- ``LiteLLMProvider.astream`` populates ``last_usage`` with
  ``stream_options={"include_usage": True}``.
- ``MarkerConverter.convert`` raises a clear
  ``ConfigurationError`` on empty / non-PDF input instead of a
  raw PDFium crash.
- ``RAG.ingest`` and ``RAG.aingest`` raise ``RagHubError`` on
  empty bytes.
- The 4 ``InsecureKeyLengthWarning`` instances that fired in
  every test run are gone.
- Fixed circular imports between ``raghub.domain`` and
  ``raghub.repositories`` (legacy code now lazy-imports).

### Deferred

- ``AppSettings.jwt_secret`` will move to ``SecretStr`` in a future
  release. The current ``str`` annotation is required for the
  Pydantic 1.x-style dataclass shape; migrating to ``BaseModel``
  is a larger refactor.
- The 48 remaining ``mypy`` warnings (legacy modules) — resolved
  in the Unreleased changeset.

### Removed

- **Dead code**: ``raghub/cache/``, ``raghub/memory/``,
  ``raghub/monitoring/`` (placeholder packages), the legacy
  ``raghub.llm.nvidia`` and ``raghub.embeddings.nvidia`` modules
  (superseded by LiteLLM), the duplicate
  ``raghub/vectorstores/`` (plural) package (merged into
  ``raghub/vectorstore/``), and the ``TemplatePromptBuilder``
  class.
- The legacy CLI's ``login`` command (replaced by the FastAPI
  ``/auth/login`` endpoint and the demo users in the Streamlit
  UI).
- Standalone ``evaluate_financebench.py`` at the repo root
  (replaced by ``raghub-financebench`` console script and
  ``raghub.cli.eval_cmd``).

---

## [v0.1.0] - 2026-06-28

`4146805` — First versioned release of the RAGHub rewrite.

Same feature set as v0.1.0-alpha. This tag marks the initial
stable version.

---

## [v0.1.0-alpha] - 2026-06-28

`71d4be0` — Initial alpha of the RAGHub platform rewrite.

- Renamed from ``dynamic_rag`` to ``raghub``.
- Eliminated ``_``-prefixed naming convention for Google-style
  compliance (``512177e``).
- Multi-format parsers (PDF, HTML, image OCR, Office, CSV, TXT).
- NVIDIA NV-Embed-QA and sentence-transformers embedding providers.
- SQLite persistence for document registry, session store, and
  user store.
- JWT authenticator and RBAC authorization.
- Structlog telemetry, Prometheus metrics, OpenTelemetry tracing.
- Rate limiter, sliding-window conversation, async ingestion,
  admin API, faceted search.
- Wiring of all components into ``build_container`` and the API
  layer.
- Comprehensive test suite.
- Developer setup via ``setup.sh`` and ``requirements/*.txt``.

---

## [v0.0.1] - 2026-06-26

`edbcb43` — Project inception.

- Initial codebase from the ``dynamic_rag`` prototype.
- Package rename to ``raghub``.
