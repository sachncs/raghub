# raghub OSS Readiness — TODO

> **Final plan.** Each phase = 10–20 commits. Each commit lands green (`ruff` /
> `interrogate` / `mypy` / `pytest`). Tests validate *correctness*, *accurate
> data translation*, and *behaviour* — not "is not None" smoke checks.

---

## Hard rules (verbatim)

```
R1  No `# noqa:`. Every lint violation is fixed in the diff.
R2  Two-tier privacy: public OR `__<one-word>`. `_`-prefix forbidden.
R3  Single-word public class names. Discriminator enums: `<Entity>Type`.
R4  No backward compat. No aliases. No deprecation. No shims.
R5  Hard rename, single `docs/migration.md`.
R6  Storage layer version-pins on-disk format; migrates v1 → v2 on read.
R7  `<Entity>Type` enums must have ≥2 values; collapse to `str` otherwise.
R8  `verify()` mandatory at every storage and API boundary.
R9  Every state change is captured by an assertion that names the value.
R10 No `Any` outside `metadata`. `metadata` is the only `Any` slot.
```

---

## Phase 0 — Inventory & lint tightening

| # | Commit | Files | What must be true after this commit |
| --- | --- | --- | --- |
| 0.1 | `.gitignore` rules | `.gitignore` | ignores `lint/`, `reports/`, `*.pyc`, `.coverage.*`. |
| 0.2 | Make targets | `Makefile` | targets `inventory`, `lint`, `format`, `test`, `mypy`, `coverage`, `docstrings`, `naming`, `migrate`. |
| 0.3 | `pyproject.toml`: ruff strict | `pyproject.toml` | `select = [E,F,I,B,UP,SIM,RUF,D,N,PLR]`; ignores empty (per-file-ignores for tests). |
| 0.4 | `pyproject.toml`: interrogate | `pyproject.toml` | `interrogate fail-under = 100`, `ignore-semiprivate = false`. |
| 0.5 | `pyproject.toml`: mypy strict | `pyproject.toml` | `disallow_untyped_defs = true`, `warn_unused_ignores = true`. |
| 0.6 | CI workflow | `.github/workflows/ci.yml` | stages: inventory → lint → docstrings → naming → mypy → test → coverage (≥85%). |
| 0.7 | Local naming hook | `lint/naming.py` (gitignored) | walks `from raghub.X import Y` paths; rejects `_*` and undeclared `__all__`. |
| 0.8 | `make inventory` JSON | `reports/inventory.json` | every class, every enum, every `_`-prefix candidate, every collision. |
| 0.9 | `make inventory` Markdown | `reports/inventory.md` | human-readable inventory generated from JSON. |
| 0.10 | Stage explicit `__init__.py` files | `raghub/{config,parser,store}/__init__.py` (existing); new `__init__.py` for every subpackage | no implicit namespace packages anywhere. |

**Test contract after Phase 0:** `pytest` returns the same baseline (410 passed / 9 skipped). `ruff` shows the *new* violations, but does not block.

---

## Phase 1 — Lint baseline

Commit-by-commit fix of every violation surfaced in Phase 0.3.

| # | Commit | Fixes | Test contract |
| --- | --- | --- | --- |
| 1.1 | `ruff format` | 378 COM812 (missing trailing commas) | `ruff format --check` passes. |
| 1.2 | Line-length sweep | 27 E501 | every line ≤100 chars. |
| 1.3 | `B008` sweep | 27 function-call-in-default | all replaced with `Field(default_factory=...)` or `functools.partial`. |
| 1.4 | `D101` sweep | undocumented public classes (60 occurrences across 5 files) | each class has a one-line docstring. |
| 1.5 | `D102` sweep | undocumented public methods (211 occurrences) | each method has a docstring. |
| 1.6 | `D103` sweep | undocumented module-level functions (45 occurrences) | each function has a docstring. |
| 1.7 | `D205/D209` sweep | missing blank lines in docstrings (34 occurrences) | docstrings follow Google style. |
| 1.8 | `D301` sweep | escape sequences in docstrings (4) | raw strings used. |
| 1.9 | `D401/D415` sweep | imperative mood + period (10 occurrences) | all docstrings end with period. |
| 1.10 | `N806` sweep | non-lowercase variables (5) | variables lower-snake. |
| 1.11 | `N818` sweep | exception suffix | none pending (already follows `*Error`). |
| 1.12 | `PLR2004` sweep | magic-number comparisons (81) | constants extracted to module scope. |
| 1.13 | `PLR0913` sweep | too-many-arguments (38) | signatures restructured with `TypedDict` payloads. |
| 1.14 | `PLR0912` sweep | too-many-branches (4) | early returns extracted to helpers. |
| 1.15 | `PLR0915` sweep | too-many-statements (3) | helpers extracted. |
| 1.16 | `SIM117` | multi-with-statement merges (2) | `with a, b:` combined. |
| 1.17 | `B017` | `pytest.raises(Exception)` (2) | specific exception types. |
| 1.18 | `SIM103` | needless bool returns (1) | direct returns. |
| 1.19 | `UP042` sweep | older typing idiom (covered by other UP-rules) | consistent modern forms. |
| 1.20 | Stragglers | any remaining | `ruff check` returns 0 with the strict config. |

**Test contract:** `pytest` still returns the baseline. `ruff check`, `ruff format --check`, `interrogate --fail-under=100` all green.

---

## Phase 1.5 — Underscore purge + Tokenizer

Single phase, fewer commits. Each item lands with the lint + test baselines still green.

| # | Commit | Files | What must be true |
| --- | --- | --- | --- |
| 1.5.1 | `stores`: rename `_is_aiosqlite_row` → `__keyed` | `raghub/stores/__init__.py` | importable only as `raghub.stores._stores__keyed` (mangled). |
| 1.5.2 | `config`: rename `_resolve_config_dir` → `__resolve` | `raghub/config.py` | same mangled-import rule. |
| 1.5.3 | `config`: inline `_env_int` call sites (8) | `raghub/config.py` | no `_env_int` symbol; behaviour identical. |
| 1.5.4 | `config`: inline `_env_float` call sites (1) | `raghub/config.py` | no `_env_float` symbol. |
| 1.5.5 | `conv`: introduce `Tokenizer` class | `raghub/conv.py` | `Tokenizer.load(model=...)` returns `Tokenizer` instance; `MissingDep` raised cleanly on missing dep. |
| 1.5.6 | `conv`: deprecate `try_load_gigatoken` (call sites only) | `raghub/conv.py` callers | every call site updated. |
| 1.5.7 | `__all__` audit pass 1 | every subpackage | every public name reachable from outside has a docstring and is in `__all__`. |

**Test contract:** Each renamed/moved symbol's behaviour is captured by an `assert`:

```
assert Tokenizer.load().model == "Qwen/Qwen3-8B"
assert Tokenizer.load(missing=True) raises MissingDep
assert stores._stores__keyed == <exact same function>
assert config._config__resolve is callable
```

---

## Phase 1.7 — Universal entity schema + `<Entity>Type` enums

**This phase and the next (1.8) commit as one. Splitting them leaves the tree red.**

| # | Commit | Files | What must be true |
| --- | --- | --- | --- |
| 1.7.1 | `DocumentRecord` → `Document` cascade | `raghub/models.py`, `raghub/repos.py`, every importer | `Document.id`, `Document.source`, `Document.target`, `Document.chunks`, `Document.type: DocType`, `Document.verify()`. |
| 1.7.2 | `ChunkRecord` + `domain.Chunk` → `Chunk` cascade | same | `Chunk.id`, `Chunk.source`, `Chunk.parent: Document.id`, `Chunk.text`, `Chunk.checksum`, `Chunk.type: ChunkType`, `Chunk.verify()`. |
| 1.7.3 | `Hit` → `Hit` (already single; rename and shape) | same | `Hit.chunk: Chunk`, `Hit.score`, `Hit.rank`, `Hit.type: HitType`, `Hit.verify()`. |
| 1.7.4 | `Citation` shape change | same | `Citation.chunk: Chunk`, `Citation.source: str` (char-span), no `text` (derive). `Citation.type: CitationType`. `Citation.verify()`. |
| 1.7.5 | `Citations` aggregate (new) | `raghub/models.py` | `class Citations(BaseModel)` with `items: list[Citation]`, `verify(chunks)`. |
| 1.7.6 | `Response` shape | same | `Response.answer`, `Response.citations: Citations`, `Response.chunks: list[Chunk]`, `Response.id`, `Response.type: ResponseType`, `Response.source = Session.id`, `Response.verify()`. |
| 1.7.7 | `Section`, `Block` rename + shape | same | `Section.blocks: list[Block]`, `Section.type: SectionType`, `Section.verify()`. |
| 1.7.8 | `Embedding` shape | same | `Embedding.target = Chunk.id`, `Embedding.type: EmbeddingType`, `Embedding.verify()` (vector length matches `Chunk.parent.embedding_dim`). |
| 1.7.9 | `Bundle` shape | same | `Bundle.sections: list[Section]`, `Bundle.target = Document.id`, `Bundle.type: BundleType`, `Bundle.verify()`. |
| 1.7.10 | `PipelineResult` shape | same | `PipelineResult.outputs: IngestOutputs | QueryOutputs | AgentOutputs` (discriminated union). `error: ErrorInfo | None`. No `success` field. `PipelineResult.verify()`. |
| 1.7.11 | `RankedList`, `RankedItem` | same | `RankedList.items: list[Hit]`, `RankedList.type: RankType`, `RankedList.verify()`. |
| 1.7.12 | `Manifest` shape | `raghub/knowledge.py` | `Manifest.entries: list[Entry]`, `Manifest.version: int` (`= 2`), `Manifest.type: ManifestType`, `Manifest.verify()`. |
| 1.7.13 | `UserRecord`, `SessionRecord`, `ConversationTurn` → `User`, `Session`, `Turn` | `raghub/models.py`, `raghub/auth.py`, `raghub/conv.py` | `Session.token` separate from `Session.target`; `Session.identity = Session.id`. |
| 1.7.14 | `PlannerEvent` → `Event` | `raghub/agent.py` | `Event.id`, `Event.kind`, `Event.step`, `Event.source = Session.id`, `Event.verify()`. |
| 1.7.15 | `IngestionJob` → `Job` | `raghub/ingest.py`, `raghub/services/__init__.py` | `Job.id`, `Job.target = Document.id`, `Job.state: State`, `Job.verify()`. |
| 1.7.16 | `Result` (eval) shape | `raghub/eval/__init__.py` | `Result.id`, `Result.target = example.id`, `Result.metrics: dict`, `Result.details: dict`, `Result.passed: bool`, `Result.verify()`. |
| 1.7.17 | `<Entity>Type` enums | `raghub/models.py` | one enum per entity class with ≥2 values. |
| 1.7.18 | Shared enums: `State`, `Class`, `Access` | `raghub/models.py` | single source of truth. |
| 1.7.19 | `VerificationError` | `raghub/errors.py` | raised by every `verify()` failure. |
| 1.7.20 | Typed `metadata: dict` → `Metadata = TypedDict` per entity | `raghub/models/base.py` (new) | typed metadata slots; no `dict[str, Any]` outside. |

**Test contract:** each commit wires up a builder + verify assertion pair:

```
assert Chunk(id="c1", text="x", checksum=sha256("x")).verify() is None
assert Chunk(id="c1", text="x", checksum="bogus").verify() raises VerificationError
assert Hit(chunk=chunk, score=0.5).verify() is None
assert Citation(chunk=chunk, source="0:1").verify() matches chunk.text[0:1]
assert Citations(items=[cit]).verify([chunk]) is None
assert Citations(items=[Citation(chunk=other, ...)]).verify([chunk]) raises VerificationError
```

---

## Phase 1.8 — Cascade propagation (single commit)

Every consumer compiles against the new schema. File-by-file updates in *one* commit because the `chunk_id` → `id` and `source_chunks` → `chunks` rewrites cross every layer.

| Files | What changes |
| --- | --- |
| `raghub/services/__init__.py` | every `chunk_id` param → `chunk.id`; `source_chunks` → `chunks`. |
| `raghub/ingest.py` | ingest pipeline reads `chunk.id`, writes `id` to repos. |
| `raghub/lifecycle/__init__.py` | chunk factory uses `Chunk.id`. |
| `raghub/retrieval/__init__.py` | hits carry `chunk: Chunk`. |
| `raghub/stores/__init__.py` | `JsonSessions`, `Documents`, `Database` use new models. |
| `raghub/repos.py` | SQL DDL uses `id` column. |
| `raghub/api.py` | response DTOs use `Response`, `Citation`. |
| `tests/test_*.py` | every test that builds an entity uses new fields. |
| `devtools/*.py` | consumer updates. |

**Test contract:** `mypy raghub/` clean. `pytest` returns the same baseline (410 / 9). `make inventory` shows no `_`-prefix public names.

---

## Phase 1.9 — Storage migration

| # | Commit | Files | What must be true |
| --- | --- | --- | --- |
| 1.9.1 | `Manifest.version = 2` constant | `raghub/knowledge.py` | manifests on disk carry a version. |
| 1.9.2 | `JsonSessions.load()` v1→v2 migration | `raghub/stores/__init__.py` | reads old + new; writes new. |
| 1.9.3 | `Documents.load()` v1→v2 migration | same | same. |
| 1.9.4 | SQLite schema migration | `raghub/repos.py` | reads `chunk_id` column, projects to `id`. |
| 1.9.5 | `raghub migrate <path>` CLI | new `raghub/migrate.py` | command line tool. |
| 1.9.6 | Devtools `migrate.py` dry-run | new `devtools/migrate.py` | reports incompatible records without modifying. |
| 1.9.7 | Migration test | `tests/test_migrate.py` | round-trips old fixtures to new format. |
| 1.9.8 | On-write version-pin | `raghub/knowledge.py`, `raghub/stores/__init__.py` | every write sets `version = 2`. |
| 1.9.9 | `IngestPipeline` on disk write | `raghub/ingest.py` | uses new schema exclusively. |
| 1.9.10 | Migration guide entry | `docs/migration.md` | section "v0 → v1 storage format". |

**Test contract:**
- `test_migrate.py` reads a v0 fixture (committed under `tests/fixtures/v0/`), runs the CLI, asserts the new file has `version: 2` and the right `Chunk`/`Document` shapes.
- `make migrate` is idempotent: running it on a v1 manifest leaves it v1 unchanged.

---

## Phase 1.10 — Docstrings 100%

10 commits, one per file-cluster where docstrings were missing.

| # | Commit | Files | Contract |
| --- | --- | --- | --- |
| 1.10.1 | `raghub/api.py` route handlers | `raghub/api.py` | each of the 22 handler closures gets a Google-style docstring. |
| 1.10.2 | `raghub/domain.py` | `raghub/domain.py` | `DatabaseManager` + 3 methods get docstrings. |
| 1.10.3 | `raghub/store.py` | `raghub/store.py` | the 8 undocumented methods get docstrings. |
| 1.10.4 | `raghub/evaluation.py` | `raghub/evaluation.py` | nested `runner`/`factory` get docstrings. |
| 1.10.5 | `raghub/helper/cli.py` | `raghub/helper/cli.py` | nested `runner` get docstrings. |
| 1.10.6 | `raghub/repos.py`, `retrieval/__init__.py` prompt builders | same | undocumented methods. |
| 1.10.7 | `raghub/pipeline.py` `bounded` | same | one-liner. |
| 1.10.8 | `raghub/telemetry.py` 4 helpers | same | each. |
| 1.10.9 | `raghub/eval/__init__.py` factory_a / factory_b / parse | same | each. |
| 1.10.10 | `make docstrings` enforced in CI | `Makefile`, `.github/workflows/ci.yml` | `interrogate --fail-under=100` is part of CI. |

**Test contract:** `interrogate -c pyproject.toml` exits 0.

---

## Phase 2 — Drop `raghub/helper/`

10 commits, one per file-move + per cascade update.

| # | Commit | Files | What must be true |
| --- | --- | --- | --- |
| 2.1 | New `raghub/api_auth/` package | `raghub/api_auth/__init__.py`, `raghub/api_auth/dependencies.py` | `App`, `Auth`, `Bearer` live at top-level. |
| 2.2 | New `raghub/api_response/` package | same | `Redaction`, `ResponseBuilder`. |
| 2.3 | New `raghub/api_sse/` package | same | `Sse`. |
| 2.4 | New `raghub/api_ratelimit/` package | same | `RateLimiterMiddleware`, `Token`. |
| 2.5 | New `raghub/cli_commands/` package | same | `CliConfig`, `ToolConfig`, `IngestCommand`, `InitCommand`, `QueryCommand`, `ServerCommand`. |
| 2.6 | Delete `raghub/helper/search.py` | `raghub/helper/search.py` removed | callers use `Tool.call` directly. |
| 2.7 | Update `raghub/api.py` imports | `raghub/api.py` | imports from new locations. |
| 2.8 | Update `raghub/cli.py` imports | `raghub/cli.py` | imports from `raghub.cli_commands`. |
| 2.9 | Update `raghub/rag.py` imports | `raghub/rag.py` | `ResponseBuilder` import path. |
| 2.10 | Delete `raghub/helper/` directory | `raghub/helper/` removed | verified no stragglers via `rg raghub.helper`. |

**Test contract:** `rg raghub.helper / raghub / tests / devtools` returns 0 hits. `pytest` baseline.

---

## Phase 3 — Pipeline reconciliation

10 commits across pipeline + ingest + services.

| # | Commit | Files | What must be true |
| --- | --- | --- | --- |
| 3.1 | `IngestionResult` → `PipelineResult` (single home) | `raghub/pipeline.py`, `raghub/ingest.py` | one class. |
| 3.2 | `IngestionJob` → `Job` | `raghub/ingest.py` | service + model sites updated. |
| 3.3 | `PersistentJobStore` → `JobStore` | same | renamed. |
| 3.4 | `PipelineResultBuilder` → `PipelineResult.from(...)` | `raghub/pipeline.py` | classmethod. |
| 3.5 | `QueryCache` → `Cache` | same | single home. |
| 3.6 | `ConversationRouter` → `Router` | `raghub/pipeline.py`, `raghub/conv.py` callers | single home. |
| 3.7 | `DurationTimer` × 2 → 1 | `raghub/utils.py` (single), `raghub/pipeline.py` (deleted copy) | one. |
| 3.8 | `ConversationManager` → `Conversations`, `SlidingWindowManager` → `SlidingWindow`, `ConversationStore` → `Store` | `raghub/conv.py` | single home each. |
| 3.9 | `Pipeline` Protocol single home | `raghub/models.py` (Protocol), `raghub/pipeline.py` (implementations) | no duplication. |
| 3.10 | Tool-prefix redundancy gone | `raghub/tools/__init__.py` | `ToolContext → Context`, `ToolResult → Result`, `ToolRegistry → Registry`. |

**Test contract:** Each renamed symbol's behaviour pinned by an assertion:

```
assert PipelineResult.from(success=True, error=None).verify() is None
assert Job(target="d1", state=State.READY).verify() requires Document(target="d1")
assert Cache(ttl_seconds=60).verify() is None
assert Router(store=Memory()).load_history("s1") == [...]
```

---

## Phase 4 — Real-impl tests + data path

The bulk of the validation work. 20 commits, each rewriting one test file to:
1. Use real implementations.
2. Pin behaviour with content assertions.
3. Drop `_`-prefixed helpers (Rule R2).
4. Drop `__new__(Class)` bypasses (Rule R4).
5. Drop `_load_services` pre-binding dead code.

| # | Commit | Test file | What must be true |
| --- | --- | --- | --- |
| 4.1 | `tests/test_data_path.py` (new) | new file | full convert → chunk → embed → store → retrieve → generate round-trip with content assertions. |
| 4.2 | `test_benchmark_smoke.py` rewrite | reuses real evaluators. |
| 4.3 | `test_config_loading.py` rewrite | asserts Settings round-trip from real YAML/TOML. |
| 4.4 | `test_config_validation.py` rewrite | asserts pydantic v2 validation errors. |
| 4.5 | `test_embedder.py` rewrite | asserts `Hasher` deterministic + dimension. |
| 4.6 | `test_end_to_end.py` rewrite | asserts RBAC, conversational follow-up, streaming. |
| 4.7 | `test_evaluation.py` rewrite | asserts `Finance`, `Frames`, `Gate` real evaluation. |
| 4.8 | `test_exceptions.py` rewrite | asserts `VerificationError` on bad entities. |
| 4.9 | `test_heuristic_llm.py` rewrite | asserts `HeuristicProvider` behaviour. |
| 4.10 | `test_hypothesis_properties.py` rewrite | asserts generators + shrinking. |
| 4.11 | `test_ingestion.py` rewrite | asserts `WordChunker` checksum fix. |
| 4.12 | `test_integration_data_flow.py` rewrite | asserts end-to-end ingestion + query. |
| 4.13 | `test_llm.py` rewrite | asserts `HeuristicProvider` for offline paths. |
| 4.14 | `test_memory_store.py` rewrite | asserts `MemoryStore` correctness. |
| 4.15 | `test_model_validators.py` rewrite | asserts new `verify()` chain. |
| 4.16 | `test_pipeline.py` rewrite | asserts real `IngestPipeline` / `QueryPipeline` with content checks. |
| 4.17 | `test_production_readiness.py` rewrite | drop `_load_services`; assert real RBAC, CORS. |
| 4.18 | `test_rag_facade.py` rewrite | drop `__globals__` patches; use `evaluator=` injection. |
| 4.19 | `test_ragas_adapter.py` rewrite | drop `RagasAdapter.__new__`; use public `from_dataset()`. |
| 4.20 | `test_security.py`, `test_services.py`, `test_sqlite_store.py`, `test_storage_database.py`, `test_store_memory.py`, `test_synthetic.py` rewrite | each file drops mocks and asserts content. |

**Test contract per file (sample):**

`tests/test_data_path.py`:

```
def test_bytes_round_trip():
    r = RAG(converter=PlainTextConverter())
    r.ingest(b"Revenue grew 12% in Q3 2024.", source_uri="mem://test")
    response = r.query("revenue")
    assert response.verify() is None
    assert any("Revenue" in c.chunk.text for c in response.citations)
    assert response.verify() is None
    assert sum(1 for _ in r.store.chunks) == r.store.health()["chunks"]

def test_chunk_checksum_round_trip():
    text = b"the quick brown fox " * 200
    r = RAG(converter=PlainTextConverter())
    r.ingest(text, source_uri="mem://ck")
    [c.verify() for c in r.store.chunks]  # all pass

def test_rbac_filters_across_chunks():
    alice = User(id="a", type=UserKind.STANDARD, identity="alice@x.com", class_=Class.INTERNAL)
    bob = User(id="b", type=UserKind.STANDARD, identity="bob@x.com", class_=Class.INTERNAL)
    r.ingest(b"<acme doc>", source="mem://a", user=alice)
    r.ingest(b"<globex doc>", source="mem://b", user=bob)
    a_resp = r.query("doc", user=alice)
    a_resp.verify()
    assert all(c.chunk.parent == "d-acme" for c in a_resp.citations)

def test_wordchunker_produces_valid_chunks():
    wc = WordChunker()
    chunks = wc.chunk_text("hello world " * 800, document_id="d1")
    for c in chunks: c.verify()  # all pass; no VerificationError

def test_reingest_dedup_by_checksum():
    r1 = r.ingest(b"abc", source_uri="mem://dup")
    r2 = r.ingest(b"abc", source_uri="mem://dup")
    assert r1.id == r2.id

def test_translate_preserves_bytes_round_trip():
    raw = b"Q3 revenue: 12% growth. " * 50
    r = RAG(converter=PlainTextConverter())
    r.ingest(raw, source="mem://tr")
    joined = b"".join(c.text.encode() for c in r.store.chunks)
    assert hashlib.sha256(joined).hexdigest() == hashlib.sha256(raw).hexdigest()  # lossless
```

`tests/test_security.py`:

```
def test_empty_question_is_caught_at_query_time():
    r = RAG(converter=PlainTextConverter())
    with pytest.raises(IngestionError, match="non-empty"):
        r.query("")
    assert r.last_error().type is ErrorKind.EMPTY

def test_invalid_input_fails_at_boundary_not_silently():
    bad = Chunk(id="c1", text="x", checksum="not_sha256")
    with pytest.raises(VerificationError, match="checksum"):
        bad.verify()
```

---

## Phase 5 — Coverage write-in

10 commits, one per module that needs coverage lift.

| # | Commit | Module | Coverage target |
| --- | --- | --- | --- |
| 5.1 | `tests/test_auth.py` (new) | `auth.py` (0% → ≥90%) | password hash, login, session, RBAC. |
| 5.2 | `tests/test_cli.py` (new) | `cli.py` (0% → ≥85%) | CLI commands, subcommands. |
| 5.3 | `tests/test_evaluation.py` extension | `evaluation.py` (0% → ≥85%) | CLI entry points. |
| 5.4 | `tests/test_api.py` extension | `api.py` (26% → ≥85%) | FastAPI exception handlers, route flows. |
| 5.5 | `tests/test_knowledge.py` extension | `knowledge.py` (25% → ≥80%) | GraphIndex, Manifest. |
| 5.6 | `tests/test_parsers.py` extension | `parsers.py` (25% → ≥80%) | Marker, plain text, MIME detection. |
| 5.7 | `tests/test_stores.py` extension | `stores/__init__.py` (27% → ≥85%) | JsonSessions, Documents. |
| 5.8 | `tests/test_repos.py` extension | `repos.py` (31% → ≥85%) | ChunkStore, DocStore, SessionStore. |
| 5.9 | `tests/test_retrieval.py` extension | `retrieval/__init__.py` (32% → ≥80%) | rerankers, transforms, fusion, search. |
| 5.10 | `tests/test_telemetry.py` etc. | telemetry, conv, prompts, plugins | each ≥80%. |

**Test contract:** each commit raises its module's coverage. `make coverage` fails until ≥85% globally.

---

## Phase 6 — Public surface + migration

10 commits.

| # | Commit | Files | Contract |
| --- | --- | --- | --- |
| 6.1 | Curated essentials in `__init__.py` | `raghub/__init__.py` | `RAG`, `Settings`, `Chunk`, `Document`, `Section`, `Block`, `Hit`, `Citation`, `Citations`, `Response`, `User`, `Session`, `Turn`, `Embedding`, `Bundle`, `PipelineResult`, `Job`, `JobStore`, `Cache`, `Router`, `Conversations`, `SlidingWindow`, `Memory`, `Store`, `Tokenizer`, `Tokenizer.DEFAULT_MODEL = "Qwen/Qwen3-8B"`, `Heuristic`, `Hasher`, `LiteLLM`, `LiteLLMEmbedder`, `LiteLLMProvider`, `Generator`, `Embedder`, `FrameIndex`, `WordChunker`, `Chonkie`, `Ingestor`, `RAG`, `RAPTOR`, `GraphIndex`, `Facade`, `Context`, `Result`, `Tool`, `Registry`, `Metrics`, `Telemetry`, `Loguru`, `Null`, `Noop`, `Prometheus`, `Langfuse`, `Manifest`. Single source. |
| 6.2 | Flat re-exports of all submodules | same | `from raghub.X import *` for every public submodule. |
| 6.3 | Drop dead `__all__` shims | `raghub/eval/`, etc. | no imported name missing. |
| 6.4 | `docs/migration.md` final | docs | old → new rename table; no aliases; schema migration steps. |
| 6.5 | `docs/quickstart.md` | docs | 10-line plugin-author example. |
| 6.6 | `docs/style/01-naming.md` | docs | the rules R1-R10 codified for contributors. |
| 6.7 | `docs/style/02-docstrings.md` | docs | Google-style spec. |
| 6.8 | `README.md` updated | `README.md` | quickstart matches new public surface. |
| 6.9 | `pyproject.toml` description update | `pyproject.toml` | "v1.0 OSS" notes. |
| 6.10 | `CHANGELOG.md` v1.0 entry | `CHANGELOG.md` | single breaking rename summary. |

---

## Phase 7 — v1.0 tag

10 commits of final hardening.

| # | Commit | What |
| --- | --- | --- |
| 7.1 | `make coverage` ≥85% gate enforced | CI gate. |
| 7.2 | `make docstrings` 100% enforced | CI gate. |
| 7.3 | `make naming` zero violations | CI gate. |
| 7.4 | `make migrate` dry-run on dev JSON | pre-tag verification. |
| 7.5 | `pip install -e .` clean install test | works on fresh venv. |
| 7.6 | `python -c "from raghub import RAG; print(RAG)"` smoke | import works. |
| 7.7 | `pytest --cov=raghub --cov-fail-under=85` green | coverage hits. |
| 7.8 | `ruff check`, `ruff format`, `interrogate`, `mypy` green | all lint. |
| 7.9 | `git tag -a v1.0.0 -m "OSS-ready release"` | tagging. |
| 7.10 | Tag push + release notes | finalisation. |

---

## Total commit count

| Phase | Commits |
| --- | --- |
| 0 | 10 |
| 1 | 20 |
| 1.5 | 7 |
| 1.7 | 20 |
| 1.8 | 1 (one-shot) |
| 1.9 | 10 |
| 1.10 | 10 |
| 2 | 10 |
| 3 | 10 |
| 4 | 20 |
| 5 | 10 |
| 6 | 10 |
| 7 | 10 |
| **Total** | **148** |

Each commit is reviewable. Each commit's "what must be true" is a list of concrete assertions, not a description.

---

## Validation patterns that every test must follow

These are not vague guidance — they are the acceptance bars:

### Correctness
A test passes only when the produced state exactly matches the expected state:
```
assert Chunk(id="x", text="y", checksum="z").verify() is None
assert Chunk(id="x", text="y", checksum="Z").verify() raises VerificationError(match="checksum")
```

### Accurate data translation
Round-trip tests for every transformation:
```
text → chunk.text (lossless via sha256 of joined chunks)
user input → bytes → chunks → embeddings → store → top-k hits → answer
```

### Behaviour
For any function with a side effect, the test exercises the side effect and asserts on the resulting state:
```
def run_pipeline_fail():
    pipeline.run(...) raises PipelineFailed with .step == "ingest"
    assert store.is_empty()
```

### Forbidden patterns

- `assert response is not None`
- `assert result.success`
- `monkeypatch.setattr(real_impl, "method", fake)`
- `Class.__new__(Class)`
- `cls.__func__.__globals__[...]`
- `_`-prefixed helper functions in production code
- `_`-prefixed module names
- `# noqa:`
- `from raghub.X import _`*` (leading underscore)

---

## Execution sequence

1. Confirm this plan is the source of truth.
2. Confirm `reports/inventory.md` is acceptable as the Phase 0 deliverable.
3. Begin Phase 0 commit 0.1.
4. Proceed commit-by-commit, "what must be true" = green.
5. Stop and ask if any commit's "what must be true" needs three iterations to satisfy.

**The plan is atomic. Each commit has a contract. No backward compat. No deprecation period. Forward only.**
