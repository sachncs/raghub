# Tier 3 — Make v0.7.7 feedback loops real (Items 16-20)

The v0.7.7 plan shipped `Feedback`, `FeedbackStore`, and two
`FeedbackScorer` algorithms — but the scorers' `boost` method is a
no-op stub, and no API endpoint accepts feedback. Tier 3 makes
the feature actually consumable.

---

## Item 16 — `Bm25BoostScorer.boost` no-op removed

- **File(s)**: `raghub/feedback/__init__.py`, `tests/test_feedback.py` (new file)
- **Change**: Either implement `boost` synchronously (cache the counts at construction time) or raise `NotImplementedError` explicitly with a pointer to `boost_async`. Pick one. Recommendation: cache at construction.
- **Test**: `tests/test_feedback.py::test_bm25_boost_scorer_boost_returns_modified_score`.
- **Acceptance criteria**:
  - R3 — single-word class name.
  - R8 — `verify()` at storage boundary.
  - T1, T3 — pass.
- **Success criteria**:
  - After construction with a `FeedbackStore` containing 3 positive feedback rows for `chunk_1`, calling `scorer.boost("chunk_1", 1.0)` returns `> 1.0`.
  - After construction with 2 negative feedback rows for `chunk_2`, calling `scorer.boost("chunk_2", 1.0)` returns `< 1.0`.

---

## Item 17 — `VectorDownWeightScorer.boost` no-op removed

- **File(s)**: `raghub/feedback/__init__.py`, `tests/test_feedback.py`
- **Change**: Same as item 16.
- **Test**: `tests/test_feedback.py::test_vector_down_weight_boost_multiplies_score`.
- **Acceptance criteria**:
  - R3, R8 — pass.
  - T1, T3 — pass.
- **Success criteria**:
  - For `chunk_x` with no feedback: `scorer.boost("chunk_x", 1.0) == 1.0`.
  - For `chunk_x` with negative feedback: `scorer.boost("chunk_x", 1.0) == 0.5` (the default factor).

---

## Item 18 — `FeedbackRouter` added to `RouteGroup`

- **File(s)**: `raghub/api.py`, `tests/test_api.py`
- **Change**: New `FeedbackRouter` class with `POST /feedback`, `GET /feedback`, `DELETE /feedback/{id}`, `GET /feedback/aggregate`. Composed into `RouteGroup` alongside the existing 6 routers.
- **Test**: `tests/test_api.py::test_feedback_router_post_get_delete_round_trip`, `::test_feedback_router_aggregate_returns_counts`.
- **Acceptance criteria**:
  - R4 — no shim; new router is added cleanly.
  - R8 — `verify()` on `Feedback` payload at the boundary.
  - T1, T3 — pass.
- **Success criteria**:
  - `POST /v1/feedback` with valid payload returns 201 and stores the record.
  - `GET /v1/feedback/{id}` retrieves it.
  - `DELETE /v1/feedback/{id}` returns 204.
  - `GET /v1/feedback/aggregate?tenant_id=alice` returns aggregate counts.

---

## Item 19 — `FeedbackStore` wired into `RAG.__init__`

- **File(s)**: `raghub/rag.py`, `tests/test_rag_facade.py`
- **Change**: When `Settings.feedback.backend == "sqlite"`, instantiate `SqliteFeedbackStore(settings.data_dir / "feedback.db")` and assign to `self.feedback_store_`. Skip when `backend == "none"`.
- **Test**: `tests/test_rag_facade.py::test_rag_constructs_feedback_store_from_settings`.
- **Acceptance criteria**:
  - C1 — `__init__` stays ≤ 30 LOC.
  - T1, T3 — pass.
- **Success criteria**:
  - `RAG(Settings(feedback=FeedbackConfig(backend="sqlite"))).feedback_store_` is a `SqliteFeedbackStore` instance.
  - `RAG().feedback_store_` is `None`.

---

## Item 20 — CLI `raghub feedback export`

- **File(s)**: `raghub/cli_commands/__init__.py`, `tests/test_cli.py`
- **Change**: New `FeedbackCommand` class with `export --jsonl <path>` and `stats --tenant <id>` sub-commands. Wire into the Typer app.
- **Test**: `tests/test_cli.py::test_feedback_export_writes_jsonl`, `::test_feedback_stats_returns_counts`.
- **Acceptance criteria**:
  - R3 — `FeedbackCommand` is single-word (no `Manager` / `Handler` suffix).
  - T3 — pass.
- **Success criteria**:
  - `raghub feedback export --jsonl /tmp/out.jsonl` writes one feedback record per line as JSON.
  - `raghub feedback stats --tenant alice` prints aggregate counts.

---

## Tier 3 acceptance gate

- `grep -n "boost" raghub/feedback/__init__.py | grep "return base_score"` returns empty (no no-op stubs).
- `POST /v1/feedback` and `GET /v1/feedback/aggregate` work end-to-end.
- `raghub feedback export` writes JSONL.
- `raghub feedback stats` returns counts.
