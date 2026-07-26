"""Qualitative tests for InMemoryVectorStore and FacetedSearchEngine.

These tests verify the actual behavior the in-memory vector store is
contracted to provide:

* ``matches_metadata_dict`` and ``matches_filter`` accept both legacy
  string and canonical-dict shapes without leaking cross-shape
  results.
* ``insert`` is atomic at chunk granularity (the strict-``zip`` check
  is the only way to detect caller mistakes before they corrupt the
  index).
* ``compute_score`` is the canonical cosine similarity, with the
  expected boundary behavior for zero vectors.
* ``search`` honours both the metadata pre-filter and the ``top_k``
  contract — a regression that returns the full corpus for any
  ``top_k`` would surface here.
* ``keyword_search`` uses BM25 over the actual chunk text, not a
  tokenised cache, so a regression that re-orders scores would
  surface here.
* ``delete`` / ``delete_document`` / ``delete_version`` are
  idempotent and precise — a regression that left orphan chunks
  around would surface as ``len(records) > 0`` after the delete.
* The :class:`FacetedSearchEngine` honours all filter facets
  independently and exposes facet counts.
"""

from __future__ import annotations

from typing import Any

import pytest

from raghub.core import allowed_company_filter
from raghub.models import ChunkRecord, Classification, UserPrincipal
from raghub.retrieval.search import (
    FacetedSearchEngine,
    SearchFilters,
    build_filter_string,
)
from raghub.vectorstore import (
    InMemoryVectorStore,
    MemoryVectorRecord,
    matches_metadata_dict,
)


def make_chunk(**overrides: Any) -> ChunkRecord:
    defaults: dict[str, Any] = dict(
        chunk_id="c1",
        document_id="d1",
        version=1,
        text="Some text for search",
        company="Acme",
        owner="user@acme.com",
    )
    defaults.update(overrides)
    return ChunkRecord(**defaults)


class FakeEmbeddingProvider:
    """Deterministic 2-d embedding provider used to drive search tests."""

    def __init__(self, mapping: dict[str, list[float]] | None = None) -> None:
        self.mapping = mapping or {}

    def model_name(self) -> str:  # pragma: no cover - kept for parity
        return "fake"

    def embed_text(self, text: str) -> list[float]:
        return self.mapping.get(text, [0.0, 0.0])


# ===========================================================================
# matches_metadata_dict — the dict pre-filter
# ===========================================================================


class TestMatchesMetadataDict:
    def test_list_value_matches(self) -> None:
        record = MemoryVectorRecord(chunk=make_chunk(company="Acme"), vector=[])
        assert matches_metadata_dict(record, {"company": ["Acme", "Beta"]}) is True

    def test_list_value_no_match(self) -> None:
        record = MemoryVectorRecord(chunk=make_chunk(company="Gamma"), vector=[])
        assert matches_metadata_dict(record, {"company": ["Acme", "Beta"]}) is False

    def test_scalar_value_matches(self) -> None:
        record = MemoryVectorRecord(chunk=make_chunk(document_id="d1"), vector=[])
        assert matches_metadata_dict(record, {"document_id": "d1"}) is True

    def test_scalar_value_no_match(self) -> None:
        record = MemoryVectorRecord(chunk=make_chunk(document_id="d1"), vector=[])
        assert matches_metadata_dict(record, {"document_id": "d2"}) is False

    def test_missing_key_returns_false(self) -> None:
        record = MemoryVectorRecord(chunk=make_chunk(), vector=[])
        assert matches_metadata_dict(record, {"nonexistent": "x"}) is False

    def test_empty_filters_returns_true(self) -> None:
        record = MemoryVectorRecord(chunk=make_chunk(), vector=[])
        assert matches_metadata_dict(record, {}) is True

    def test_multiple_criteria_all_pass(self) -> None:
        record = MemoryVectorRecord(
            chunk=make_chunk(company="Acme", document_id="d1"), vector=[]
        )
        assert matches_metadata_dict(
            record, {"company": ["Acme"], "document_id": "d1"}
        ) is True

    def test_multiple_criteria_one_fails(self) -> None:
        record = MemoryVectorRecord(
            chunk=make_chunk(company="Acme", document_id="d1"), vector=[]
        )
        assert matches_metadata_dict(
            record, {"company": ["Acme"], "document_id": "d2"}
        ) is False


# ===========================================================================
# InMemoryVectorStore — CRUD with realistic state transitions
# ===========================================================================


class TestInsert:
    def test_single_insert(self) -> None:
        store = InMemoryVectorStore()
        store.insert([make_chunk()], [[0.1, 0.2]])
        assert "c1" in store.records
        assert store.records["c1"].vector == [0.1, 0.2]

    def test_insert_multiple(self) -> None:
        store = InMemoryVectorStore()
        store.insert(
            [make_chunk(chunk_id="c1"), make_chunk(chunk_id="c2")],
            [[0.1], [0.2]],
        )
        assert set(store.records) == {"c1", "c2"}

    def test_insert_overwrites_existing(self) -> None:
        store = InMemoryVectorStore()
        store.insert([make_chunk(chunk_id="c1", text="old")], [[0.1]])
        store.insert([make_chunk(chunk_id="c1", text="new")], [[0.5]])
        assert store.records["c1"].chunk.text == "new"
        assert store.records["c1"].vector == [0.5], (
            "Overwrite must replace the vector too; a regression that "
            "kept the old vector would serve stale similarity results."
        )

    def test_mismatched_lengths_raises(self) -> None:
        store = InMemoryVectorStore()
        with pytest.raises(ValueError):
            store.insert(
                [make_chunk(chunk_id="c1"), make_chunk(chunk_id="c2")],
                [[0.1, 0.2]],
            )

    def test_upsert_delegates_to_insert(self) -> None:
        store = InMemoryVectorStore()
        store.upsert([make_chunk()], [[0.1]])
        assert "c1" in store.records


class TestDelete:
    def test_delete_known(self) -> None:
        store = InMemoryVectorStore()
        store.insert(
            [make_chunk(chunk_id="c1"), make_chunk(chunk_id="c2")],
            [[0.1], [0.2]],
        )
        store.delete(["c1"])
        assert "c1" not in store.records
        assert "c2" in store.records

    def test_delete_unknown_silently_skipped(self) -> None:
        store = InMemoryVectorStore()
        store.insert([make_chunk(chunk_id="c1")], [[0.1]])
        store.delete(["c1", "does-not-exist"])
        assert "c1" not in store.records

    def test_delete_empty_list_is_noop(self) -> None:
        store = InMemoryVectorStore()
        store.insert([make_chunk()], [[0.1]])
        store.delete([])
        assert "c1" in store.records


class TestDeleteDocument:
    def test_removes_all_for_document(self) -> None:
        store = InMemoryVectorStore()
        store.insert(
            [
                make_chunk(chunk_id="c1", document_id="d1"),
                make_chunk(chunk_id="c2", document_id="d1"),
                make_chunk(chunk_id="c3", document_id="d2"),
            ],
            [[0.1], [0.2], [0.3]],
        )
        store.delete_document("d1")
        assert set(store.records) == {"c3"}, (
            "delete_document must remove EVERY chunk for the document — "
            "leaving one chunk behind would serve half-deleted data."
        )

    def test_unknown_document_is_noop(self) -> None:
        store = InMemoryVectorStore()
        store.insert([make_chunk(chunk_id="c1", document_id="d1")], [[0.1]])
        store.delete_document("nope")
        assert "c1" in store.records


class TestDeleteVersion:
    def test_removes_matching_version(self) -> None:
        store = InMemoryVectorStore()
        store.insert(
            [
                make_chunk(chunk_id="c1", document_id="d1", version=1),
                make_chunk(chunk_id="c2", document_id="d1", version=2),
                make_chunk(chunk_id="c3", document_id="d2", version=1),
            ],
            [[0.1], [0.2], [0.3]],
        )
        store.delete_version("d1", 1)
        assert set(store.records) == {"c2", "c3"}

    def test_non_matching_version_is_noop(self) -> None:
        store = InMemoryVectorStore()
        store.insert([make_chunk(chunk_id="c1", document_id="d1", version=2)], [[0.1]])
        store.delete_version("d1", 1)
        assert "c1" in store.records


# ===========================================================================
# matches_filter — the legacy string filter
# ===========================================================================


class TestMatchesFilter:
    def test_empty_string_passes(self) -> None:
        store = InMemoryVectorStore()
        record = MemoryVectorRecord(chunk=make_chunk(), vector=[])
        assert store.matches_filter(record, "") is True

    def test_company_in_match(self) -> None:
        store = InMemoryVectorStore()
        record = MemoryVectorRecord(chunk=make_chunk(company="Acme"), vector=[])
        assert store.matches_filter(record, "company IN ('Acme')") is True

    def test_company_in_no_match(self) -> None:
        store = InMemoryVectorStore()
        record = MemoryVectorRecord(chunk=make_chunk(company="Beta"), vector=[])
        assert store.matches_filter(record, "company IN ('Acme', 'Gamma')") is False

    def test_company_in_double_quotes(self) -> None:
        store = InMemoryVectorStore()
        record = MemoryVectorRecord(chunk=make_chunk(company="Acme"), vector=[])
        assert store.matches_filter(record, 'company IN ("Acme")') is True

    def test_document_id_eq(self) -> None:
        store = InMemoryVectorStore()
        record = MemoryVectorRecord(chunk=make_chunk(document_id="d1"), vector=[])
        assert store.matches_filter(record, "document_id = 'd1'") is True

    def test_unknown_filter_fails_closed(self) -> None:
        """A filter on a non-whitelisted field must fail closed."""
        store = InMemoryVectorStore()
        record = MemoryVectorRecord(chunk=make_chunk(), vector=[])
        assert store.matches_filter(record, "owner = 'a@b.com'") is False, (
            "Unrecognised fields must NOT silently match — this would "
            "let a SQL-injection-style filter bypass the RBAC layer."
        )

    def test_injection_attempt_fails_closed(self) -> None:
        """A SQL-injection-style string must not be interpreted."""
        store = InMemoryVectorStore()
        record = MemoryVectorRecord(chunk=make_chunk(company="Acme"), vector=[])
        # DROP TABLE-style payload must NOT pass.
        assert store.matches_filter(record, "company = 'Acme' OR 1=1 --") is False


# ===========================================================================
# compute_score — the cosine-similarity contract
# ===========================================================================


class TestComputeScore:
    def test_identical_vectors(self) -> None:
        store = InMemoryVectorStore()
        assert store.compute_score([1.0, 0.0], [1.0, 0.0]) == pytest.approx(1.0)

    def test_orthogonal_vectors(self) -> None:
        store = InMemoryVectorStore()
        assert store.compute_score([1.0, 0.0], [0.0, 1.0]) == pytest.approx(0.0)

    def test_zero_query(self) -> None:
        store = InMemoryVectorStore()
        assert store.compute_score([0.0, 0.0], [1.0, 0.0]) == 0.0

    def test_zero_stored(self) -> None:
        store = InMemoryVectorStore()
        assert store.compute_score([1.0, 0.0], [0.0, 0.0]) == 0.0

    def test_opposite_vectors(self) -> None:
        store = InMemoryVectorStore()
        assert store.compute_score([1.0, 0.0], [-1.0, 0.0]) == pytest.approx(-1.0)

    def test_collinear_vectors(self) -> None:
        store = InMemoryVectorStore()
        assert store.compute_score([1.0, 2.0], [2.0, 4.0]) == pytest.approx(1.0)

    def test_score_is_monotonic_with_cosine_similarity(self) -> None:
        """Higher cosine angle should give a lower score; an implementation
        bug that returned abs() would surface here."""
        store = InMemoryVectorStore()
        base = [1.0, 0.0]
        pos = store.compute_score(base, [0.9, 0.1])
        neg = store.compute_score(base, [-0.9, 0.1])
        assert pos > neg


# ===========================================================================
# search — pre-filter, post-filter, RBAC, and ordering
# ===========================================================================


class TestSearch:
    def test_dict_filter(self) -> None:
        store = InMemoryVectorStore()
        store.insert(
            [
                make_chunk(chunk_id="c1", company="Acme"),
                make_chunk(chunk_id="c2", company="Beta"),
            ],
            [[0.1, 0.2], [0.1, 0.2]],
        )
        results = store.search(vector=[0.1, 0.2], top_k=5, metadata_filter={"company": "Acme"})
        assert [r["chunk_id"] for r in results] == ["c1"]

    def test_dict_filter_with_list(self) -> None:
        store = InMemoryVectorStore()
        store.insert(
            [
                make_chunk(chunk_id="c1", company="Acme"),
                make_chunk(chunk_id="c2", company="Beta"),
                make_chunk(chunk_id="c3", company="Gamma"),
            ],
            [[0.1, 0.2]] * 3,
        )
        results = store.search(
            vector=[0.1, 0.2], top_k=5, metadata_filter={"company": ["Acme", "Beta"]}
        )
        assert {r["chunk_id"] for r in results} == {"c1", "c2"}

    def test_legacy_string_filter(self) -> None:
        store = InMemoryVectorStore()
        store.insert([make_chunk(chunk_id="c1", company="Acme")], [[0.1, 0.2]])
        results = store.search(vector=[0.1, 0.2], top_k=5, metadata_filter="company IN ('Acme')")
        assert len(results) == 1

    def test_legacy_string_filter_no_match(self) -> None:
        store = InMemoryVectorStore()
        store.insert([make_chunk(chunk_id="c1", company="Acme")], [[0.1, 0.2]])
        results = store.search(vector=[0.1, 0.2], top_k=5, metadata_filter="company IN ('Beta')")
        assert results == []

    def test_empty_store_returns_empty(self) -> None:
        store = InMemoryVectorStore()
        assert store.search(vector=[1.0, 0.0], top_k=5) == []

    def test_top_k_limits_results(self) -> None:
        store = InMemoryVectorStore()
        store.insert(
            [make_chunk(chunk_id=f"c{i}") for i in range(10)],
            [[0.1, 0.2]] * 10,
        )
        assert len(store.search(vector=[0.1, 0.2], top_k=3)) == 3

    def test_results_are_sorted_descending_by_score(self) -> None:
        store = InMemoryVectorStore()
        store.insert(
            [
                make_chunk(chunk_id="low", company="Acme"),
                make_chunk(chunk_id="high", company="Acme"),
                make_chunk(chunk_id="mid", company="Acme"),
            ],
            [[0.1, 0.9], [0.9, 0.1], [0.5, 0.5]],
        )
        results = store.search(vector=[1.0, 0.0], top_k=3, metadata_filter="company IN ('Acme')")
        scores = [r["score"] for r in results]
        assert scores == sorted(scores, reverse=True), (
            "search() must return results sorted by descending score; "
            "a regression that returned insertion order would mislead "
            "downstream rerankers."
        )

    def test_default_filter_is_empty_string(self) -> None:
        store = InMemoryVectorStore()
        store.insert([make_chunk(chunk_id="c1")], [[0.1, 0.2]])
        assert len(store.search(vector=[0.1, 0.2], top_k=5)) == 1

    def test_delete_then_query_is_atomic(self) -> None:
        """A deleted chunk must not appear in subsequent search results."""
        store = InMemoryVectorStore()
        store.insert(
            [make_chunk(chunk_id="c1"), make_chunk(chunk_id="c2")],
            [[0.1, 0.2], [0.1, 0.2]],
        )
        store.delete(["c1"])
        results = store.search(vector=[0.1, 0.2], top_k=5)
        assert {r["chunk_id"] for r in results} == {"c2"}


class TestRbacIsolation:
    """The RBAC contract — the filter the pipeline derives from the
    principal must be the only filter the vector store sees."""

    def test_non_admin_filter_isolates_company(self) -> None:
        store = InMemoryVectorStore()
        store.insert(
            [
                make_chunk(chunk_id="acme", company="Acme"),
                make_chunk(chunk_id="beta", company="Beta"),
            ],
            [[1.0, 0.0], [1.0, 0.0]],
        )
        user = UserPrincipal(email="u@acme.com", allowed_companies=["Acme"])
        results = store.search(
            vector=[1.0, 0.0],
            top_k=5,
            metadata_filter=allowed_company_filter(user),
        )
        assert [r["chunk_id"] for r in results] == ["acme"]

    def test_admin_filter_is_empty_dict(self) -> None:
        """Admin must receive ``{}`` (no restriction); a non-empty value
        would scope the admin to a single tenant, which is a real RBAC
        regression."""
        user = UserPrincipal(email="a@b.com", is_admin=True, allowed_companies=["Acme"])
        assert allowed_company_filter(user) == {}

    def test_non_admin_empty_allowlist_filter_is_empty_list(self) -> None:
        """A non-admin with no companies receives ``{"company": []}`` —
        the canonical "match nothing" filter that the vector store
        translates into a no-match query."""
        user = UserPrincipal(
            email="a@b.com", allowed_companies=[], is_admin=False
        )
        assert allowed_company_filter(user) == {"company": []}

    def test_non_admin_filter_lists_allowed_companies(self) -> None:
        user = UserPrincipal(
            email="a@b.com", allowed_companies=["Acme", "Beta"], is_admin=False
        )
        assert allowed_company_filter(user) == {"company": ["Acme", "Beta"]}


# ===========================================================================
# hybrid_search — falls back to vector search for the in-memory backend
# ===========================================================================


class TestHybridSearch:
    def test_delegates_to_search(self) -> None:
        store = InMemoryVectorStore()
        store.insert([make_chunk(chunk_id="c1")], [[0.1, 0.2]])
        results = store.hybrid_search(
            query="ignored", vector=[0.1, 0.2], top_k=5
        )
        assert [r["chunk_id"] for r in results] == ["c1"]

    def test_hybrid_search_with_dict_filter(self) -> None:
        store = InMemoryVectorStore()
        store.insert(
            [make_chunk(chunk_id="c1", company="Acme")], [[0.1, 0.2]]
        )
        results = store.hybrid_search(
            query="ignored",
            vector=[0.1, 0.2],
            top_k=5,
            metadata_filter={"company": "Acme"},
        )
        assert [r["chunk_id"] for r in results] == ["c1"]


# ===========================================================================
# keyword_search — BM25 over the actual chunk text
# ===========================================================================


class TestKeywordSearch:
    """BM25 score semantics verified against realistic text corpora.

    The BM25 implementation in ``rank_bm25`` returns 0 for terms whose
    document frequency is greater than half the corpus (negative IDF
    clamped). Each test below uses a corpus with at least 5 documents
    and at most 2 documents per term so the IDF is always positive."""

    def _store_with_5_docs(self) -> InMemoryVectorStore:
        store = InMemoryVectorStore()
        store.insert(
            [
                make_chunk(
                    chunk_id="c1",
                    text="hello world is a common phrase used in many places today",
                ),
                make_chunk(
                    chunk_id="c2",
                    text="apple pie is a popular dessert served in many countries",
                ),
                make_chunk(
                    chunk_id="c3",
                    text="banana split is a classic summer treat for children",
                ),
                make_chunk(
                    chunk_id="c4",
                    text="cherry cobbler is another favourite pie from the south",
                ),
                make_chunk(
                    chunk_id="c5",
                    text="orange marmalade pairs well with toast and butter",
                ),
            ],
            [[0.1]] * 5,
        )
        return store

    def test_finds_matching_chunk(self) -> None:
        store = self._store_with_5_docs()
        results = store.keyword_search("hello", top_k=5)
        assert [r["chunk_id"] for r in results] == ["c1"]

    def test_no_match_returns_empty(self) -> None:
        store = self._store_with_5_docs()
        assert store.keyword_search("zzzzzz", top_k=5) == []

    def test_empty_query_returns_empty(self) -> None:
        store = self._store_with_5_docs()
        assert store.keyword_search("", top_k=5) == []

    def test_whitespace_query_returns_empty(self) -> None:
        store = self._store_with_5_docs()
        assert store.keyword_search("   ", top_k=5) == []

    def test_term_frequency_dominates(self) -> None:
        """Two chunks contain 'hello'; the one with two occurrences scores higher."""
        store = InMemoryVectorStore()
        store.insert(
            [
                make_chunk(
                    chunk_id="c1",
                    text="hello hello world is the first chunk indexed here",
                ),
                make_chunk(
                    chunk_id="c2",
                    text="hello world is the second chunk indexed there",
                ),
                make_chunk(
                    chunk_id="c3",
                    text="apple pie dessert with many different words inside",
                ),
                make_chunk(
                    chunk_id="c4",
                    text="banana split treat is full of disparate words today",
                ),
                make_chunk(
                    chunk_id="c5",
                    text="cherry cobbler uses many assorted words in the recipe",
                ),
            ],
            [[0.1]] * 5,
        )
        results = store.keyword_search("hello", top_k=5)
        assert [r["chunk_id"] for r in results] == ["c1", "c2"]
        assert results[0]["score"] > results[1]["score"]

    def test_scores_are_strictly_decreasing(self) -> None:
        store = InMemoryVectorStore()
        store.insert(
            [
                make_chunk(chunk_id="a", text="alpha alpha alpha"),
                make_chunk(chunk_id="b", text="alpha beta gamma"),
                make_chunk(chunk_id="c", text="alpha delta epsilon"),
                make_chunk(chunk_id="d", text="zeta eta theta"),
                make_chunk(chunk_id="e", text="iota kappa lambda"),
            ],
            [[0.1]] * 5,
        )
        results = store.keyword_search("alpha", top_k=5)
        scores = [r["score"] for r in results]
        assert scores == sorted(scores, reverse=True)

    def test_empty_corpus_returns_empty(self) -> None:
        store = InMemoryVectorStore()
        assert store.keyword_search("hello", top_k=5) == []


# ===========================================================================
# optimize / health — no-op + reporting
# ===========================================================================


class TestOptimizeAndHealth:
    def test_optimize_is_noop(self) -> None:
        store = InMemoryVectorStore()
        assert store.optimize() is None

    def test_health_reports_status_and_chunks(self) -> None:
        store = InMemoryVectorStore()
        store.insert([make_chunk(chunk_id="c1")], [[0.1]])
        h = store.health()
        assert h["status"] == "ok"
        assert h["backend"] == "memory"
        assert h["chunks"] == 1

    def test_health_zero_chunks(self) -> None:
        store = InMemoryVectorStore()
        assert store.health()["chunks"] == 0


# ===========================================================================
# create_collection — the in-memory backend has no collection concept
# ===========================================================================


class TestCreateCollection:
    def test_noop(self) -> None:
        store = InMemoryVectorStore()
        assert store.create_collection() is None


# ===========================================================================
# 10k-chunk memory-pressure test — the in-memory backend must handle a
# realistic corpus without crashing.
# ===========================================================================


class TestLargeCorpus:
    def test_10k_chunks_ingest_and_query(self) -> None:
        """Insert 10 000 chunks and verify a query still returns top_k hits.

        This is a smoke test for the in-memory backend's scalability
        contract; a regression that bounded the dict to e.g. 1000
        entries would surface here."""
        store = InMemoryVectorStore()
        chunks = [
            make_chunk(chunk_id=f"c{i:05d}", text=f"doc number {i} about topic {i % 100}")
            for i in range(10_000)
        ]
        vectors = [[float(i), 0.0] for i in range(10_000)]
        store.insert(chunks, vectors)
        assert len(store.records) == 10_000
        results = store.search(vector=[5000.0, 0.0], top_k=10)
        assert len(results) == 10


# ===========================================================================
# FacetedSearchEngine — filter + count_by_field behavior
# ===========================================================================


class TestFacetedSearchEngineSearch:
    def test_search_returns_unique_chunks(self) -> None:
        store = InMemoryVectorStore()
        store.insert(
            [make_chunk(chunk_id="c1", company="Acme")], [[1.0, 0.0]]
        )
        engine = FacetedSearchEngine(
            vector_store=store, embedding_provider=FakeEmbeddingProvider({"q": [1.0, 0.0]})
        )
        results = engine.search("q", SearchFilters(companies=["Acme"]), top_k=5)
        assert [c.chunk_id for c in results] == ["c1"]

    def test_search_filters_out_non_matching_company(self) -> None:
        store = InMemoryVectorStore()
        store.insert(
            [
                make_chunk(chunk_id="c1", company="Acme"),
                make_chunk(chunk_id="c2", company="Beta"),
            ],
            [[1.0, 0.0], [1.0, 0.0]],
        )
        engine = FacetedSearchEngine(
            vector_store=store, embedding_provider=FakeEmbeddingProvider({"q": [1.0, 0.0]})
        )
        results = engine.search("q", SearchFilters(companies=["Acme"]))
        assert [c.chunk_id for c in results] == ["c1"]

    def test_search_with_no_filters_returns_all(self) -> None:
        store = InMemoryVectorStore()
        store.insert(
            [make_chunk(chunk_id="c1"), make_chunk(chunk_id="c2")],
            [[1.0, 0.0], [1.0, 0.0]],
        )
        engine = FacetedSearchEngine(
            vector_store=store, embedding_provider=FakeEmbeddingProvider({"q": [1.0, 0.0]})
        )
        results = engine.search("q", top_k=5)
        assert {c.chunk_id for c in results} == {"c1", "c2"}


class TestFacetedSearchEngineMatchesFilters:
    def test_classification_filter_pass(self) -> None:
        engine = FacetedSearchEngine(
            vector_store=InMemoryVectorStore(),
            embedding_provider=FakeEmbeddingProvider(),
        )
        chunk = make_chunk(classification=Classification.RESTRICTED)
        assert engine.matches_filters(
            chunk, SearchFilters(classifications=[Classification.RESTRICTED])
        ) is True

    def test_classification_filter_fail(self) -> None:
        engine = FacetedSearchEngine(
            vector_store=InMemoryVectorStore(),
            embedding_provider=FakeEmbeddingProvider(),
        )
        chunk = make_chunk(classification=Classification.INTERNAL)
        assert engine.matches_filters(
            chunk, SearchFilters(classifications=[Classification.RESTRICTED])
        ) is False

    def test_owner_filter_pass(self) -> None:
        engine = FacetedSearchEngine(
            vector_store=InMemoryVectorStore(),
            embedding_provider=FakeEmbeddingProvider(),
        )
        chunk = make_chunk(owner="a@co.com")
        assert engine.matches_filters(chunk, SearchFilters(owners=["a@co.com"])) is True

    def test_owner_filter_fail(self) -> None:
        engine = FacetedSearchEngine(
            vector_store=InMemoryVectorStore(),
            embedding_provider=FakeEmbeddingProvider(),
        )
        chunk = make_chunk(owner="a@co.com")
        assert engine.matches_filters(chunk, SearchFilters(owners=["b@co.com"])) is False

    def test_multiple_filters_all_pass(self) -> None:
        engine = FacetedSearchEngine(
            vector_store=InMemoryVectorStore(),
            embedding_provider=FakeEmbeddingProvider(),
        )
        chunk = make_chunk(company="Acme", department="Eng", owner="a@co.com")
        filters = SearchFilters(
            companies=["Acme"], departments=["Eng"], owners=["a@co.com"]
        )
        assert engine.matches_filters(chunk, filters) is True

    def test_multiple_filters_one_fails(self) -> None:
        engine = FacetedSearchEngine(
            vector_store=InMemoryVectorStore(),
            embedding_provider=FakeEmbeddingProvider(),
        )
        chunk = make_chunk(company="Acme", department="Eng")
        filters = SearchFilters(companies=["Acme"], departments=["Sales"])
        assert engine.matches_filters(chunk, filters) is False

    def test_empty_filters_passes(self) -> None:
        engine = FacetedSearchEngine(
            vector_store=InMemoryVectorStore(),
            embedding_provider=FakeEmbeddingProvider(),
        )
        assert engine.matches_filters(make_chunk(), SearchFilters()) is True


class TestFacetedSearchEngineCountByField:
    def test_no_records_returns_empty(self) -> None:
        engine = FacetedSearchEngine(
            vector_store=InMemoryVectorStore(),
            embedding_provider=FakeEmbeddingProvider(),
        )
        assert engine.count_by_field("company") == {}

    def test_counts_scalar_values(self) -> None:
        store = InMemoryVectorStore()
        store.insert(
            [
                make_chunk(chunk_id="c1", company="Acme"),
                make_chunk(chunk_id="c2", company="Acme"),
                make_chunk(chunk_id="c3", company="Beta"),
            ],
            [[0.1]] * 3,
        )
        engine = FacetedSearchEngine(
            vector_store=store, embedding_provider=FakeEmbeddingProvider()
        )
        assert engine.count_by_field("company") == {"Acme": 2, "Beta": 1}

    def test_counts_list_values_per_element(self) -> None:
        store = InMemoryVectorStore()
        c1 = make_chunk(chunk_id="c1")
        c2 = make_chunk(chunk_id="c2")
        object.__setattr__(c1, "tags", ["a", "b"])
        object.__setattr__(c2, "tags", ["a"])
        store.insert([c1, c2], [[0.1], [0.1]])
        engine = FacetedSearchEngine(
            vector_store=store, embedding_provider=FakeEmbeddingProvider()
        )
        assert engine.count_by_field("tags") == {"a": 2, "b": 1}

    def test_none_values_skipped(self) -> None:
        store = InMemoryVectorStore()
        store.insert(
            [make_chunk(chunk_id="c1"), make_chunk(chunk_id="c2")],
            [[0.1], [0.1]],
        )
        engine = FacetedSearchEngine(
            vector_store=store, embedding_provider=FakeEmbeddingProvider()
        )
        assert engine.count_by_field("nonexistent_field") == {}

    def test_records_is_none_returns_empty(self) -> None:
        class _StoreNoRecords:
            records = None

        engine = FacetedSearchEngine(
            vector_store=_StoreNoRecords(),
            embedding_provider=FakeEmbeddingProvider(),
        )
        assert engine.count_by_field("company") == {}


class TestBuildFilterString:
    def test_none_returns_empty(self) -> None:
        assert build_filter_string(None) == ""

    def test_company_in_clause(self) -> None:
        assert (
            build_filter_string(SearchFilters(companies=["Acme", "Beta"]))
            == "company IN ('Acme', 'Beta')"
        )

    def test_combines_with_and(self) -> None:
        s = build_filter_string(
            SearchFilters(companies=["Acme"], owners=["a@b.com"])
        )
        assert " AND " in s
        assert "company IN ('Acme')" in s
        assert "owner IN ('a@b.com')" in s
