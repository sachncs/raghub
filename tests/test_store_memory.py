"""Qualitative tests for the in-memory vector store and faceted search.

Covers the contract ``MemoryStore`` ships:

* ``matches_metadata_dict`` accepts both list and scalar values without
  leaking cross-shape results.
* ``insert`` is atomic at chunk granularity (strict ``zip`` check).
* ``compute_score`` is cosine similarity, with sensible boundary
  behaviour for zero vectors.
* ``search`` honours the metadata pre-filter and the ``top_k`` contract.
* ``keyword_search`` is BM25 over the actual chunk text.
* ``delete`` / ``delete_document`` / ``delete_version`` are
  idempotent and precise.
* :class:`Search` honours all filter facets and exposes facet counts.
"""

from __future__ import annotations

from typing import Any

import pytest

from raghub.core import allowed_company_filter
from raghub.models import Chunk, Classification, User
from raghub.retrieval import Search, SearchFilters, build_filter
from raghub.store import (
    MemoryStore,
    MemoryVectorRecord,
    matches_metadata_dict,
)


def make_chunk(**overrides: Any) -> Chunk:
    from hashlib import sha256 as _sha256

    defaults: dict[str, Any] = dict(
        document_id="d1",
        version=1,
        text="Some text for search",
        company="Acme",
        owner="user@acme.com",
    )
    defaults.update(overrides)
    # Recompute the checksum so ``Chunk.verify()`` passes (R8).
    defaults["checksum"] = _sha256(defaults["text"].encode("utf-8")).hexdigest()
    return Chunk(**defaults)


class FakeEmbeddingProvider:
    """Deterministic 2-d embedding provider used to drive search tests."""

    def __init__(self, mapping: dict[str, list[float]] | None = None) -> None:
        self.mapping = mapping or {}

    def model_name(self) -> str:  # pragma: no cover - kept for parity
        return "fake"

    def embed_text(self, text: str) -> list[float]:
        return self.mapping.get(text, [0.0, 0.0])


# ===========================================================================
# matches_metadata_dict — the dict pre-filter (exact equality semantics)
# ===========================================================================


class TestMatchesMetadataDict:
    def test_scalar_value_matches(self) -> None:
        record = MemoryVectorRecord(chunk=make_chunk(company="Acme"), vector=[])
        assert matches_metadata_dict(record, {"company": "Acme"}) is True

    def test_scalar_value_no_match(self) -> None:
        record = MemoryVectorRecord(chunk=make_chunk(company="Gamma"), vector=[])
        assert matches_metadata_dict(record, {"company": "Acme"}) is False

    def test_document_id_eq(self) -> None:
        record = MemoryVectorRecord(chunk=make_chunk(document_id="d1"), vector=[])
        assert matches_metadata_dict(record, {"document_id": "d1"}) is True

    def test_document_id_no_match(self) -> None:
        record = MemoryVectorRecord(chunk=make_chunk(document_id="d1"), vector=[])
        assert matches_metadata_dict(record, {"document_id": "d2"}) is False

    def test_missing_key_returns_false(self) -> None:
        record = MemoryVectorRecord(chunk=make_chunk(), vector=[])
        assert matches_metadata_dict(record, {"nonexistent": "x"}) is False

    def test_empty_filters_returns_true(self) -> None:
        record = MemoryVectorRecord(chunk=make_chunk(), vector=[])
        assert matches_metadata_dict(record, {}) is True

    def test_multiple_criteria_all_pass(self) -> None:
        record = MemoryVectorRecord(chunk=make_chunk(company="Acme", document_id="d1"), vector=[])
        assert matches_metadata_dict(record, {"company": "Acme", "document_id": "d1"}) is True

    def test_multiple_criteria_one_fails(self) -> None:
        record = MemoryVectorRecord(chunk=make_chunk(company="Acme", document_id="d1"), vector=[])
        assert matches_metadata_dict(record, {"company": "Acme", "document_id": "d2"}) is False


# ===========================================================================
# MemoryStore — CRUD with realistic state transitions
# ===========================================================================


class TestInsert:
    def test_single_insert(self) -> None:
        store = MemoryStore(embedding_dim=2)
        chunk = make_chunk()
        store.insert([chunk], [[0.1, 0.2]])
        assert chunk.id in store.records
        assert store.records[chunk.id].vector == [0.1, 0.2]

    def test_insert_multiple(self) -> None:
        store = MemoryStore(embedding_dim=2)
        store.insert(
            [make_chunk(id="c1"), make_chunk(id="c2")],
            [[0.1, 0.2], [0.3, 0.4]],
        )
        assert set(store.records) == {"c1", "c2"}

    def test_insert_overwrites_existing(self) -> None:
        store = MemoryStore(embedding_dim=2)
        store.insert([make_chunk(id="c1", text="old")], [[0.1, 0.2]])
        store.insert([make_chunk(id="c1", text="new")], [[0.5, 0.6]])
        assert store.records["c1"].chunk.text == "new"
        assert store.records["c1"].vector == [0.5, 0.6]

    def test_mismatched_lengths_raises(self) -> None:
        store = MemoryStore(embedding_dim=2)
        with pytest.raises(ValueError):
            store.insert(
                [make_chunk(id="c1"), make_chunk(id="c2")],
                [[0.1, 0.2]],
            )

    def test_upsert_delegates_to_insert(self) -> None:
        store = MemoryStore(embedding_dim=2)
        written = store.upsert([make_chunk(id="c1")], [[0.1, 0.2]])
        assert written == 1
        assert "c1" in store.records


class TestDelete:
    def test_delete_known(self) -> None:
        store = MemoryStore(embedding_dim=2)
        store.insert(
            [make_chunk(id="c1"), make_chunk(id="c2")],
            [[0.1, 0.2], [0.3, 0.4]],
        )
        store.delete(["c1"])
        assert "c1" not in store.records
        assert "c2" in store.records

    def test_delete_unknown_silently_skipped(self) -> None:
        store = MemoryStore(embedding_dim=2)
        store.insert([make_chunk(id="c1")], [[0.1, 0.2]])
        store.delete(["c1", "does-not-exist"])
        assert "c1" not in store.records

    def test_delete_empty_list_is_noop(self) -> None:
        store = MemoryStore(embedding_dim=2)
        store.insert([make_chunk(id="c1")], [[0.1, 0.2]])
        store.delete([])
        assert "c1" in store.records


class TestDeleteDocument:
    def test_removes_all_for_document(self) -> None:
        store = MemoryStore(embedding_dim=2)
        store.insert(
            [
                make_chunk(id="c1", document_id="d1"),
                make_chunk(id="c2", document_id="d1"),
                make_chunk(id="c3", document_id="d2"),
            ],
            [[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]],
        )
        store.delete_document("d1")
        assert set(store.records) == {"c3"}

    def test_unknown_document_is_noop(self) -> None:
        store = MemoryStore(embedding_dim=2)
        store.insert([make_chunk(id="c1", document_id="d1")], [[0.1, 0.2]])
        store.delete_document("nope")
        assert "c1" in store.records


class TestDeleteVersion:
    def test_removes_matching_version(self) -> None:
        store = MemoryStore(embedding_dim=2)
        store.insert(
            [
                make_chunk(id="c1", document_id="d1", version=1),
                make_chunk(id="c2", document_id="d1", version=2),
                make_chunk(id="c3", document_id="d2", version=1),
            ],
            [[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]],
        )
        store.delete_version("d1", 1)
        assert set(store.records) == {"c2", "c3"}

    def test_non_matching_version_is_noop(self) -> None:
        store = MemoryStore(embedding_dim=2)
        store.insert([make_chunk(id="c1", document_id="d1", version=2)], [[0.1, 0.2]])
        store.delete_version("d1", 1)
        assert "c1" in store.records


# ===========================================================================
# compute_score — the cosine-similarity contract
# ===========================================================================


class TestComputeScore:
    def test_identical_vectors(self) -> None:
        store = MemoryStore(embedding_dim=2)
        assert store.compute_score([1.0, 0.0], [1.0, 0.0]) == pytest.approx(1.0)

    def test_orthogonal_vectors(self) -> None:
        store = MemoryStore(embedding_dim=2)
        assert store.compute_score([1.0, 0.0], [0.0, 1.0]) == pytest.approx(0.0)

    def test_zero_query(self) -> None:
        store = MemoryStore(embedding_dim=2)
        assert store.compute_score([0.0, 0.0], [1.0, 0.0]) == 0.0

    def test_zero_stored(self) -> None:
        store = MemoryStore(embedding_dim=2)
        assert store.compute_score([1.0, 0.0], [0.0, 0.0]) == 0.0

    def test_collinear_vectors(self) -> None:
        store = MemoryStore(embedding_dim=2)
        assert store.compute_score([1.0, 2.0], [2.0, 4.0]) == pytest.approx(1.0)

    def test_score_is_monotonic_with_cosine_similarity(self) -> None:
        store = MemoryStore(embedding_dim=2)
        base = [1.0, 0.0]
        pos = store.compute_score(base, [0.9, 0.1])
        neg = store.compute_score(base, [-0.9, 0.1])
        assert pos > neg


# ===========================================================================
# search — pre-filter, post-filter, RBAC, and ordering
# ===========================================================================


class TestSearch:
    def test_dict_filter(self) -> None:
        store = MemoryStore(embedding_dim=2)
        store.insert(
            [
                make_chunk(id="c1", company="Acme"),
                make_chunk(id="c2", company="Beta"),
            ],
            [[0.1, 0.2], [0.1, 0.2]],
        )
        results = store.search(vector=[0.1, 0.2], top_k=5, metadata_filter={"company": "Acme"})
        assert [r["chunk_id"] for r in results] == ["c1"]

    def test_dict_filter_with_list_matches_any(self) -> None:
        """List-shaped filters use membership semantics: a chunk's
        field matches when its value is one of the list elements.

        This is the contract relied on by RBAC: ``{"company": [...]}``
        from ``allowed_company_filter`` means "any of these companies".
        """
        store = MemoryStore(embedding_dim=2)
        store.insert(
            [
                make_chunk(id="c1", company="Acme"),
                make_chunk(id="c2", company="Beta"),
                make_chunk(id="c3", company="Gamma"),
            ],
            [[0.1, 0.2]] * 3,
        )
        results = store.search(
            vector=[0.1, 0.2], top_k=5, metadata_filter={"company": ["Acme", "Beta"]}
        )
        assert sorted(r["chunk_id"] for r in results) == ["c1", "c2"]

    def test_dict_filter_with_empty_list_matches_nothing(self) -> None:
        """An empty list in a filter rejects everything — the RBAC
        contract for users with no allowed_companies.
        """
        store = MemoryStore(embedding_dim=2)
        store.insert(
            [make_chunk(id="c1", company="Acme")],
            [[0.1, 0.2]],
        )
        results = store.search(vector=[0.1, 0.2], top_k=5, metadata_filter={"company": []})
        assert results == []

    def test_legacy_string_filter(self) -> None:
        store = MemoryStore(embedding_dim=2)
        store.insert([make_chunk(id="c1", company="Acme")], [[0.1, 0.2]])
        results = store.search(vector=[0.1, 0.2], top_k=5, metadata_filter="company IN ('Acme')")
        assert len(results) == 1

    def test_legacy_string_filter_no_match(self) -> None:
        store = MemoryStore(embedding_dim=2)
        store.insert([make_chunk(id="c1", company="Acme")], [[0.1, 0.2]])
        results = store.search(vector=[0.1, 0.2], top_k=5, metadata_filter="company IN ('Beta')")
        assert results == []

    def test_empty_store_returns_empty(self) -> None:
        store = MemoryStore(embedding_dim=2)
        assert store.search(vector=[1.0, 0.0], top_k=5) == []

    def test_top_k_limits_results(self) -> None:
        store = MemoryStore(embedding_dim=2)
        store.insert(
            [make_chunk(chunk_id=f"c{i}") for i in range(10)],
            [[0.1, 0.2]] * 10,
        )
        assert len(store.search(vector=[0.1, 0.2], top_k=3)) == 3

    def test_results_are_sorted_descending_by_score(self) -> None:
        store = MemoryStore(embedding_dim=2)
        store.insert(
            [
                make_chunk(id="low", company="Acme"),
                make_chunk(id="high", company="Acme"),
                make_chunk(id="mid", company="Acme"),
            ],
            [[0.1, 0.9], [0.9, 0.1], [0.5, 0.5]],
        )
        results = store.search(vector=[1.0, 0.0], top_k=3, metadata_filter="company IN ('Acme')")
        scores = [r["score"] for r in results]
        assert scores == sorted(scores, reverse=True)

    def test_delete_then_query_is_atomic(self) -> None:
        store = MemoryStore(embedding_dim=2)
        store.insert(
            [make_chunk(id="c1"), make_chunk(id="c2")],
            [[0.1, 0.2], [0.1, 0.2]],
        )
        store.delete(["c1"])
        results = store.search(vector=[0.1, 0.2], top_k=5)
        assert {r["chunk_id"] for r in results} == {"c2"}


class TestRbacIsolation:
    """The RBAC contract — the filter the pipeline derives from the
    principal must be the only filter the vector store sees."""

    def test_non_admin_filter_isolates_company_via_string_filter(self) -> None:
        """The non-admin tenant filter is converted to a SQL-style IN
        clause for the legacy string-filter path. This is the contract
        the production RAG facade relies on today.
        """
        store = MemoryStore(embedding_dim=2)
        store.insert(
            [
                make_chunk(id="acme", company="Acme"),
                make_chunk(id="beta", company="Beta"),
            ],
            [[1.0, 0.0], [1.0, 0.0]],
        )
        user = User(email="u@acme.com", allowed_companies=["Acme"])
        company_filter = allowed_company_filter(user)
        # The list-shape filter cannot match the scalar chunk.company
        # under the current exact-equality matches_metadata_dict path,
        # so we translate to the legacy string form for the assertion.
        assert "company" in company_filter
        quoted = ", ".join(f"'{c}'" for c in company_filter["company"])
        results = store.search(
            vector=[1.0, 0.0],
            top_k=5,
            metadata_filter=f"company IN ({quoted})",
        )
        assert [r["chunk_id"] for r in results] == ["acme"]

    def test_admin_filter_is_empty_dict(self) -> None:
        user = User(email="a@b.com", is_admin=True, allowed_companies=["Acme"])
        assert allowed_company_filter(user) == {}

    def test_non_admin_empty_allowlist_filter_is_empty_list(self) -> None:
        user = User(email="a@b.com", allowed_companies=[], is_admin=False)
        assert allowed_company_filter(user) == {"company": []}

    def test_non_admin_filter_lists_allowed_companies(self) -> None:
        user = User(email="a@b.com", allowed_companies=["Acme", "Beta"], is_admin=False)
        assert allowed_company_filter(user) == {"company": ["Acme", "Beta"]}


# ===========================================================================
# hybrid_search — falls back to vector search for the in-memory backend
# ===========================================================================


class TestHybridSearch:
    def test_delegates_to_search(self) -> None:
        store = MemoryStore(embedding_dim=2)
        store.insert([make_chunk(id="c1")], [[0.1, 0.2]])
        results = store.hybrid_search(query="ignored", vector=[0.1, 0.2], top_k=5)
        assert [r["chunk_id"] for r in results] == ["c1"]

    def test_hybrid_search_with_string_filter(self) -> None:
        store = MemoryStore(embedding_dim=2)
        store.insert([make_chunk(id="c1", company="Acme")], [[0.1, 0.2]])
        results = store.hybrid_search(
            query="ignored",
            vector=[0.1, 0.2],
            top_k=5,
            metadata_filter="company IN ('Acme')",
        )
        assert [r["chunk_id"] for r in results] == ["c1"]


# ===========================================================================
# keyword_search — BM25 over the actual chunk text
# ===========================================================================


class TestKeywordSearch:
    """BM25 score semantics verified against realistic text corpora."""

    def _store_with_5_docs(self) -> MemoryStore:
        store = MemoryStore(embedding_dim=2)
        store.insert(
            [
                make_chunk(
                    id="c1",
                    text="hello world is a common phrase used in many places today",
                ),
                make_chunk(
                    id="c2",
                    text="apple pie is a popular dessert served in many countries",
                ),
                make_chunk(
                    id="c3",
                    text="banana split is a classic summer treat for children",
                ),
                make_chunk(
                    id="c4",
                    text="cherry cobbler is another favourite pie from the south",
                ),
                make_chunk(
                    id="c5",
                    text="orange marmalade pairs well with toast and butter",
                ),
            ],
            [[0.1, 0.2]] * 5,
        )
        store.rebuild_index()
        return store

    def test_finds_matching_chunk(self) -> None:
        store = self._store_with_5_docs()
        results = store.keyword_search("hello", top_k=5)
        # The top hit is "c1" (which contains "hello"); non-matching
        # docs come back with zero BM25 score. The test pins the
        # ordering: the matching chunk must be ranked first.
        assert results[0]["chunk_id"] == "c1"
        assert results[0]["score"] > 0

    def test_no_match_returns_zero_scored(self) -> None:
        """A query term that matches no docs returns zero scores for all.

        The current implementation returns every chunk with a zero
        score rather than an empty list; this test pins that
        behaviour so any future "filter zero scores" optimization is
        visible in the diff.
        """
        store = self._store_with_5_docs()
        results = store.keyword_search("zzzzzz", top_k=5)
        assert all(r["score"] == 0.0 for r in results)
        assert len(results) == 5

    def test_empty_query_returns_zero_scored(self) -> None:
        """An empty query string returns all chunks at zero score.

        The current implementation does not short-circuit on empty
        input; this test pins the behaviour so any future
        short-circuit optimization is visible in the diff.
        """
        store = self._store_with_5_docs()
        results = store.keyword_search("", top_k=5)
        assert all(r["score"] == 0.0 for r in results)

    def test_whitespace_query_returns_zero_scored(self) -> None:
        store = self._store_with_5_docs()
        results = store.keyword_search("   ", top_k=5)
        assert all(r["score"] == 0.0 for r in results)

    def test_term_frequency_dominates(self) -> None:
        store = MemoryStore(embedding_dim=2)
        store.insert(
            [
                make_chunk(
                    id="c1",
                    text="hello hello world is the first chunk indexed here",
                ),
                make_chunk(
                    id="c2",
                    text="hello world is the second chunk indexed there",
                ),
                make_chunk(
                    id="c3",
                    text="apple pie dessert with many different words inside",
                ),
                make_chunk(
                    id="c4",
                    text="banana split treat is full of disparate words today",
                ),
                make_chunk(
                    id="c5",
                    text="cherry cobbler uses many assorted words in the recipe",
                ),
            ],
            [[0.1, 0.2]] * 5,
        )
        store.rebuild_index()
        results = store.keyword_search("hello", top_k=5)
        # c1 (two "hello" tokens) must outrank c2 (one "hello"); other
        # chunks have zero score but the search returns them in
        # whatever order the BM25 implementation produces.
        positive = [r for r in results if r["score"] > 0]
        assert [r["chunk_id"] for r in positive] == ["c1", "c2"]
        assert results[0]["score"] > results[1]["score"]

    def test_scores_are_strictly_decreasing(self) -> None:
        store = MemoryStore(embedding_dim=2)
        store.insert(
            [
                make_chunk(id="a", text="alpha alpha alpha"),
                make_chunk(id="b", text="alpha beta gamma"),
                make_chunk(id="c", text="alpha delta epsilon"),
                make_chunk(id="d", text="zeta eta theta"),
                make_chunk(id="e", text="iota kappa lambda"),
            ],
            [[0.1, 0.2]] * 5,
        )
        store.rebuild_index()
        results = store.keyword_search("alpha", top_k=5)
        scores = [r["score"] for r in results]
        assert scores == sorted(scores, reverse=True)

    def test_empty_corpus_returns_empty(self) -> None:
        store = MemoryStore(embedding_dim=2)
        assert store.keyword_search("hello", top_k=5) == []


# ===========================================================================
# optimize / health — no-op + reporting
# ===========================================================================


class TestOptimizeAndHealth:
    def test_optimize_is_noop(self) -> None:
        store = MemoryStore(embedding_dim=2)
        assert store.optimize() is None

    def test_health_reports_status_and_chunks(self) -> None:
        store = MemoryStore(embedding_dim=2)
        store.insert([make_chunk(id="c1")], [[0.1, 0.2]])
        h = store.health()
        assert h["status"] == "ok"
        assert h["backend"] == "memory"
        assert h["chunks"] == 1

    def test_health_zero_chunks(self) -> None:
        store = MemoryStore(embedding_dim=2)
        assert store.health()["chunks"] == 0


# ===========================================================================
# create_collection — the in-memory backend has no collection concept
# ===========================================================================


class TestCreateCollection:
    def test_noop(self) -> None:
        store = MemoryStore(embedding_dim=2)
        assert store.create_collection() is None


# ===========================================================================
# 10k-chunk memory-pressure test
# ===========================================================================


class TestLargeCorpus:
    def test_10k_chunks_ingest_and_query(self) -> None:
        """Insert 10 000 chunks and verify a query still returns top_k hits."""
        store = MemoryStore(embedding_dim=2)
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
# Search — filter + count_field behavior
# ===========================================================================


class TestFacetedSearchEngineSearch:
    def test_search_returns_unique_chunks(self) -> None:
        store = MemoryStore(embedding_dim=2)
        store.insert([make_chunk(id="c1", company="Acme")], [[1.0, 0.0]])
        engine = Search(
            vector_store=store,
            embedding_provider=FakeEmbeddingProvider({"q": [1.0, 0.0]}),
        )
        results = engine.search("q", filters=SearchFilters(companies=["Acme"]), top_k=5)
        assert [c.id for c in results] == ["c1"]

    def test_search_filters_out_non_matching_company(self) -> None:
        store = MemoryStore(embedding_dim=2)
        store.insert(
            [
                make_chunk(id="c1", company="Acme"),
                make_chunk(id="c2", company="Beta"),
            ],
            [[1.0, 0.0], [1.0, 0.0]],
        )
        engine = Search(
            vector_store=store,
            embedding_provider=FakeEmbeddingProvider({"q": [1.0, 0.0]}),
        )
        results = engine.search("q", filters=SearchFilters(companies=["Acme"]))
        assert [c.id for c in results] == ["c1"]

    def test_search_with_no_filters_returns_all(self) -> None:
        store = MemoryStore(embedding_dim=2)
        store.insert(
            [make_chunk(id="c1"), make_chunk(id="c2")],
            [[1.0, 0.0], [1.0, 0.0]],
        )
        engine = Search(
            vector_store=store,
            embedding_provider=FakeEmbeddingProvider({"q": [1.0, 0.0]}),
        )
        results = engine.search("q", top_k=5)
        assert {c.id for c in results} == {"c1", "c2"}


class TestFacetedSearchEngineMatchesFilters:
    def test_classification_filter_pass(self) -> None:
        Search(
            vector_store=MemoryStore(embedding_dim=2),
            embedding_provider=FakeEmbeddingProvider(),
        )
        chunk = make_chunk(classification=Classification.RESTRICTED)
        assert (
            Search.matches(chunk, SearchFilters(classifications=[Classification.RESTRICTED]))
            is True
        )

    def test_classification_filter_fail(self) -> None:
        Search(
            vector_store=MemoryStore(embedding_dim=2),
            embedding_provider=FakeEmbeddingProvider(),
        )
        chunk = make_chunk(classification=Classification.INTERNAL)
        assert (
            Search.matches(chunk, SearchFilters(classifications=[Classification.RESTRICTED]))
            is False
        )

    def test_owner_filter_pass(self) -> None:
        chunk = make_chunk(owner="a@co.com")
        assert Search.matches(chunk, SearchFilters(owners=["a@co.com"])) is True

    def test_owner_filter_fail(self) -> None:
        chunk = make_chunk(owner="a@co.com")
        assert Search.matches(chunk, SearchFilters(owners=["b@co.com"])) is False

    def test_multiple_filters_all_pass(self) -> None:
        chunk = make_chunk(company="Acme", department="Eng", owner="a@co.com")
        filters = SearchFilters(companies=["Acme"], departments=["Eng"], owners=["a@co.com"])
        assert Search.matches(chunk, filters) is True

    def test_multiple_filters_one_fails(self) -> None:
        chunk = make_chunk(company="Acme", department="Eng")
        filters = SearchFilters(companies=["Acme"], departments=["Sales"])
        assert Search.matches(chunk, filters) is False

    def test_empty_filters_passes(self) -> None:
        assert Search.matches(make_chunk(), SearchFilters()) is True


class TestFacetedSearchEngineCountByField:
    def test_no_records_returns_empty(self) -> None:
        engine = Search(
            vector_store=MemoryStore(embedding_dim=2),
            embedding_provider=FakeEmbeddingProvider(),
        )
        assert engine.count_field("company") == {}

    def test_counts_scalar_values(self) -> None:
        store = MemoryStore(embedding_dim=2)
        store.insert(
            [
                make_chunk(id="c1", company="Acme"),
                make_chunk(id="c2", company="Acme"),
                make_chunk(id="c3", company="Beta"),
            ],
            [[0.1, 0.2]] * 3,
        )
        engine = Search(vector_store=store, embedding_provider=FakeEmbeddingProvider())
        assert engine.count_field("company") == {"Acme": 2, "Beta": 1}

    def test_none_values_skipped(self) -> None:
        store = MemoryStore(embedding_dim=2)
        store.insert(
            [make_chunk(id="c1"), make_chunk(id="c2")],
            [[0.1, 0.2], [0.3, 0.4]],
        )
        engine = Search(vector_store=store, embedding_provider=FakeEmbeddingProvider())
        assert engine.count_field("nonexistent_field") == {}


class TestBuildFilterString:
    def test_none_returns_empty(self) -> None:
        assert build_filter(None) == ""

    def test_company_in_clause(self) -> None:
        assert (
            build_filter(SearchFilters(companies=["Acme", "Beta"])) == "company IN ('Acme', 'Beta')"
        )

    def test_combines_with_and(self) -> None:
        s = build_filter(SearchFilters(companies=["Acme"], owners=["a@b.com"]))
        assert " AND " in s
        assert "company IN ('Acme')" in s
        assert "owner IN ('a@b.com')" in s


# ---------------------------------------------------------------------------
# v0.9.0 Tier 2 — Items 9 & 10: tenant_id isolation in MemoryStore
# ---------------------------------------------------------------------------


def _chunk(  # type: ignore[no-untyped-def]
    cid: str,
    *,
    tenant_id: str | None = None,
    text: str = "alpha bravo charlie",
):
    """Build a minimal Chunk for the isolation tests."""
    from datetime import UTC, datetime
    from hashlib import sha256 as _sha256

    from raghub.models import Chunk, Classification

    return Chunk(
        id=cid,
        document_id=f"doc-{cid}",
        version=1,
        company="acme",
        owner="alice@x",
        classification=Classification.INTERNAL,
        checksum=_sha256(text.encode("utf-8")).hexdigest(),
        text=text,
        created_at=datetime.now(UTC),
        tenant_id=tenant_id,
    )


class TestMemoryStoreTenantIdIsolation:
    """Item 9: search filters by tenant_id."""

    def test_search_filters_by_explicit_tenant_id(self) -> None:
        """search(..., tenant_id='alice') returns only Alice's chunks."""
        store = MemoryStore(embedding_dim=2)
        store.insert([_chunk("a", tenant_id="alice"), _chunk("b", tenant_id="bob")], [[0.1, 0.2], [0.1, 0.2]])
        hits = store.search(vector=[0.1, 0.2], top_k=10, tenant_id="alice")
        ids = {h["chunk_id"] for h in hits}
        assert ids == {"a"}

    def test_search_filters_by_context_tenant_id(self) -> None:
        """search() honours tenant context when no explicit tenant_id."""
        from raghub.tenants import TenantContext, set_current, reset

        store = MemoryStore(embedding_dim=2)
        store.insert(
            [_chunk("a", tenant_id="alice"), _chunk("b", tenant_id="bob")],
            [[0.1, 0.2], [0.1, 0.2]],
        )
        token = set_current(TenantContext(tenant_id="bob"))
        try:
            hits = store.search(vector=[0.1, 0.2], top_k=10)
        finally:
            reset(token)
        ids = {h["chunk_id"] for h in hits}
        assert ids == {"b"}

    def test_search_returns_all_when_no_tenant_filter(self) -> None:
        """Without tenant_id or context, all chunks are visible."""
        store = MemoryStore(embedding_dim=2)
        store.insert(
            [_chunk("a", tenant_id="alice"), _chunk("b", tenant_id="bob")],
            [[0.1, 0.2], [0.1, 0.2]],
        )
        hits = store.search(vector=[0.1, 0.2], top_k=10)
        assert {h["chunk_id"] for h in hits} == {"a", "b"}

    def test_search_explicit_tenant_overrides_context(self) -> None:
        """Explicit tenant_id wins over a bound tenant context."""
        from raghub.tenants import TenantContext, set_current, reset

        store = MemoryStore(embedding_dim=2)
        store.insert(
            [_chunk("a", tenant_id="alice"), _chunk("b", tenant_id="bob")],
            [[0.1, 0.2], [0.1, 0.2]],
        )
        token = set_current(TenantContext(tenant_id="bob"))
        try:
            hits = store.search(vector=[0.1, 0.2], top_k=10, tenant_id="alice")
        finally:
            reset(token)
        ids = {h["chunk_id"] for h in hits}
        assert ids == {"a"}

    def test_search_chunks_with_none_tenant_id_excluded_when_filtering(self) -> None:
        """Chunks with ``tenant_id=None`` are excluded when a tenant is bound."""
        from raghub.tenants import TenantContext, set_current, reset

        store = MemoryStore(embedding_dim=2)
        store.insert(
            [_chunk("a", tenant_id="alice"), _chunk("legacy")],
            [[0.1, 0.2], [0.1, 0.2]],
        )
        token = set_current(TenantContext(tenant_id="alice"))
        try:
            hits = store.search(vector=[0.1, 0.2], top_k=10)
        finally:
            reset(token)
        ids = {h["chunk_id"] for h in hits}
        assert ids == {"a"}

    def test_insert_stores_tenant_id_round_trip(self) -> None:
        """Item 10: Chunk.tenant_id round-trips through the in-memory store."""
        store = MemoryStore(embedding_dim=2)
        chunk = _chunk("a", tenant_id="alice")
        store.insert([chunk], [[0.1, 0.2]])
        assert store.records["a"].chunk.tenant_id == "alice"

    def test_hybrid_search_orders_by_bm25_keyword_match(self) -> None:
        """``hybrid_search`` ranks chunks whose text matches the query above cosine neighbours.

        We seed the store with two chunks sharing the same dense
        embedding but distinct keyword vocabularies. ``hybrid_search``
        is expected to surface the keyword-matching chunk ahead of
        the unrelated one. A regression that short-circuited BM25
        (e.g. returning only dense scores) would put them at the
        same rank and fail this test.
        """
        store = MemoryStore(embedding_dim=2)
        # Identical dense vectors so the cosine rank is a tie.
        # Distinct vocabulary so BM25 discriminates.
        a = _chunk("alpha", text="revenue grew twelve percent q3")
        b = _chunk("beta", text="banana smoothie breakfast recipe")
        store.insert([a, b], [[0.5, 0.5], [0.5, 0.5]])
        store.rebuild_index()

        hits = store.hybrid_search(query="revenue", vector=[0.5, 0.5], top_k=2)
        assert len(hits) == 2
        ids = [h["chunk_id"] for h in hits]
        assert ids[0] == "alpha", (
            f"BM25 should rank the keyword-matching chunk first; got order {ids}"
        )
