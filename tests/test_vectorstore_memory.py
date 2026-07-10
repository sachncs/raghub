"""Tests for InMemoryVectorStore and FacetedSearchEngine."""

from __future__ import annotations

from typing import Any

import pytest

from raghub.models import ChunkRecord, Classification
from raghub.retrieval.search import (
    FacetedSearchEngine,
    SearchFilters,
    build_filter_string,
)
from raghub.vectorstore.memory import (
    InMemoryVectorStore,
    MemoryVectorRecord,
    matches_metadata_dict,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

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


# ===================================================================
# matches_metadata_dict  (memory.py lines 40–51)
# ===================================================================

class TestMatchesMetadataDict:
    def test_list_value_matches(self) -> None:
        record = MemoryVectorRecord(chunk=make_chunk(company="Acme"), vector=[])
        assert matches_metadata_dict(record, {"company": ["Acme", "Beta"]})

    def test_list_value_no_match(self) -> None:
        record = MemoryVectorRecord(chunk=make_chunk(company="Gamma"), vector=[])
        assert not matches_metadata_dict(record, {"company": ["Acme", "Beta"]})

    def test_scalar_value_matches(self) -> None:
        record = MemoryVectorRecord(chunk=make_chunk(document_id="d1"), vector=[])
        assert matches_metadata_dict(record, {"document_id": "d1"})

    def test_scalar_value_no_match(self) -> None:
        record = MemoryVectorRecord(chunk=make_chunk(document_id="d1"), vector=[])
        assert not matches_metadata_dict(record, {"document_id": "d2"})

    def test_missing_key_returns_false(self) -> None:
        record = MemoryVectorRecord(chunk=make_chunk(), vector=[])
        assert not matches_metadata_dict(record, {"nonexistent": "x"})

    def test_empty_filters_returns_true(self) -> None:
        record = MemoryVectorRecord(chunk=make_chunk(), vector=[])
        assert matches_metadata_dict(record, {})

    def test_multiple_criteria_all_pass(self) -> None:
        record = MemoryVectorRecord(
            chunk=make_chunk(company="Acme", document_id="d1"), vector=[]
        )
        assert matches_metadata_dict(record, {"company": ["Acme"], "document_id": "d1"})

    def test_multiple_criteria_one_fails(self) -> None:
        record = MemoryVectorRecord(
            chunk=make_chunk(company="Acme", document_id="d1"), vector=[]
        )
        assert not matches_metadata_dict(record, {"company": ["Acme"], "document_id": "d2"})


# ===================================================================
# InMemoryVectorStore
# ===================================================================

class TestInMemoryVectorStoreInit:
    def test_initial_state(self) -> None:
        store = InMemoryVectorStore()
        assert store.records == {}
        assert store.lock is not None


class TestCreateCollection:
    def test_noop(self) -> None:
        store = InMemoryVectorStore()
        assert store.create_collection() is None


class TestInsert:
    def test_insert_single(self) -> None:
        store = InMemoryVectorStore()
        chunk = make_chunk()
        store.insert([chunk], [[0.1, 0.2]])
        assert len(store.records) == 1
        assert store.records["c1"].chunk == chunk

    def test_insert_multiple(self) -> None:
        store = InMemoryVectorStore()
        c1 = make_chunk(chunk_id="c1")
        c2 = make_chunk(chunk_id="c2")
        store.insert([c1, c2], [[0.1], [0.2]])
        assert set(store.records) == {"c1", "c2"}

    def test_insert_overwrites_existing(self) -> None:
        store = InMemoryVectorStore()
        store.insert([make_chunk(chunk_id="c1", text="old")], [[0.1]])
        store.insert([make_chunk(chunk_id="c1", text="new")], [[0.5]])
        assert store.records["c1"].chunk.text == "new"

    def test_insert_strict_zip_validation(self) -> None:
        store = InMemoryVectorStore()
        chunk = make_chunk()
        with pytest.raises(ValueError):
            store.insert([chunk], [[0.1], [0.2]])

    def test_upsert_delegates_to_insert(self) -> None:
        store = InMemoryVectorStore()
        chunk = make_chunk()
        store.upsert([chunk], [[0.1]])
        assert "c1" in store.records


class TestDelete:
    def test_delete_known_ids(self) -> None:
        store = InMemoryVectorStore()
        store.insert([make_chunk(chunk_id="c1"), make_chunk(chunk_id="c2")], [[0.1], [0.2]])
        store.delete(["c1"])
        assert "c1" not in store.records
        assert "c2" in store.records

    def test_delete_unknown_ids_silently_skipped(self) -> None:
        store = InMemoryVectorStore()
        store.insert([make_chunk(chunk_id="c1")], [[0.1]])
        store.delete(["c1", "nonexistent"])
        assert len(store.records) == 0

    def test_delete_empty_list(self) -> None:
        store = InMemoryVectorStore()
        store.insert([make_chunk()], [[0.1]])
        store.delete([])
        assert len(store.records) == 1


class TestDeleteDocument:
    def test_removes_all_chunks_for_document(self) -> None:
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
        assert set(store.records) == {"c3"}

    def test_noop_for_unknown_document(self) -> None:
        store = InMemoryVectorStore()
        store.insert([make_chunk(chunk_id="c1", document_id="d1")], [[0.1]])
        store.delete_document("nonexistent")
        assert len(store.records) == 1


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

    def test_noop_for_non_matching_version(self) -> None:
        store = InMemoryVectorStore()
        store.insert([make_chunk(chunk_id="c1", document_id="d1", version=2)], [[0.1]])
        store.delete_version("d1", 1)
        assert "c1" in store.records


class TestMatchesFilter:
    def test_empty_string_returns_true(self) -> None:
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

    def test_company_in_with_double_quotes(self) -> None:
        store = InMemoryVectorStore()
        record = MemoryVectorRecord(chunk=make_chunk(company='Acme'), vector=[])
        assert store.matches_filter(record, 'company IN ("Acme")') is True

    def test_document_id_eq_match(self) -> None:
        store = InMemoryVectorStore()
        record = MemoryVectorRecord(chunk=make_chunk(document_id="d1"), vector=[])
        assert store.matches_filter(record, "document_id = 'd1'") is True

    def test_document_id_eq_no_match(self) -> None:
        store = InMemoryVectorStore()
        record = MemoryVectorRecord(chunk=make_chunk(document_id="d2"), vector=[])
        assert store.matches_filter(record, "document_id = 'd1'") is False

    def test_unknown_filter_pass_through_returns_true(self) -> None:
        store = InMemoryVectorStore()
        record = MemoryVectorRecord(chunk=make_chunk(), vector=[])
        assert store.matches_filter(record, "unknown_field = 'x'") is True


class TestComputeScore:
    def test_identical_vectors(self) -> None:
        store = InMemoryVectorStore()
        score = store.compute_score([1.0, 0.0], [1.0, 0.0])
        assert score == pytest.approx(1.0)

    def test_orthogonal_vectors(self) -> None:
        store = InMemoryVectorStore()
        score = store.compute_score([1.0, 0.0], [0.0, 1.0])
        assert score == pytest.approx(0.0)

    def test_zero_query_vector(self) -> None:
        store = InMemoryVectorStore()
        score = store.compute_score([0.0, 0.0], [1.0, 0.0])
        assert score == 0.0

    def test_zero_stored_vector(self) -> None:
        store = InMemoryVectorStore()
        score = store.compute_score([1.0, 0.0], [0.0, 0.0])
        assert score == 0.0

    def test_opposite_vectors(self) -> None:
        store = InMemoryVectorStore()
        score = store.compute_score([1.0, 0.0], [-1.0, 0.0])
        assert score == pytest.approx(-1.0)

    def test_partial_similarity(self) -> None:
        store = InMemoryVectorStore()
        score = store.compute_score([1.0, 2.0], [2.0, 4.0])
        assert score == pytest.approx(1.0)


class TestSearch:
    def test_vector_filter(self) -> None:
        store = InMemoryVectorStore()
        chunk = make_chunk(chunk_id="c1", company="Acme")
        store.insert([chunk], [[1.0, 0.0]])
        results = store.search(vector=[1.0, 0.0], top_k=5, metadata_filter="company IN ('Acme')")
        assert len(results) == 1
        assert results[0]["chunk_id"] == "c1"

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
        assert len(results) == 1
        assert results[0]["chunk_id"] == "c1"

    def test_dict_filter_with_list(self) -> None:
        store = InMemoryVectorStore()
        store.insert(
            [
                make_chunk(chunk_id="c1", company="Acme"),
                make_chunk(chunk_id="c2", company="Beta"),
                make_chunk(chunk_id="c3", company="Gamma"),
            ],
            [[0.1, 0.2], [0.1, 0.2], [0.1, 0.2]],
        )
        results = store.search(vector=[0.1, 0.2], top_k=5, metadata_filter={"company": ["Acme", "Beta"]})
        assert len(results) == 2
        assert {r["chunk_id"] for r in results} == {"c1", "c2"}

    def test_string_filter_empty(self) -> None:
        store = InMemoryVectorStore()
        store.insert([make_chunk(chunk_id="c1")], [[0.1, 0.2]])
        results = store.search(vector=[0.1, 0.2], top_k=5, metadata_filter="")
        assert len(results) == 1

    def test_string_filter_no_match(self) -> None:
        store = InMemoryVectorStore()
        store.insert([make_chunk(chunk_id="c1", company="Acme")], [[0.1, 0.2]])
        results = store.search(vector=[0.1, 0.2], top_k=5, metadata_filter="company IN ('Beta')")
        assert len(results) == 0

    def test_empty_store_returns_empty(self) -> None:
        store = InMemoryVectorStore()
        results = store.search(vector=[1.0, 0.0], top_k=5)
        assert results == []

    def test_top_k_limits_results(self) -> None:
        store = InMemoryVectorStore()
        chunks = [make_chunk(chunk_id=f"c{i}") for i in range(10)]
        store.insert(chunks, [[0.1, 0.2]] * 10)
        results = store.search(vector=[0.1, 0.2], top_k=3)
        assert len(results) == 3

    def test_default_filter_is_empty_string(self) -> None:
        store = InMemoryVectorStore()
        store.insert([make_chunk(chunk_id="c1")], [[0.1, 0.2]])
        results = store.search(vector=[0.1, 0.2], top_k=5)
        assert len(results) == 1


class TestHybridSearch:
    def test_delegates_to_search(self) -> None:
        store = InMemoryVectorStore()
        store.insert([make_chunk(chunk_id="c1")], [[0.1, 0.2]])
        results = store.hybrid_search(
            query="ignored", vector=[0.1, 0.2], top_k=5, metadata_filter=""
        )
        assert len(results) == 1
        assert results[0]["chunk_id"] == "c1"

    def test_hybrid_search_with_dict_filter(self) -> None:
        store = InMemoryVectorStore()
        store.insert([make_chunk(chunk_id="c1", company="Acme")], [[0.1, 0.2]])
        results = store.hybrid_search(
            query="ignored", vector=[0.1, 0.2], top_k=5, metadata_filter={"company": "Acme"}
        )
        assert len(results) == 1


class TestKeywordSearch:
    def test_finds_matching_chunks(self) -> None:
        store = InMemoryVectorStore()
        store.insert([make_chunk(chunk_id="c1", text="hello world")], [[0.1]])
        store.insert([make_chunk(chunk_id="c2", text="foo bar")], [[0.1]])
        results = store.keyword_search("hello", top_k=5)
        assert len(results) == 1
        assert results[0]["chunk_id"] == "c1"

    def test_scores_reflect_term_frequency(self) -> None:
        store = InMemoryVectorStore()
        store.insert([make_chunk(chunk_id="c1", text="hello hello world")], [[0.1]])
        store.insert([make_chunk(chunk_id="c2", text="hello world")], [[0.1]])
        results = store.keyword_search("hello", top_k=5)
        assert len(results) == 2
        assert results[0]["chunk_id"] == "c1"
        assert results[0]["score"] > results[1]["score"]

    def test_empty_query_returns_empty_list(self) -> None:
        store = InMemoryVectorStore()
        store.insert([make_chunk(chunk_id="c1")], [[0.1]])
        assert store.keyword_search("", top_k=5) == []

    def test_blank_query_returns_empty(self) -> None:
        store = InMemoryVectorStore()
        store.insert([make_chunk(chunk_id="c1")], [[0.1]])
        assert store.keyword_search("   ", top_k=5) == []

    def test_no_match_returns_empty(self) -> None:
        store = InMemoryVectorStore()
        store.insert([make_chunk(chunk_id="c1", text="hello world")], [[0.1]])
        results = store.keyword_search("zzzzzz", top_k=5)
        assert results == []

    def test_empty_text_chunks_are_skipped(self) -> None:
        store = InMemoryVectorStore()
        store.insert([make_chunk(chunk_id="c1", text="")], [[0.1]])
        store.insert([make_chunk(chunk_id="c2", text="hello")], [[0.1]])
        results = store.keyword_search("hello", top_k=5)
        assert len(results) == 1
        assert results[0]["chunk_id"] == "c2"

    def test_top_k_limits(self) -> None:
        store = InMemoryVectorStore()
        chunks = [make_chunk(chunk_id=f"c{i}", text="hello world") for i in range(5)]
        store.insert(chunks, [[0.1]] * 5)
        results = store.keyword_search("hello", top_k=2)
        assert len(results) == 2


class TestOptimize:
    def test_noop_returns_none(self) -> None:
        store = InMemoryVectorStore()
        assert store.optimize() is None


class TestHealth:
    def test_returns_status_dict(self) -> None:
        store = InMemoryVectorStore()
        store.insert([make_chunk()], [[0.1]])
        status = store.health()
        assert status == {"status": "ok", "backend": "memory", "chunks": 1}

    def test_chunks_count_reflects_records(self) -> None:
        store = InMemoryVectorStore()
        assert store.health()["chunks"] == 0
        store.insert([make_chunk(chunk_id="c1")], [[0.1]])
        assert store.health()["chunks"] == 1


# ===================================================================
# build_filter_string  (search.py lines 33–54)
# ===================================================================

class TestBuildFilterString:
    def test_none_returns_empty_string(self) -> None:
        assert build_filter_string(None) == ""

    def test_empty_filters_returns_empty_string(self) -> None:
        filters = SearchFilters()
        assert build_filter_string(filters) == ""

    def test_companies_produces_correct_clause(self) -> None:
        filters = SearchFilters(companies=["Acme", "Beta"])
        result = build_filter_string(filters)
        assert result == "company IN ('Acme', 'Beta')"

    def test_owners_produces_correct_clause(self) -> None:
        filters = SearchFilters(owners=["alice@co.com", "bob@co.com"])
        result = build_filter_string(filters)
        assert result == "owner IN ('alice@co.com', 'bob@co.com')"

    def test_file_types_produces_correct_clause(self) -> None:
        filters = SearchFilters(file_types=["pdf", "docx"])
        result = build_filter_string(filters)
        assert result == "file_type IN ('pdf', 'docx')"

    def test_multiple_fields_combined_with_and(self) -> None:
        filters = SearchFilters(
            companies=["Acme"],
            owners=["alice@co.com"],
            file_types=["pdf"],
        )
        result = build_filter_string(filters)
        expected = "company IN ('Acme') AND owner IN ('alice@co.com') AND file_type IN ('pdf')"
        assert result == expected

    def test_single_item_does_not_add_trailing_comma(self) -> None:
        filters = SearchFilters(companies=["Acme"])
        result = build_filter_string(filters)
        assert result == "company IN ('Acme')"


# ===================================================================
# FacetedSearchEngine
# ===================================================================

class FakeEmbeddingProvider:
    def embed_text(self, text: str) -> list[float]:
        return [0.1, 0.2]


class TestFacetedSearchEngineInit:
    def test_stores_vector_store_and_embedding_provider(self) -> None:
        vs = InMemoryVectorStore()
        ep = FakeEmbeddingProvider()
        engine = FacetedSearchEngine(vector_store=vs, embedding_provider=ep)
        assert engine.vector_store is vs
        assert engine.embedding_provider is ep


class TestFacetedSearchEngineSearch:
    def test_no_filters_returns_all_hits(self) -> None:
        vs = InMemoryVectorStore()
        vs.insert([make_chunk(chunk_id="c1"), make_chunk(chunk_id="c2")], [[0.1, 0.2], [0.1, 0.2]])
        engine = FacetedSearchEngine(vector_store=vs, embedding_provider=FakeEmbeddingProvider())
        results = engine.search("test", top_k=10)
        assert len(results) == 2

    def test_filters_post_filter(self) -> None:
        vs = InMemoryVectorStore()
        vs.insert(
            [
                make_chunk(chunk_id="c1", company="Acme", owner="a@co.com"),
                make_chunk(chunk_id="c2", company="Beta", owner="b@co.com"),
            ],
            [[0.1, 0.2], [0.1, 0.2]],
        )
        engine = FacetedSearchEngine(vector_store=vs, embedding_provider=FakeEmbeddingProvider())
        filters = SearchFilters(companies=["Acme"])
        results = engine.search("test", filters=filters, top_k=10)
        assert len(results) == 1
        assert results[0].chunk_id == "c1"

    def test_filters_with_no_matches_returns_empty(self) -> None:
        vs = InMemoryVectorStore()
        vs.insert([make_chunk(chunk_id="c1", company="Acme")], [[0.1, 0.2]])
        engine = FacetedSearchEngine(vector_store=vs, embedding_provider=FakeEmbeddingProvider())
        filters = SearchFilters(companies=["Nonexistent"])
        results = engine.search("test", filters=filters, top_k=10)
        assert results == []

    def test_post_filter_rejects_chunks(self) -> None:
        vs = InMemoryVectorStore()
        vs.insert(
            [
                make_chunk(chunk_id="c1", company="Acme", department="Engineering"),
                make_chunk(chunk_id="c2", company="Beta", department="Sales"),
            ],
            [[0.1, 0.2], [0.1, 0.2]],
        )
        engine = FacetedSearchEngine(vector_store=vs, embedding_provider=FakeEmbeddingProvider())
        filters = SearchFilters(
            companies=["Acme", "Beta"],
            departments=["Engineering"],
        )
        results = engine.search("test", filters=filters, top_k=10)
        assert len(results) == 1
        assert results[0].chunk_id == "c1"
        assert results[0].company == "Acme"

    def test_duplicate_chunks_deduplicated(self) -> None:
        class DedupStore(InMemoryVectorStore):
            def search(self, **kwargs: Any) -> list[dict[str, Any]]:
                chunk = make_chunk(chunk_id="c1")
                return [
                    {"chunk_id": "c1", "score": 1.0, "chunk": chunk},
                    {"chunk_id": "c1", "score": 0.9, "chunk": chunk},
                ]

        engine = FacetedSearchEngine(vector_store=DedupStore(), embedding_provider=FakeEmbeddingProvider())
        results = engine.search("test", top_k=10)
        assert len(results) == 1

    def test_top_k_limits(self) -> None:
        vs = InMemoryVectorStore()
        vs.insert(
            [make_chunk(chunk_id=f"c{i}") for i in range(5)],
            [[0.1, 0.2]] * 5,
        )
        engine = FacetedSearchEngine(vector_store=vs, embedding_provider=FakeEmbeddingProvider())
        results = engine.search("test", top_k=3)
        assert len(results) == 3


class TestFacetedSearchEngineMatchesFilters:
    def test_passes_when_no_filters_active(self) -> None:
        engine = FacetedSearchEngine(vector_store=InMemoryVectorStore(), embedding_provider=FakeEmbeddingProvider())
        chunk = make_chunk()
        assert engine.matches_filters(chunk, SearchFilters()) is True

    def test_company_filter_pass(self) -> None:
        engine = FacetedSearchEngine(vector_store=InMemoryVectorStore(), embedding_provider=FakeEmbeddingProvider())
        chunk = make_chunk(company="Acme")
        assert engine.matches_filters(chunk, SearchFilters(companies=["Acme"])) is True

    def test_company_filter_fail(self) -> None:
        engine = FacetedSearchEngine(vector_store=InMemoryVectorStore(), embedding_provider=FakeEmbeddingProvider())
        chunk = make_chunk(company="Acme")
        assert engine.matches_filters(chunk, SearchFilters(companies=["Beta"])) is False

    def test_department_filter_pass(self) -> None:
        engine = FacetedSearchEngine(vector_store=InMemoryVectorStore(), embedding_provider=FakeEmbeddingProvider())
        chunk = make_chunk(department="Engineering")
        assert engine.matches_filters(chunk, SearchFilters(departments=["Engineering"])) is True

    def test_department_filter_fail(self) -> None:
        engine = FacetedSearchEngine(vector_store=InMemoryVectorStore(), embedding_provider=FakeEmbeddingProvider())
        chunk = make_chunk(department="Engineering")
        assert engine.matches_filters(chunk, SearchFilters(departments=["Sales"])) is False

    def test_classification_filter_pass(self) -> None:
        engine = FacetedSearchEngine(vector_store=InMemoryVectorStore(), embedding_provider=FakeEmbeddingProvider())
        chunk = make_chunk(classification=Classification.CONFIDENTIAL)
        assert engine.matches_filters(chunk, SearchFilters(classifications=[Classification.CONFIDENTIAL])) is True

    def test_classification_filter_fail(self) -> None:
        engine = FacetedSearchEngine(vector_store=InMemoryVectorStore(), embedding_provider=FakeEmbeddingProvider())
        chunk = make_chunk(classification=Classification.INTERNAL)
        assert engine.matches_filters(chunk, SearchFilters(classifications=[Classification.RESTRICTED])) is False

    def test_owner_filter_pass(self) -> None:
        engine = FacetedSearchEngine(vector_store=InMemoryVectorStore(), embedding_provider=FakeEmbeddingProvider())
        chunk = make_chunk(owner="a@co.com")
        assert engine.matches_filters(chunk, SearchFilters(owners=["a@co.com"])) is True

    def test_owner_filter_fail(self) -> None:
        engine = FacetedSearchEngine(vector_store=InMemoryVectorStore(), embedding_provider=FakeEmbeddingProvider())
        chunk = make_chunk(owner="a@co.com")
        assert engine.matches_filters(chunk, SearchFilters(owners=["b@co.com"])) is False

    def test_multiple_filters_all_pass(self) -> None:
        engine = FacetedSearchEngine(vector_store=InMemoryVectorStore(), embedding_provider=FakeEmbeddingProvider())
        chunk = make_chunk(company="Acme", department="Eng", owner="a@co.com")
        filters = SearchFilters(companies=["Acme"], departments=["Eng"], owners=["a@co.com"])
        assert engine.matches_filters(chunk, filters) is True

    def test_multiple_filters_one_fails(self) -> None:
        engine = FacetedSearchEngine(vector_store=InMemoryVectorStore(), embedding_provider=FakeEmbeddingProvider())
        chunk = make_chunk(company="Acme", department="Eng")
        filters = SearchFilters(companies=["Acme"], departments=["Sales"])
        assert engine.matches_filters(chunk, filters) is False


class TestFacetedSearchEngineCountByField:
    def test_no_records_returns_empty_dict(self) -> None:
        vs = InMemoryVectorStore()
        engine = FacetedSearchEngine(vector_store=vs, embedding_provider=FakeEmbeddingProvider())
        assert engine.count_by_field("company") == {}

    def test_counts_scalar_values(self) -> None:
        vs = InMemoryVectorStore()
        vs.insert(
            [
                make_chunk(chunk_id="c1", company="Acme"),
                make_chunk(chunk_id="c2", company="Acme"),
                make_chunk(chunk_id="c3", company="Beta"),
            ],
            [[0.1], [0.1], [0.1]],
        )
        engine = FacetedSearchEngine(vector_store=vs, embedding_provider=FakeEmbeddingProvider())
        counts = engine.count_by_field("company")
        assert counts == {"Acme": 2, "Beta": 1}

    def test_counts_list_values(self) -> None:
        vs = InMemoryVectorStore()
        c1 = make_chunk(chunk_id="c1", department="Eng")
        c2 = make_chunk(chunk_id="c2", department="Eng")
        c3 = make_chunk(chunk_id="c3", department="Sales")
        vs.insert([c1, c2, c3], [[0.1], [0.1], [0.1]])
        engine = FacetedSearchEngine(vector_store=vs, embedding_provider=FakeEmbeddingProvider())
        counts = engine.count_by_field("department")
        assert counts == {"Eng": 2, "Sales": 1}

    def test_none_values_are_skipped(self) -> None:
        vs = InMemoryVectorStore()
        vs.insert(
            [
                make_chunk(chunk_id="c1", version=1),
                make_chunk(chunk_id="c2", version=2),
            ],
            [[0.1], [0.1]],
        )
        engine = FacetedSearchEngine(vector_store=vs, embedding_provider=FakeEmbeddingProvider())
        counts = engine.count_by_field("nonexistent_field")
        assert counts == {}

    def test_records_is_none_returns_empty(self) -> None:
        class StoreNoRecords:
            records = None

        engine = FacetedSearchEngine(vector_store=StoreNoRecords(), embedding_provider=FakeEmbeddingProvider())
        assert engine.count_by_field("company") == {}

    def test_list_value_in_field_counts_each_element(self) -> None:
        vs = InMemoryVectorStore()
        c1 = make_chunk(chunk_id="c1")
        c2 = make_chunk(chunk_id="c2")
        object.__setattr__(c1, "tags", ["a", "b"])
        object.__setattr__(c2, "tags", ["a"])
        vs.insert([c1, c2], [[0.1], [0.1]])
        engine = FacetedSearchEngine(vector_store=vs, embedding_provider=FakeEmbeddingProvider())
        counts = engine.count_by_field("tags")
        assert counts == {"a": 2, "b": 1}
