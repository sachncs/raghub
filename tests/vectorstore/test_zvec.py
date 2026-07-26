"""Tests for ``raghub.vectorstore.zvec.ZvecVectorStore`` and helpers."""
from __future__ import annotations

import os
import shutil
import tempfile
from collections.abc import Sequence
from datetime import UTC, datetime
from typing import Any
from uuid import uuid4

import pytest

from raghub.models import ChunkRecord, Classification
from raghub.vectorstore import (
    RealZvecBackend,
    ZvecVectorStore,
    native_filter,
)


def _tmpdir() -> str:
    """Return a fresh, non-existent path under the system temp dir.

    Zvec's ``create_and_open`` requires a path that does not yet
    exist, so the fixture must not pre-create the directory.
    """
    return os.path.join(tempfile.gettempdir(), f"raghub_zvec_{uuid4().hex}")


def _make_chunk(**overrides: Any) -> ChunkRecord:
    defaults: dict[str, Any] = dict(
        chunk_id=str(uuid4()),
        document_id="doc-1",
        version=1,
        page=1,
        text="hello world",
        company="acme",
        owner="owner@acme.com",
        classification=Classification.INTERNAL,
        created_at=datetime.now(UTC),
        embedding_model="hashing-bge",
        hash="h1",
    )
    defaults.update(overrides)
    return ChunkRecord(**defaults)


@pytest.fixture
def tmp_path() -> str:
    path = _tmpdir()
    try:
        yield path
    finally:
        shutil.rmtree(path, ignore_errors=True)


# ---------------------------------------------------------------------------
# ``native_filter`` translation
# ---------------------------------------------------------------------------


def test_native_filter_returns_none_for_empty_string() -> None:
    """Empty / ``None`` filters translate to ``None`` (no clause)."""
    assert native_filter("") is None
    assert native_filter(None) is None


def test_native_filter_empty_dict_returns_none() -> None:
    """An empty dict filter also yields no clause."""
    assert native_filter({}) is None


def test_native_filter_scalar_string() -> None:
    """A scalar string value is rendered as ``key = 'value'``."""
    assert native_filter({"company": "acme"}) == "company = 'acme'"


def test_native_filter_list_value() -> None:
    """A list value is rendered as ``key IN (...)`` with quoted literals."""
    assert (
        native_filter({"company": ["a", "b"]}) == "company IN ('a', 'b')"
    )


def test_native_filter_empty_list_returns_false_clause() -> None:
    """Empty list values produce ``"false"`` to match no records."""
    assert native_filter({"company": []}) == "false"


def test_native_filter_escapes_single_quotes() -> None:
    """Embedded single quotes in values are doubled for SQL safety."""
    assert (
        native_filter({"company": ["a'b"]}) == "company IN ('a''b')"
    )


def test_native_filter_supports_document_id() -> None:
    """``document_id`` is allowed as a key alongside ``company``."""
    assert (
        native_filter({"document_id": "doc-1"})
        == "document_id = 'doc-1'"
    )


def test_native_filter_passes_through_legacy_string() -> None:
    """Non-empty strings are returned verbatim (legacy contract)."""
    assert native_filter("document_id = 'x'") == "document_id = 'x'"


def test_native_filter_rejects_unknown_keys() -> None:
    """Unknown dict keys raise :class:`ValueError`."""
    with pytest.raises(ValueError):
        native_filter({"foo": "bar"})


def test_native_filter_rejects_unsupported_scalar_type() -> None:
    """Numeric values are rejected — only ``str`` and ``list[str]``."""
    with pytest.raises(ValueError):
        native_filter({"company": 5})


def test_native_filter_rejects_unsupported_filter_type() -> None:
    """Non-string/dict filters raise :class:`ValueError`."""
    with pytest.raises(ValueError):
        native_filter(123)


def test_native_filter_combines_multiple_keys_with_and() -> None:
    """Multi-key filters are joined with `` AND ``."""
    assert (
        native_filter({"company": "acme", "document_id": "d1"})
        == "company = 'acme' AND document_id = 'd1'"
    )


# ---------------------------------------------------------------------------
# ``RealZvecBackend`` helper methods (no I/O).
# ---------------------------------------------------------------------------


def test_real_zvec_backend_sanitize_id_strips_disallowed_chars() -> None:
    """``sanitize_id`` keeps only alphanumerics and ``-_.``."""
    backend = RealZvecBackend.__new__(RealZvecBackend)
    backend.path = ""
    backend.embedding_dim = 4
    assert backend.sanitize_id("abc-DEF.123") == "abc-DEF.123"
    assert backend.sanitize_id("hello world!") == "helloworld"


def test_real_zvec_backend_matches_metadata_dict() -> None:
    """``matches_metadata`` accepts supported dict shapes."""
    backend = RealZvecBackend.__new__(RealZvecBackend)
    backend.path = ""
    backend.embedding_dim = 4
    assert backend.matches_metadata({"company": "a"}) is True
    assert backend.matches_metadata({"company": ["a", "b"]}) is True
    assert backend.matches_metadata({"foo": "a"}) is False
    assert backend.matches_metadata({"company": 5}) is False


def test_real_zvec_backend_matches_metadata_passes_through_strings() -> None:
    """String filters are always accepted (legacy pass-through)."""
    backend = RealZvecBackend.__new__(RealZvecBackend)
    backend.path = ""
    backend.embedding_dim = 4
    assert backend.matches_metadata("document_id = 'x'") is True
    assert backend.matches_metadata("") is True
    assert backend.matches_metadata(None) is True


def test_real_zvec_backend_keyword_search_returns_empty() -> None:
    """The native backend has no keyword index — always empty."""
    backend = RealZvecBackend.__new__(RealZvecBackend)
    backend.path = ""
    backend.embedding_dim = 4
    assert backend.keyword_search("anything", top_k=5) == []


# ---------------------------------------------------------------------------
# ``ZvecVectorStore`` integration: native backend when zvec is importable.
# ---------------------------------------------------------------------------


def test_zvec_store_uses_native_backend_when_module_present(tmp_path: str) -> None:
    """When the ``zvec`` module is importable the wrapper selects it."""
    store = ZvecVectorStore(path=tmp_path, embedding_dim=4)
    assert store.zvec_module is not None
    assert isinstance(store.backend, RealZvecBackend)
    assert store.embedding_dim == 4
    assert store.path == tmp_path


def test_zvec_store_create_collection_is_noop(tmp_path: str) -> None:
    """``create_collection`` is a no-op once the backend opens the collection."""
    store = ZvecVectorStore(path=tmp_path, embedding_dim=4)
    assert store.create_collection() is None


def test_zvec_store_insert_and_search(tmp_path: str) -> None:
    """Inserted chunks are retrievable via vector search."""
    store = ZvecVectorStore(path=tmp_path, embedding_dim=4)
    store.create_collection()
    chunks = [_make_chunk(chunk_id=f"c{i}", text=f"text {i}") for i in range(3)]
    vectors = [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0]]
    store.insert(chunks, vectors)
    hits = store.search(vector=[1.0, 0.0, 0.0, 0.0], top_k=2)
    assert len(hits) == 2
    assert {hit["chunk_id"] for hit in hits} == {"c0", "c1"}


def test_zvec_store_upsert_replaces_records(tmp_path: str) -> None:
    """``upsert`` delegates to insert; the latest vector wins."""
    store = ZvecVectorStore(path=tmp_path, embedding_dim=4)
    store.create_collection()
    chunk = _make_chunk(chunk_id="dup")
    store.insert([chunk], [[1.0, 0.0, 0.0, 0.0]])
    store.upsert([chunk], [[0.0, 1.0, 0.0, 0.0]])
    hits = store.search(vector=[0.0, 1.0, 0.0, 0.0], top_k=1)
    assert len(hits) == 1


def test_zvec_store_delete_chunks(tmp_path: str) -> None:
    """Deleting by ``chunk_id`` removes the chunk."""
    store = ZvecVectorStore(path=tmp_path, embedding_dim=4)
    store.create_collection()
    chunks = [_make_chunk(chunk_id=f"c{i}") for i in range(3)]
    vectors = [[1.0, 0.0, 0.0, 0.0]] * 3
    store.insert(chunks, vectors)
    store.delete(["c0", "c1"])
    hits = store.search(vector=[1.0, 0.0, 0.0, 0.0], top_k=5)
    assert {hit["chunk_id"] for hit in hits} == {"c2"}


def test_zvec_store_delete_document(tmp_path: str) -> None:
    """``delete_document`` removes every chunk of a document."""
    store = ZvecVectorStore(path=tmp_path, embedding_dim=4)
    store.create_collection()
    store.insert(
        [
            _make_chunk(chunk_id="a1", document_id="A"),
            _make_chunk(chunk_id="a2", document_id="A"),
            _make_chunk(chunk_id="b1", document_id="B"),
        ],
        [[1.0, 0.0, 0.0, 0.0]] * 3,
    )
    store.delete_document("A")
    hits = store.search(vector=[1.0, 0.0, 0.0, 0.0], top_k=5)
    assert {hit["chunk_id"] for hit in hits} == {"b1"}


def test_zvec_store_delete_version(tmp_path: str) -> None:
    """``delete_version`` removes chunks for one document version only."""
    store = ZvecVectorStore(path=tmp_path, embedding_dim=4)
    store.create_collection()
    store.insert(
        [
            _make_chunk(chunk_id="v1", document_id="D", version=1),
            _make_chunk(chunk_id="v2", document_id="D", version=2),
        ],
        [[1.0, 0.0, 0.0, 0.0]] * 2,
    )
    store.delete_version("D", 1)
    hits = store.search(vector=[1.0, 0.0, 0.0, 0.0], top_k=5)
    assert {hit["chunk_id"] for hit in hits} == {"v2"}


def test_zvec_store_search_with_metadata_filter(tmp_path: str) -> None:
    """A canonical dict filter is translated to a Zvec SQL fragment."""
    store = ZvecVectorStore(path=tmp_path, embedding_dim=4)
    store.create_collection()
    store.insert(
        [
            _make_chunk(chunk_id="a", company="acme"),
            _make_chunk(chunk_id="b", company="globex"),
        ],
        [[1.0, 0.0, 0.0, 0.0]] * 2,
    )
    hits = store.search(
        vector=[1.0, 0.0, 0.0, 0.0], top_k=5, metadata_filter={"company": "acme"}
    )
    assert {hit["chunk_id"] for hit in hits} == {"a"}


def test_zvec_store_hybrid_search_delegates(tmp_path: str) -> None:
    """``hybrid_search`` is the same as ``search`` (no keyword channel)."""
    store = ZvecVectorStore(path=tmp_path, embedding_dim=4)
    store.create_collection()
    store.insert([_make_chunk(chunk_id="a")], [[1.0, 0.0, 0.0, 0.0]])
    hits = store.hybrid_search(
        query="a", vector=[1.0, 0.0, 0.0, 0.0], top_k=5
    )
    assert len(hits) == 1


def test_zvec_store_keyword_search_returns_empty(tmp_path: str) -> None:
    """``keyword_search`` is unsupported on the native backend."""
    store = ZvecVectorStore(path=tmp_path, embedding_dim=4)
    assert store.keyword_search("anything", top_k=5) == []


def test_zvec_store_optimize_does_not_raise(tmp_path: str) -> None:
    """``optimize`` is forwarded to the native backend."""
    store = ZvecVectorStore(path=tmp_path, embedding_dim=4)
    store.optimize()


def test_zvec_store_health_reports_native_backend(tmp_path: str) -> None:
    """Health reports the native backend when zvec is installed."""
    store = ZvecVectorStore(path=tmp_path, embedding_dim=4)
    health = store.health()
    assert health["status"] == "ok"
    assert health["backend"] == "zvec"


# ---------------------------------------------------------------------------
# Fallback path: ``ZvecVectorStore`` falls back to ``InMemoryVectorStore``
# when ``zvec`` is unavailable (tested by injecting a missing module).
# ---------------------------------------------------------------------------


def test_zvec_store_falls_back_when_module_missing(
    tmp_path: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Without ``zvec``, the wrapper picks the in-memory fallback."""
    from raghub.vectorstore import zvec as zvec_mod
    from raghub.vectorstore import InMemoryVectorStore

    monkeypatch.setattr(zvec_mod, "zvec_module", None)
    store = ZvecVectorStore(path=tmp_path, embedding_dim=4)
    assert store.zvec_module is None
    assert isinstance(store.backend, InMemoryVectorStore)
    assert store.health()["backend"] == "memory"


def test_zvec_store_require_zvec_raises_when_missing(
    tmp_path: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``require_zvec=True`` raises when zvec is unavailable."""
    from raghub.vectorstore import zvec as zvec_mod

    monkeypatch.setattr(zvec_mod, "zvec_module", None)
    with pytest.raises(RuntimeError):
        ZvecVectorStore(path=tmp_path, embedding_dim=4, require_zvec=True)


def test_zvec_store_normalize_search_result_handles_none(
    tmp_path: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``normalize_search_result`` returns ``[]`` when the native result is ``None``."""
    from raghub.vectorstore import zvec as zvec_mod

    monkeypatch.setattr(zvec_mod, "zvec_module", None)
    backend = RealZvecBackend.__new__(RealZvecBackend)
    backend.path = ""
    backend.embedding_dim = 4
    assert backend.normalize_search_result(None) == []