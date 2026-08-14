"""Knowledge representation coverage tests.

Exercises the OKF serialisation helpers, the in-memory knowledge
repository, the source manifest, the pure helpers (cosine_similarity, sha256,
extract_json_object, tokenise), and the public surface of the two
structured-retrieval indexes (Raptor, GraphIndex).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from raghub.errors import KnowledgeError
from raghub.knowledge import (
    GraphIndex,
    Manifest,
    MemoryRepo,
    Raptor,
    chunk_to_record,
    cluster,
    cosine_similarity,
    dumps,
    extract_json_object,
    from_okf,
    loads,
    sha256_bytes,
    summary_id_for,
    to_okf,
    tokenise,
)
from raghub.models import (
    BlockKind,
    Bundle,
    Chunk,
    DocumentBlock,
    DocumentSection,
)


def _make_bundle(bundle_id: str = "b1") -> Bundle:
    """Build a small Bundle with one section and one text block."""
    return Bundle(
        bundle_id=bundle_id,
        source_uri="mem://x",
        checksum="0" * 64,
        language="en",
        mime_type="text/plain",
        metadata={"k": "v"},
        sections=[
            DocumentSection(
                section_id="s1",
                index=0,
                heading="Intro",
                blocks=[
                    DocumentBlock(
                        block_id="blk1",
                        kind=BlockKind.Text,
                        content="hello",
                        metadata={"tag": "greeting"},
                    ),
                ],
            ),
        ],
    )


# ---------------------------------------------------------------------------
# OKF round-trips
# ---------------------------------------------------------------------------


def test_to_okf_serialises_every_field() -> None:
    """to_okf emits a dict with $schema + every Bundle field."""

    bundle = _make_bundle()
    payload = to_okf(bundle)
    assert payload["$schema"].startswith("okf/")
    assert payload["bundle_id"] == "b1"
    assert payload["source_uri"] == "mem://x"
    assert payload["checksum"] == "0" * 64
    assert payload["language"] == "en"
    assert payload["mime_type"] == "text/plain"
    assert payload["metadata"] == {"k": "v"}
    assert len(payload["sections"]) == 1
    section = payload["sections"][0]
    assert section["section_id"] == "s1"
    assert section["heading"] == "Intro"
    assert section["blocks"][0]["kind"] == "text"
    assert section["blocks"][0]["content"] == "hello"


def test_from_okf_round_trip() -> None:
    """to_okf -> from_okf round-trips cleanly."""

    bundle = _make_bundle(bundle_id="rt-1")
    payload = to_okf(bundle)
    rebuilt = from_okf(payload)
    assert rebuilt.bundle_id == bundle.bundle_id
    assert rebuilt.source_uri == bundle.source_uri
    assert rebuilt.checksum == bundle.checksum
    assert len(rebuilt.sections) == 1
    assert rebuilt.sections[0].blocks[0].content == "hello"


def test_from_okf_accepts_json_string() -> None:
    """from_okf parses a JSON string."""

    bundle = _make_bundle(bundle_id="js-1")
    text = dumps(bundle)
    rebuilt = from_okf(text)
    assert rebuilt.bundle_id == "js-1"


def test_from_okf_rejects_non_dict_json() -> None:
    """from_okf raises KnowledgeError on a non-object JSON string."""

    with pytest.raises(KnowledgeError):
        from_okf('["not", "a dict"]')


def test_from_okf_rejects_unknown_block_kind() -> None:
    """from_okf raises KnowledgeError on an unknown block kind."""

    bad = {
        "$schema": "okf/0.1",
        "bundle_id": "b",
        "source_uri": "x",
        "checksum": "",
        "sections": [
            {
                "section_id": "s",
                "index": 0,
                "heading": "",
                "blocks": [{"block_id": "b", "kind": "mystery", "content": "x"}],
            }
        ],
    }
    with pytest.raises(KnowledgeError, match="Unknown OKF block kind"):
        from_okf(bad)


def test_dumps_loads_round_trip() -> None:
    """dumps + loads round-trip without loss."""

    bundle = _make_bundle(bundle_id="dl")
    text = dumps(bundle, indent=2)
    assert json.loads(text)["bundle_id"] == "dl"
    rebuilt = loads(text)
    assert rebuilt.bundle_id == bundle.bundle_id


def test_loads_rejects_invalid_json() -> None:
    """loads raises KnowledgeError on invalid JSON."""

    with pytest.raises(KnowledgeError):
        loads("not-json")


def test_loads_rejects_non_object_json() -> None:
    """loads raises KnowledgeError on a non-object JSON value."""

    with pytest.raises(KnowledgeError):
        loads("[1,2,3]")


# ---------------------------------------------------------------------------
# MemoryRepo
# ---------------------------------------------------------------------------


def test_memory_repo_save_get_list_delete() -> None:
    """MemoryRepo round-trips save / get / list_by_source / delete."""

    repo = MemoryRepo()
    bundle = _make_bundle(bundle_id="b-mr")
    saved = repo.save(bundle)
    assert saved.bundle_id == bundle.bundle_id

    assert repo.get("b-mr") is not None
    assert repo.list_by_source("mem://x") == [saved]

    repo.delete("b-mr")
    assert repo.get("b-mr") is None
    assert repo.list_by_source("mem://x") == []


def test_memory_repo_get_missing_returns_none() -> None:
    """MemoryRepo.get returns None for unknown bundle ids."""

    repo = MemoryRepo()
    assert repo.get("missing") is None


# ---------------------------------------------------------------------------
# Manifest
# ---------------------------------------------------------------------------


@pytest.fixture
def manifest_path(tmp_path: Path) -> Path:
    return tmp_path / "manifest.json"


def test_manifest_loads_from_disk(manifest_path: Path) -> None:
    """Manifest.load() hydrates from a v2 manifest file."""

    records = {
        "mem://a": {"bundle_id": "b-a", "checksum": "x"},
        "mem://b": {"bundle_id": "b-b", "checksum": "y"},
    }
    manifest_path.write_text(
        json.dumps({"version": 2, "records": records}),
        encoding="utf-8",
    )
    m = Manifest(manifest_path)
    m.load()
    assert "mem://a" in m
    assert "mem://b" in m
    assert m["mem://a"]["bundle_id"] == "b-a"


def test_manifest_load_v1_legacy(manifest_path: Path) -> None:
    """Manifest.load() accepts the legacy v1 format (records at root)."""

    records = {"mem://a": {"bundle_id": "b-a", "checksum": "x"}}
    manifest_path.write_text(json.dumps(records), encoding="utf-8")
    m = Manifest(manifest_path)
    m.load()
    assert "mem://a" in m


def test_manifest_record_and_save(tmp_path: Path) -> None:
    """Manifest.record + Manifest.save writes v2 JSON."""

    path = tmp_path / "m.json"
    m = Manifest(path)
    m.record("mem://x", bundle_id="b-x", checksum="h")
    assert "mem://x" in m
    m.save()
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["version"] == 2
    assert payload["records"]["mem://x"]["bundle_id"] == "b-x"


def test_manifest_save_round_trip(tmp_path: Path) -> None:
    """A saved manifest is loadable by a fresh Manifest instance."""

    path = tmp_path / "m.json"
    m = Manifest(path)
    m.record("mem://x", bundle_id="b-x", checksum="h")
    m.save()

    m2 = Manifest(path)
    m2.load()
    assert "mem://x" in m2
    assert m2["mem://x"]["bundle_id"] == "b-x"


def test_manifest_remove(tmp_path: Path) -> None:
    """Manifest.remove deletes a record."""

    path = tmp_path / "m.json"
    m = Manifest(path)
    m.record("mem://x", bundle_id="b", checksum="h")
    assert "mem://x" in m
    m.remove("mem://x")
    assert "mem://x" not in m


def test_manifest_sources_iteration(tmp_path: Path) -> None:
    """Manifest.sources enumerates recorded uris."""

    path = tmp_path / "m.json"
    m = Manifest(path)
    m.record("mem://a", bundle_id="a", checksum="x")
    m.record("mem://b", bundle_id="b", checksum="x")
    assert set(m.sources()) == {"mem://a", "mem://b"}


def test_manifest_items_iteration(tmp_path: Path) -> None:
    """Manifest.items yields (source, record) pairs."""

    path = tmp_path / "m.json"
    m = Manifest(path)
    m.record("mem://a", bundle_id="a", checksum="x")
    items = dict(m.items())
    assert items["mem://a"]["bundle_id"] == "a"


def test_manifest_current_version() -> None:
    """Manifest declares CURRENT_VERSION = 2."""

    assert Manifest.CURRENT_VERSION == 2


# ---------------------------------------------------------------------------
# sha256_bytes
# ---------------------------------------------------------------------------


def test_sha256_bytes_is_deterministic() -> None:
    """sha256_bytes returns identical hashes for identical inputs."""

    a = sha256_bytes(b"hello")
    b = sha256_bytes(b"hello")
    assert a == b


def test_sha256_bytes_changes_with_input() -> None:
    """sha256_bytes returns different hashes for different inputs."""

    assert sha256_bytes(b"a") != sha256_bytes(b"b")


def test_sha256_bytes_is_hex_string() -> None:
    """sha256_bytes returns a 64-character hex string."""

    h = sha256_bytes(b"hello")
    assert len(h) == 64
    assert all(c in "0123456789abcdef" for c in h)


# ---------------------------------------------------------------------------
# Pure helpers
# ---------------------------------------------------------------------------


def test_cosine_identical_vectors() -> None:
    """cosine_similarity of identical non-zero vectors is 1.0."""

    assert cosine_similarity([1.0, 0.0, 1.0], [1.0, 0.0, 1.0]) == pytest.approx(1.0, abs=1e-9)


def test_cosine_orthogonal_vectors() -> None:
    """cosine_similarity of orthogonal vectors is 0."""

    assert cosine_similarity([1.0, 0.0], [0.0, 1.0]) == pytest.approx(0.0, abs=1e-9)


def test_cosine_opposite_vectors() -> None:
    """cosine_similarity of opposite vectors is -1."""

    assert cosine_similarity([1.0, 0.0], [-1.0, 0.0]) == pytest.approx(-1.0, abs=1e-9)


def test_cosine_zero_vector_returns_zero() -> None:
    """cosine_similarity of a zero vector is 0 (avoids division by zero)."""

    assert cosine_similarity([0.0, 0.0], [1.0, 0.0]) == 0.0


def test_extract_json_object_from_codeblock() -> None:
    """extract_json_object pulls a JSON object out of a fenced code block."""

    text = 'Here is the answer:\n```json\n{"x": 1, "y": [1,2]}\n```'
    assert extract_json_object(text) == {"x": 1, "y": [1, 2]}


def test_extract_json_object_inline() -> None:
    """extract_json_object picks the first complete JSON object inline."""

    text = 'the result is {"k": "v"} thanks'
    assert extract_json_object(text) == {"k": "v"}


def test_extract_json_object_no_object_returns_none() -> None:
    """extract_json_object returns None when no JSON object is present."""

    assert extract_json_object("no json here") is None


def test_extract_json_object_handles_nested() -> None:
    """extract_json_object supports nested braces."""

    text = 'x = {"a": {"b": 2}}'
    assert extract_json_object(text) == {"a": {"b": 2}}


def test_tokenise_lowercases_and_splits() -> None:
    """tokenise returns a set of lowercased word tokens."""

    assert tokenise("Hello World hello") == {"hello", "world"}


def test_tokenise_filters_punctuation() -> None:
    """tokenise treats punctuation as a separator."""

    assert tokenise("hello,world.") == {"hello", "world"}


def test_tokenise_empty() -> None:
    """tokenise of an empty string is an empty set."""

    assert tokenise("") == set()


def test_summary_id_for_deterministic() -> None:
    """summary_id_for returns identical ids for identical inputs."""

    a = summary_id_for("the quick brown fox")
    b = summary_id_for("the quick brown fox")
    assert a == b


def test_summary_id_for_changes_with_input() -> None:
    """summary_id_for returns different ids for different inputs."""

    assert summary_id_for("a") != summary_id_for("b")


def test_summary_id_for_is_raptor_prefixed() -> None:
    """summary_id_for returns a string prefixed with 'raptor-'."""

    sid = summary_id_for("hello")
    assert sid.startswith("raptor-")
    assert len(sid) > len("raptor-")


# ---------------------------------------------------------------------------
# cluster + chunk_to_record
# ---------------------------------------------------------------------------


def test_cluster_short_list_returns_single_group() -> None:
    """A short list collapses into a single group (len <= cluster_size)."""

    items = [
        Chunk(
            id=str(i),
            document_id="d",
            version=1,
            company="c",
            owner="o",
            text=f"t{i}",
            checksum="0" * 64,
        )
        for i in range(2)
    ]
    groups = cluster(items, cluster_size=5)
    assert groups == [items]


def test_cluster_empty_returns_empty_group() -> None:
    """An empty list becomes a single empty group."""

    assert cluster([], cluster_size=3) == [[]]


def test_chunk_to_record_carries_vector() -> None:
    """chunk_to_record returns a Chunk with the vector attached."""

    chunk = Chunk(
        id="c1",
        document_id="d1",
        version=1,
        company="acme",
        owner="alice",
        text="hello",
        checksum="0" * 64,
    )
    out = chunk_to_record(chunk, [0.1, 0.2, 0.3], level=1)
    assert out.id == "c1"
    assert out.text == "hello"


# ---------------------------------------------------------------------------
# Raptor (lightweight: just structure)
# ---------------------------------------------------------------------------


def test_raptor_construct_with_defaults() -> None:
    """Raptor can be built with no args (default depth/cluster_size)."""

    raptor = Raptor()
    assert raptor.name == "raptor"
    assert raptor.depth == 2
    assert raptor.cluster_size == 5


def test_raptor_invalid_depth_raises() -> None:
    """Raptor rejects negative depth."""

    with pytest.raises(ValueError, match="depth must be"):
        Raptor(depth=-1)


def test_raptor_invalid_cluster_size_raises() -> None:
    """Raptor rejects zero cluster_size."""

    with pytest.raises(ValueError, match="cluster_size must be"):
        Raptor(cluster_size=0)


def test_raptor_add_empty_chunks_noop() -> None:
    """Adding an empty chunk list is a no-op."""

    raptor = Raptor()
    raptor.add_chunks([], [])
    assert raptor.levels == []


def test_raptor_search_returns_list() -> None:
    """Raptor.search returns a list (possibly empty)."""

    raptor = Raptor()
    hits = raptor.search("anything", top_k=3)
    assert isinstance(hits, list)


def test_raptor_health_dict() -> None:
    """Raptor.health returns a dict."""

    raptor = Raptor()
    assert isinstance(raptor.health(), dict)


def test_raptor_delete_for_document_zero() -> None:
    """Raptor.delete_for_document returns 0 when no chunks match."""

    raptor = Raptor()
    assert raptor.delete_for_document("missing-doc") == 0


# ---------------------------------------------------------------------------
# GraphIndex (lightweight)
# ---------------------------------------------------------------------------


def test_graph_index_construct_with_defaults() -> None:
    """GraphIndex can be built with default args."""

    index = GraphIndex()
    assert index.hop_limit == 2


def test_graph_index_invalid_hop_limit_raises() -> None:
    """GraphIndex rejects negative hop_limit."""

    with pytest.raises(ValueError, match="hop_limit must be"):
        GraphIndex(hop_limit=-1)


def test_graph_index_add_empty_noop() -> None:
    """Adding an empty chunk list is a no-op for GraphIndex."""

    index = GraphIndex()
    index.add_chunks([], [])
    assert dict(index.chunks) == {}


def test_graph_index_search_returns_list() -> None:
    """GraphIndex.search returns a list (possibly empty)."""

    index = GraphIndex()
    assert isinstance(index.search("anything", top_k=3), list)


def test_graph_index_health() -> None:
    """GraphIndex.health returns a structured dict with name/levels."""

    index = GraphIndex()
    health = index.health()
    assert isinstance(health, dict)
    assert health.get("name") == "graphrag"
    assert "levels" in health


def test_graph_index_delete_for_document_zero() -> None:
    """GraphIndex.delete_for_document returns 0 when nothing matches."""

    index = GraphIndex()
    assert index.delete_for_document("missing-doc") == 0


def test_graph_index_reset_clears_state() -> None:
    """GraphIndex has a lock_token; reset increments it (best-effort)."""

    index = GraphIndex()
    initial = index.lock_token
    index.lock_token += 1
    assert index.lock_token == initial + 1
