"""Tests for OKF round-trip and the in-memory knowledge repository."""

from __future__ import annotations

from raghub.knowledge import dumps, from_okf, loads, to_okf
from raghub.knowledge import InMemoryKnowledgeRepository
from raghub.models import (
    BlockKind,
    DocumentBlock,
    DocumentSection,
    KnowledgeBundle,
)


def test_okf_round_trip() -> None:
    """Bundles survive a dumps/loads cycle."""
    bundle = KnowledgeBundle(
        source_uri="file://example",
        sections=[
            DocumentSection(
                index=0,
                heading="Intro",
                page_numbers=[1],
                source_location="page 1",
                blocks=[
                    DocumentBlock(kind=BlockKind.TEXT, content="Hello"),
                    DocumentBlock(kind=BlockKind.TABLE, content="|a|b|"),
                ],
            )
        ],
    )
    restored = loads(dumps(bundle))
    assert restored.bundle_id == bundle.bundle_id
    assert restored.sections[0].blocks[0].kind == BlockKind.TEXT
    assert restored.sections[0].blocks[1].kind == BlockKind.TABLE
    assert restored.sections[0].blocks[0].content == "Hello"


def test_okf_to_dict_shape() -> None:
    """to_okf emits the spec-mandated keys."""
    bundle = KnowledgeBundle(source_uri="file://x")
    payload = to_okf(bundle)
    assert payload["source_uri"] == "file://x"
    assert payload["sections"] == []


def test_okf_rejects_invalid_block_kind() -> None:
    """Unknown block kinds raise KnowledgeError."""
    from raghub.exceptions import KnowledgeError

    payload = {
        "source_uri": "x",
        "sections": [{"index": 0, "blocks": [{"kind": "wat"}]}],
    }
    try:
        from_okf(payload)
    except KnowledgeError:
        return
    raise AssertionError("expected KnowledgeError")


def test_in_memory_repo_save_and_get() -> None:
    """Save then get returns the same bundle."""
    repo = InMemoryKnowledgeRepository()
    bundle = KnowledgeBundle(source_uri="file://y", bundle_id="b1")
    repo.save(bundle)
    assert repo.get("b1") is bundle
    assert repo.list_by_source("file://y") == [bundle]


def test_in_memory_repo_delete() -> None:
    """Delete removes the bundle."""
    repo = InMemoryKnowledgeRepository()
    repo.save(KnowledgeBundle(source_uri="file://z", bundle_id="b1"))
    repo.delete("b1")
    assert repo.get("b1") is None
    assert repo.list_by_source("file://z") == []


def test_in_memory_repo_delete_unknown_is_noop() -> None:
    """Deleting an unknown bundle is a no-op (no exception)."""
    repo = InMemoryKnowledgeRepository()
    repo.delete("nope")  # must not raise


def test_in_memory_repo_save_overwrites_existing() -> None:
    """Saving with the same bundle_id replaces the prior bundle."""
    repo = InMemoryKnowledgeRepository()
    repo.save(KnowledgeBundle(source_uri="file://a", bundle_id="b1"))
    second = KnowledgeBundle(
        source_uri="file://a", bundle_id="b1", metadata={"company": "Acme"}
    )
    repo.save(second)
    assert repo.get("b1") is second


def test_in_memory_repo_list_by_source_returns_matching() -> None:
    repo = InMemoryKnowledgeRepository()
    repo.save(KnowledgeBundle(source_uri="file://a", bundle_id="b1"))
    repo.save(KnowledgeBundle(source_uri="file://a", bundle_id="b2"))
    repo.save(KnowledgeBundle(source_uri="file://b", bundle_id="b3"))
    bundles = repo.list_by_source("file://a")
    assert {b.bundle_id for b in bundles} == {"b1", "b2"}


def test_in_memory_repo_list_by_source_unknown_returns_empty() -> None:
    repo = InMemoryKnowledgeRepository()
    assert repo.list_by_source("file://nope") == []


def test_okf_round_trip_preserves_empty_bundle() -> None:
    """An empty bundle survives a dumps/loads cycle."""
    bundle = KnowledgeBundle(source_uri="file://empty")
    restored = loads(dumps(bundle))
    assert restored.source_uri == "file://empty"
    assert restored.sections == []


def test_okf_round_trip_preserves_metadata() -> None:
    """Bundle metadata is round-tripped through the OKF format."""
    bundle = KnowledgeBundle(
        source_uri="file://meta",
        metadata={"company": "Acme", "owner": "alice"},
    )
    restored = loads(dumps(bundle))
    assert restored.metadata["company"] == "Acme"
    assert restored.metadata["owner"] == "alice"


def test_okf_round_trip_preserves_multiple_sections() -> None:
    """Multiple sections + multiple blocks per section round-trip."""
    bundle = KnowledgeBundle(
        source_uri="file://multi",
        sections=[
            DocumentSection(
                index=0,
                heading="Section 1",
                blocks=[DocumentBlock(kind=BlockKind.TEXT, content="one")],
            ),
            DocumentSection(
                index=1,
                heading="Section 2",
                blocks=[
                    DocumentBlock(kind=BlockKind.TEXT, content="two"),
                    DocumentBlock(kind=BlockKind.TEXT, content="three"),
                ],
            ),
        ],
    )
    restored = loads(dumps(bundle))
    assert len(restored.sections) == 2
    assert [s.heading for s in restored.sections] == ["Section 1", "Section 2"]
    assert [b.content for b in restored.sections[1].blocks] == ["two", "three"]
