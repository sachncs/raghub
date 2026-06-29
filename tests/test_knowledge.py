"""Tests for OKF round-trip and the in-memory knowledge repository."""

from __future__ import annotations

from raghub.knowledge.okf import dumps, from_okf, loads, to_okf
from raghub.knowledge.repository import InMemoryKnowledgeRepository
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
