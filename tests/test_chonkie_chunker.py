"""Tests for the Chonkie chunker wrapper."""

from __future__ import annotations

import pytest

from raghub.ingestion.chunkers.chonkie import (
    ChonkieChunker,
    build_chonkie_chunker,
)
from raghub.models import BlockKind, DocumentBlock, DocumentSection, KnowledgeBundle

pytestmark = pytest.mark.skipif(
    not pytest.importorskip("chonkie", reason="chonkie not installed"),
    reason="chonkie is the system under test",
)


def _bundle(text: str) -> KnowledgeBundle:
    return KnowledgeBundle(
        source_uri="file://x.txt",
        sections=[
            DocumentSection(
                index=0,
                blocks=[DocumentBlock(kind=BlockKind.TEXT, content=text)],
            )
        ],
    )


def test_build_chonkie_chunker_auto_uses_chonkie() -> None:
    """``build_chonkie_chunker('auto')`` returns a ChonkieChunker."""
    chunker = build_chonkie_chunker("auto", chunk_size=16, chunk_overlap=2)
    assert isinstance(chunker, ChonkieChunker)


def test_build_chonkie_chunker_explicit_word_window() -> None:
    """``build_chonkie_chunker('word_window')`` returns a WordWindowChunker."""
    from raghub.ingestion.chunkers.word_window import WordWindowChunker

    chunker = build_chonkie_chunker("word_window", chunk_size=16, chunk_overlap=2)
    assert isinstance(chunker, WordWindowChunker)


def test_build_chonkie_chunker_unknown_raises() -> None:
    """Unknown chunker names raise :class:`ConfigurationError`."""
    from raghub.exceptions import ConfigurationError

    with pytest.raises(ConfigurationError):
        from typing import cast

        build_chonkie_chunker(cast(str, "not-a-real-chunker"))


def test_chonkie_chunker_chunk_handles_bundle() -> None:
    """``ChonkieChunker.chunk`` returns chunks for a bundle."""
    chunker = ChonkieChunker(chunk_size=8, chunk_overlap=1)
    text = "Hello world. " * 30
    bundle = _bundle(text)
    chunks = chunker.chunk(bundle)
    assert chunks
    assert all(c.document_id == bundle.bundle_id for c in chunks)
    assert all(c.text.strip() for c in chunks)
    assert all(c.metadata["chunker"] == "chonkie" for c in chunks)


def test_chonkie_chunker_chunk_text_returns_chunks() -> None:
    """``ChonkieChunker.chunk_text`` returns :class:`Chunk` records."""
    chunker = ChonkieChunker(chunk_size=8, chunk_overlap=1)
    text = "Tokenized " * 20
    chunks = chunker.chunk_text(text, document_id="doc1", company="Apple")
    assert chunks
    assert all(c.document_id == "doc1" for c in chunks)
    assert all(c.company == "Apple" for c in chunks)
    assert all(c.metadata["chunker"] == "chonkie" for c in chunks)


def test_chonkie_chunker_skips_non_text_blocks() -> None:
    """Non-text blocks produce no chunks."""
    chunker = ChonkieChunker(chunk_size=8, chunk_overlap=1)
    bundle = KnowledgeBundle(
        source_uri="file://x.txt",
        sections=[
            DocumentSection(
                index=0,
                blocks=[
                    DocumentBlock(kind=BlockKind.IMAGE, content="fig-1.png"),
                    DocumentBlock(kind=BlockKind.TABLE, content="|a|b|"),
                ],
            )
        ],
    )
    assert chunker.chunk(bundle) == []
