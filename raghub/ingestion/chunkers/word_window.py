"""Built-in word-window chunker adapter.

Thin wrapper around the legacy :func:`chunk_words` helper that
exposes it under the :class:`raghub.interfaces.chunker.Chunker`
contract.
"""

from __future__ import annotations

from typing import Any

from raghub.documents.chunker import ChunkingPlan as LegacyPlan
from raghub.documents.chunker import chunk_words as legacy_chunk_words
from raghub.documents.chunker import normalize_text as normalize_text
from raghub.interfaces.chunker import Chunker
from raghub.models import Chunk
from raghub.models.canonical import deterministic_id


class WordWindowChunker(Chunker):
    """Overlap-aware word-window chunker."""

    chunk_size: int
    chunk_overlap: int

    def __init__(
        self,
        *,
        chunk_size: int = 800,
        chunk_overlap: int = 100,
    ) -> None:
        """Initialise the chunker.

        Args:
            chunk_size: Number of words per chunk.
            chunk_overlap: Overlap between consecutive chunks.
        """
        if chunk_size < 1:
            raise ValueError("chunk_size must be >= 1")
        if chunk_overlap < 0 or chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must satisfy 0 <= overlap < chunk_size")
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.plan = LegacyPlan(chunk_size_words=chunk_size, overlap_words=chunk_overlap)

    def chunk(self, bundle: Any) -> list[Chunk]:
        """Chunk ``bundle`` into overlapping windows.

        Args:
            bundle: A :class:`raghub.models.KnowledgeBundle`.

        Returns:
            The list of :class:`Chunk` records.
        """
        chunks: list[Chunk] = []
        for section in bundle.sections:
            for block in section.blocks:
                if not block.kind.value == "text":
                    continue
                for text in self.word_window_chunks(block.content):
                    chunk_id = deterministic_id(
                        "chunk",
                        bundle.source_uri,
                        str(section.index),
                        block.block_id,
                        text[:64],
                    )
                    chunks.append(
                        Chunk(
                            chunk_id=chunk_id,
                            document_id=bundle.bundle_id,
                            version=1,
                            page=(
                                section.page_numbers[0] if section.page_numbers else section.index
                            ),
                            source_location=section.source_location or bundle.source_uri,
                            section=section.heading,
                            company="",
                            owner=bundle.metadata.get("owner", ""),
                            department=bundle.metadata.get("department", ""),
                            text=text,
                            metadata={
                                "block_kind": "text",
                                "block_id": block.block_id,
                                "section_index": section.index,
                            },
                        )
                    )
        return chunks

    def chunk_text(
        self,
        text: str,
        *,
        document_id: str,
        version: int = 1,
        company: str = "",
        owner: str = "",
    ) -> list[Chunk]:
        """Chunk raw ``text`` (no bundle).

        Args:
            text: Raw text.
            document_id: Document id to install on each chunk.
            version: Document version.
            company: Tenant (company) tag; required by :class:`Chunk`.
            owner: Owning user email; required by :class:`Chunk`.

        Returns:
            The list of :class:`Chunk` records.
        """
        result: list[Chunk] = []
        for chunk_text in self.word_window_chunks(text):
            chunk_id = deterministic_id(
                "chunk",
                document_id,
                str(version),
                chunk_text[:64],
            )
            result.append(
                Chunk(
                    chunk_id=chunk_id,
                    document_id=document_id,
                    version=version,
                    company=company,
                    owner=owner,
                    text=chunk_text,
                )
            )
        return result

    def word_window_chunks(self, text: str) -> list[str]:
        """Split ``text`` into overlapping windows."""
        return legacy_chunk_words(normalize_text(text), self.plan)


__all__ = ["WordWindowChunker"]
