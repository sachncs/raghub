"""Chunker contract.

The strategy that turns a :class:`KnowledgeBundle` (or raw text) into
:class:`raghub.models.Chunk` records. Concrete implementations include
the Chonkie adapter and the built-in word-window chunker.
"""

from __future__ import annotations

from typing import Protocol

from raghub.models import Chunk, KnowledgeBundle


class Chunker(Protocol):
    """Splits a bundle into overlapping chunks."""

    chunk_size: int
    chunk_overlap: int

    def chunk(self, bundle: KnowledgeBundle) -> list[Chunk]:
        """Chunk ``bundle`` into :class:`Chunk` records.

        Args:
            bundle: The source representation. The chunker is free
                to ignore metadata and walk ``bundle.sections``.

        Returns:
            A list of chunks in source order.
        """

    def chunk_text(self, text: str, *, document_id: str, version: int = 1) -> list[Chunk]:
        """Chunk raw ``text`` not wrapped in a bundle.

        Useful for streaming ingestion or evaluation. Default impls
        may raise :class:`NotImplementedError` when not meaningful
        (e.g. table-aware chunkers).

        Args:
            text: The raw text to split.
            document_id: Owning document id.
            version: Owning document version.

        Returns:
            A list of :class:`Chunk` records.
        """
