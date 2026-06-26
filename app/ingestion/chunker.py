"""Chunking utilities for ingestion."""

from __future__ import annotations

from dataclasses import dataclass

from app.ingestion.parser import ParsedPage


@dataclass(frozen=True, slots=True)
class Chunk:
    """A text chunk produced during ingestion."""

    page: int
    text: str


class Chunker:
    """Splits parsed pages into overlapping chunks."""

    def __init__(self, chunk_size: int, overlap: int) -> None:
        self._chunk_size = chunk_size
        self._overlap = overlap

    def chunk(self, pages: list[ParsedPage]) -> list[Chunk]:
        """Chunk the parsed pages."""

        chunks: list[Chunk] = []
        for page in pages:
            chunks.extend(self._chunk_page(page))
        return chunks

    def _chunk_page(self, page: ParsedPage) -> list[Chunk]:
        words = page.text.split()
        if not words:
            return []
        chunks: list[Chunk] = []
        start = 0
        while start < len(words):
            end = min(start + self._chunk_size, len(words))
            text = " ".join(words[start:end]).strip()
            if text:
                chunks.append(Chunk(page=page.page_number, text=text))
            if end >= len(words):
                break
            start = max(end - self._overlap, start + 1)
        return chunks

