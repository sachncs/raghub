"""Abstract base classes and value objects for document parsers.

The parser interface is intentionally tiny: a parser takes raw bytes
plus identifying metadata and returns a list of
:class:`ParsedSection` records. The chunker later turns these sections
into overlapping word windows for embedding.

Concrete parsers live in sibling modules
(:mod:`.pdf_parser`, :mod:`.html_parser`, :mod:`.image_parser`, ...).
They are aggregated by :class:`raghub.documents.parsers.registry.ParserRegistry`,
which the ingestion pipeline queries by MIME type or file extension.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass(frozen=True)
class ParsedSection:
    """A single parsed chunk of a document.

    Attributes:
        section_index: 0-based ordinal of this section within the file
            (e.g. PDF page number minus 1, or 0 for whole-file formats).
        source_location: Human-readable location string used as the
            ``source_location`` field on chunk records.
        text: The extracted text content.
        metadata: Parser-specific metadata (e.g. PDF ``/Title``,
            ``/Author``). Optional; defaults to an empty dict.
    """

    section_index: int
    source_location: str
    text: str
    metadata: dict


class FileParser(ABC):
    """Abstract base for all document parsers."""

    @abstractmethod
    def parse(self, file_bytes: bytes, file_name: str, mime_type: str) -> list[ParsedSection]:
        """Parse ``file_bytes`` into a list of :class:`ParsedSection`.

        Args:
            file_bytes: Raw file contents.
            file_name: Original filename; useful for format-specific
                hints (e.g. extension-based fallbacks).
            mime_type: The MIME type reported by the validator.

        Returns:
            A list of :class:`ParsedSection` objects. Empty when the
            parser finds no extractable text.
        """
