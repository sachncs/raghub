"""Plain-text parser.

Decodes the file as UTF-8 (``errors="replace"``) and returns a single
:class:`ParsedSection`. No structural splitting is performed; callers
that need line- or paragraph-level chunks should use a richer parser.
"""

from __future__ import annotations

from .base import FileParser, ParsedSection


class TxtParser(FileParser):
    """Parser for plain text files."""

    def parse(self, file_bytes: bytes, file_name: str, mime_type: str) -> list[ParsedSection]:
        """Decode text bytes as UTF-8 and emit one whole-file section.

        Args:
            file_bytes: Raw text bytes.
            file_name: Original filename (unused).
            mime_type: MIME type (unused).

        Returns:
            A single-element list containing the decoded text. Invalid
            byte sequences are replaced with the Unicode replacement
            character (``errors="replace"``).
        """
        text = file_bytes.decode("utf-8", errors="replace")
        return [
            ParsedSection(
                section_index=0,
                source_location="full file",
                text=text,
                metadata={},
            )
        ]
