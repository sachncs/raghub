"""CSV parser.

Decodes CSV bytes as UTF-8 and returns the whole file as a single
:class:`ParsedSection`. Structural column/row parsing is left to
downstream consumers; the parser does not attempt to interpret the
delimiter.
"""

from __future__ import annotations

from .base import FileParser, ParsedSection


class CsvParser(FileParser):
    """Parser for CSV files (UTF-8 decoded, no structural splitting)."""

    def parse(self, file_bytes: bytes, file_name: str, mime_type: str) -> list[ParsedSection]:
        """Decode CSV bytes as UTF-8 and emit one whole-file section.

        Args:
            file_bytes: Raw CSV bytes.
            file_name: Original filename (unused).
            mime_type: MIME type (unused).

        Returns:
            A single-element list containing the full decoded text.
            Invalid byte sequences are replaced with the Unicode
            replacement character (``errors="replace"``).
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
