"""PDF parser.

Wraps :mod:`pypdf` to produce one :class:`ParsedSection` per page.
The page's text is extracted via ``page.extract_text()`` and the
section's metadata carries the page's media-box dimensions so
downstream renderers can preserve aspect ratios.
"""

from __future__ import annotations

from io import BytesIO

from pypdf import PdfReader

from .base import FileParser, ParsedSection


class PdfParser(FileParser):
    """Parser for PDF files using :mod:`pypdf`."""

    def parse(self, file_bytes: bytes, file_name: str, mime_type: str) -> list[ParsedSection]:
        """Parse a PDF into one section per page.

        Args:
            file_bytes: Raw PDF bytes.
            file_name: Original filename (unused beyond diagnostics).
            mime_type: MIME type (unused; the parser always uses
                :class:`pypdf.PdfReader`).

        Returns:
            A list of :class:`ParsedSection`, one per page. Empty
            strings are returned for image-only pages rather than
            raising. The section's ``section_index`` is the 1-based
            page number and ``source_location`` is ``"page N"``. The
            metadata dict contains ``width`` and ``height`` (from the
            page's media box) when available.
        """
        reader = PdfReader(BytesIO(file_bytes))
        sections: list[ParsedSection] = []
        for i, page in enumerate(reader.pages, start=1):
            text = page.extract_text() or ""
            sections.append(
                ParsedSection(
                    section_index=i,
                    source_location=f"page {i}",
                    text=text,
                    metadata={
                        "width": page.mediabox.width,
                        "height": page.mediabox.height,
                    }
                    if page.mediabox
                    else {},
                )
            )
        return sections