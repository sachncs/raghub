"""HTML parser.

Uses :mod:`BeautifulSoup` to extract text from HTML documents. The
text is gathered from the ``<body>`` element (falling back to the
whole document when no body is present) and headings are captured
into the section metadata for later citation work.
"""

from __future__ import annotations

from bs4 import BeautifulSoup

from .base import FileParser, ParsedSection


class HtmlParser(FileParser):
    """Parser for HTML files using :mod:`BeautifulSoup`."""

    def parse(self, file_bytes: bytes, file_name: str, mime_type: str) -> list[ParsedSection]:
        """Parse an HTML document into a single section.

        Args:
            file_bytes: Raw HTML bytes.
            file_name: Original filename (unused).
            mime_type: MIME type (unused).

        Returns:
            A single-element list with one :class:`ParsedSection`
            containing the concatenated body text. ``section_index``
            is 0 and ``source_location`` is ``"full file"``. The
            section metadata includes a ``headings`` list with the
            text of every ``<h1>``, ``<h2>``, and ``<h3>`` element.
        """
        soup = BeautifulSoup(file_bytes, "lxml")
        body = soup.find("body") or soup
        text = body.get_text(separator=" ", strip=True)
        return [
            ParsedSection(
                section_index=0,
                source_location="full file",
                text=text,
                metadata={
                    "headings": [h.get_text(strip=True) for h in soup.find_all(["h1", "h2", "h3"])],
                },
            )
        ]
