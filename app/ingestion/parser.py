"""Parser utilities for ingestion."""

from __future__ import annotations

from dataclasses import dataclass

from app.ingestion.loader import LoadedPage


@dataclass(frozen=True, slots=True)
class ParsedPage:
    """Parsed page content."""

    page_number: int
    text: str


class Parser:
    """Normalizes loaded pages."""

    def parse(self, pages: list[LoadedPage]) -> list[ParsedPage]:
        """Convert loaded pages into parsed pages."""

        return [ParsedPage(page_number=page.page_number, text=" ".join(page.text.split())) for page in pages]

