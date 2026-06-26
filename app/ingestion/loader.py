"""PDF loading utilities.

LlamaIndex is used here for ingestion only. If it is unavailable, the module
falls back to a small local PDF loader so the rest of the application remains
usable in development.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from pypdf import PdfReader


@dataclass(frozen=True, slots=True)
class LoadedPage:
    """A single loaded PDF page."""

    page_number: int
    text: str


class Loader:
    """Loads PDF pages for ingestion."""

    def load(self, path: Path) -> list[LoadedPage]:
        """Load PDF pages from disk."""

        try:
            from llama_index.core import SimpleDirectoryReader  # type: ignore
        except ImportError:
            return self._fallback_load(path)

        documents = SimpleDirectoryReader(input_files=[str(path)]).load_data()
        loaded_pages: list[LoadedPage] = []
        for index, document in enumerate(documents, start=1):
            text = getattr(document, "text", str(document))
            loaded_pages.append(LoadedPage(page_number=index, text=text))
        return loaded_pages

    def _fallback_load(self, path: Path) -> list[LoadedPage]:
        reader = PdfReader(str(path))
        pages: list[LoadedPage] = []
        for page_number, page in enumerate(reader.pages, start=1):
            pages.append(LoadedPage(page_number=page_number, text=page.extract_text() or ""))
        return pages
