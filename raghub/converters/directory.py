"""Convenience helpers for converting on-disk files.

The :func:`convert_path` helper turns a single file into a
:class:`KnowledgeBundle` without going through the full RAG
ingestion pipeline. It is intended for callers that want to use the
OKF conversion layer in isolation, e.g. inside a custom data
preparation step.
"""

from __future__ import annotations

from pathlib import Path

from raghub.converters.marker import MarkerConverter
from raghub.converters.plaintext import PlainTextConverter
from raghub.interfaces.converter import DocumentConverter
from raghub.models import KnowledgeBundle


def select_converter_for_path(path: Path) -> DocumentConverter:
    """Pick a converter for ``path`` based on its extension.

    Args:
        path: File system path.

    Returns:
        A :class:`MarkerConverter` for PDFs and a
        :class:`PlainTextConverter` for everything else. Falls back
        to :class:`PlainTextConverter` for PDFs when Marker is not
        installed.
    """
    if path.suffix.lower() == ".pdf":
        try:
            return MarkerConverter()
        except Exception:
            return PlainTextConverter()
    return PlainTextConverter()


def convert_path(
    path: str | Path,
    *,
    converter: DocumentConverter | None = None,
) -> KnowledgeBundle:
    """Convert a file at ``path`` into a :class:`KnowledgeBundle`.

    Args:
        path: File system path. PDF files are routed to
            :class:`MarkerConverter`; all other files fall back to
            :class:`PlainTextConverter`.
        converter: Optional pre-built converter. When ``None`` a
            converter is selected by extension.

    Returns:
        The canonical :class:`KnowledgeBundle`.
    """
    p = Path(path)
    active = converter or select_converter_for_path(p)
    data = p.read_bytes()
    return active.convert(
        source_uri=str(p.resolve()),
        file_bytes=data,
        mime_type="application/pdf" if p.suffix.lower() == ".pdf" else "text/plain",
    )


__all__ = ["convert_path"]
