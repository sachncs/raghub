"""Document parsing, chunking, validation, lifecycle, versioning, and conversion.

Public surface is re-exported from this package; the implementation
lives in two sibling modules::

    raghub.documents.parser
        format-specific :class:`File` classes, the
        :class:`Catalog` that aggregates them, and the unified
        :func:`parse` entry point.

    raghub.documents.helper
        :class:`DocumentLifecycleManager`, versioning,
        MIME detection, the word-window chunker, the PDF extractor,
        the :class:`MarkdownSection` state machine, and the
        Marker / PlainText converters.

Callers usually orchestrate these pieces via :mod:`raghub.ingestion`
rather than touching this package directly. The :func:`parse` function
is the recommended entry point for ad-hoc dispatch.
"""

from __future__ import annotations

import sys

from raghub.documents.helper import (
    MARKER_AVAILABLE,
    ChunkingPlan,
    DocumentLifecycleManager,
    MarkdownSection,
    MarkerConverter,
    PlainTextConverter,
    build_chunk_records,
    build_marker_converter,
    chunk_words,
    convert_path,
    datetime_now_utc,
    detect_mime_type,
    extract_pdf_metadata,
    extract_pdf_pages,
    extract_pdf_text,
    extract_text_from_content,
    looks_like_pdf,
    markdown_to_document_blocks,
    new_version,
    normalise_markdown,
    normalize_text,
    select_converter_for_path,
    validate_upload,
)
from raghub.documents.parser import (
    Catalog,
    Csv,
    File,
    HTML,
    Image,
    Office,
    Pdf,
    Registry,
    Section,
    Txt,
    parse,
)

# Aliases so legacy ``from raghub.documents import directory as
# directory_module`` / ``marker as marker_module`` imports still work
# after the flatten. The patches in tests/test_converters.py rely on
# these aliases resolving to the same module.
sys.modules.setdefault("raghub.documents.directory", sys.modules[__name__])
sys.modules.setdefault("raghub.documents.marker", sys.modules[__name__])
sys.modules.setdefault("raghub.documents.markdown", sys.modules[__name__])
sys.modules.setdefault("raghub.documents.plaintext", sys.modules[__name__])

directory: object = sys.modules[__name__]
marker: object = sys.modules[__name__]
markdown: object = sys.modules[__name__]
plaintext: object = sys.modules[__name__]


__all__ = [
    # Parser surface (parser.py)
    "File",
    "Section",
    "Pdf",
    "HTML",
    "Image",
    "Office",
    "Csv",
    "Txt",
    "Catalog",
    "Registry",
    "parse",
    # Lifecycle / versioning / validation (helper.py)
    "DocumentLifecycleManager",
    "new_version",
    "datetime_now_utc",
    "detect_mime_type",
    "validate_upload",
    # Chunking (helper.py)
    "ChunkingPlan",
    "chunk_words",
    "normalize_text",
    "extract_pdf_pages",
    "extract_pdf_text",
    "extract_pdf_metadata",
    "extract_text_from_content",
    "build_chunk_records",
    # Markdown (helper.py)
    "MarkdownSection",
    "markdown_to_document_blocks",
    "normalise_markdown",
    # Converters (helper.py)
    "MARKER_AVAILABLE",
    "PlainTextConverter",
    "MarkerConverter",
    "looks_like_pdf",
    "build_marker_converter",
    "select_converter_for_path",
    "convert_path",
] 
