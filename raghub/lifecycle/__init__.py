"""Document lifecycle, versioning, validation, chunking, and conversion.

The package is split into focused submodules while preserving the
single ``from raghub.lifecycle import …`` ergonomic for callers:

- :mod:`raghub.lifecycle.state` — Lifecycle state machine, ChunkingPlan,
  and the Markdown → :class:`DocumentBlock` state machine.
- :mod:`raghub.lifecycle.scanner` — file scanning, MIME detection, and
  the word-window chunking pipeline.
- :mod:`raghub.lifecycle.converters` — :class:`DocumentConverter`
  implementations (``PlainTextConverter``, ``Marker``) and the
  extension-keyed dispatch helpers.

Re-exports here keep the historical ``from raghub.lifecycle import X``
imports in callers, tests, and downstream modules working unchanged.
"""

from __future__ import annotations

from raghub.lifecycle.chunking import (
    build_chunk_records,
    chunk_words,
    normalize_text,
)
from raghub.lifecycle.converters import (
    Marker,
    PlainTextConverter,
    build_marker_converter,
    convert_path,
    pick_converter,
)
from raghub.lifecycle.scanner import (
    MAGIC_BYTES,
    MIME_TYPES,
    detect_mime_type,
    extract_pdf_metadata,
    extract_pdf_pages,
    extract_pdf_text,
    extract_text,
    extract_text_fallback,
    looks_like_pdf,
    validate_upload,
)
from raghub.lifecycle.state import (
    ChunkingPlan,
    Lifecycle,
    Section,
    datetime_now_utc,
    md_to_blocks,
    new_version,
    normalise_markdown,
)

__all__ = [
    "MAGIC_BYTES",
    "MIME_TYPES",
    "ChunkingPlan",
    "Lifecycle",
    "Marker",
    "PlainTextConverter",
    "Section",
    "build_chunk_records",
    "build_marker_converter",
    "chunk_words",
    "convert_path",
    "datetime_now_utc",
    "detect_mime_type",
    "extract_pdf_metadata",
    "extract_pdf_pages",
    "extract_pdf_text",
    "extract_text",
    "extract_text_fallback",
    "looks_like_pdf",
    "md_to_blocks",
    "new_version",
    "normalise_markdown",
    "normalize_text",
    "pick_converter",
    "validate_upload",
]