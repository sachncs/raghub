"""Document parsing, chunking, validation, lifecycle, and versioning.

This package bundles every concern related to a single document: how
it is parsed, how its text is chunked, how its lifecycle state
advances, and how versions are minted. Callers usually orchestrate
these pieces via :mod:`raghub.ingestion.service` rather than touching
this package directly.
"""

from .chunker import (
    ChunkingPlan,
    build_chunk_records,
    chunk_words,
    extract_pdf_pages,
    normalize_text,
)
from .lifecycle import DocumentLifecycleManager
from .validation import detect_mime_type, validate_upload
from .versioning import new_version

__all__ = [
    "ChunkingPlan",
    "DocumentLifecycleManager",
    "build_chunk_records",
    "chunk_words",
    "detect_mime_type",
    "extract_pdf_pages",
    "new_version",
    "normalize_text",
    "validate_upload",
]