"""PDF text extraction, normalisation, and word-window chunking.

This module is the canonical home of the chunker. It exposes:

* :class:`ChunkingPlan` — the chunk-size / overlap configuration.
* :func:`extract_pdf_pages` / :func:`extract_pdf_text` — real per-page
  text extraction for PDF via :mod:`pypdf`.
* :func:`normalize_text` — collapse whitespace runs into single spaces.
* :func:`chunk_words` — overlap-aware word-window chunker.
* :func:`extract_text_from_content` — dispatch by MIME/extension with
  a UTF-8 fallback for non-PDF formats.
* :func:`build_chunk_records` — high-level helper that emits
  :class:`ChunkRecord` objects ready for embedding.

NOTE on multi-format coverage:

:func:`extract_text_from_content` only does **real** extraction for
PDFs (via :func:`extract_pdf_text`). For all other formats the
function currently decodes the raw bytes as UTF-8
(``errors="replace"``) and returns the result as a single section.
Structural parsing for DOCX/XLSX/PPTX is provided by
:class:`raghub.documents.parsers.registry.ParserRegistry` and should
be preferred when those formats are common in the workload.
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from io import BytesIO
from pathlib import Path
from uuid import uuid4

from pypdf import PdfReader

from raghub.models import ChunkRecord, Classification


@dataclass(frozen=True)
class ChunkingPlan:
    """Configuration for the word-window chunker.

    Attributes:
        chunk_size_words: Target number of words per chunk. The actual
            chunk may be slightly smaller when the source has no
            overlap-free remainder.
        overlap_words: Number of words carried over from one chunk to
            the next to preserve cross-boundary context.
    """

    chunk_size_words: int = 800
    overlap_words: int = 100


def extract_pdf_pages(pdf_bytes: bytes) -> list[tuple[int, str]]:
    """Extract text per page from a PDF.

    Args:
        pdf_bytes: The raw PDF bytes.

    Returns:
        A list of ``(page_number, text)`` tuples. Page numbers are
        1-based to match the convention used elsewhere in the
        application (e.g. the citation builder). Empty strings are
        returned for pages with no extractable text rather than
        raising.
    """
    reader = PdfReader(BytesIO(pdf_bytes))
    pages: list[tuple[int, str]] = []
    for page_index, page in enumerate(reader.pages, start=1):
        # ``extract_text`` may return ``None`` for image-only pages;
        # coalesce to ``""`` so downstream code never sees ``None``.
        pages.append((page_index, page.extract_text() or ""))
    return pages


def normalize_text(text: str) -> str:
    """Collapse any run of whitespace into a single space.

    Args:
        text: The input string.

    Returns:
        The whitespace-normalised string. Tabs, newlines, and
        consecutive spaces are all reduced to one space; leading and
        trailing whitespace is stripped.
    """
    return " ".join(text.split())


def chunk_words(text: str, plan: ChunkingPlan) -> list[str]:
    """Split ``text`` into overlapping word windows.

    Algorithm:

    1. Normalise whitespace and split into a list of words.
    2. Slide a window of ``chunk_size_words`` over the word list,
       advancing by ``chunk_size_words - overlap_words`` per step.
    3. Stop when the window reaches the end of the list.

    The ``+ 1`` guard on the step (``start + 1``) ensures we make
    progress even when ``overlap_words >= chunk_size_words`` (a
    misconfiguration that would otherwise loop forever).

    Args:
        text: The text to chunk.
        plan: The :class:`ChunkingPlan` to use.

    Returns:
        A list of chunk strings in source order. Empty when the input
        text has no words.
    """
    words = normalize_text(text).split()
    if not words:
        return []
    chunks: list[str] = []
    start = 0
    while start < len(words):
        end = min(start + plan.chunk_size_words, len(words))
        chunk = " ".join(words[start:end]).strip()
        if chunk:
            chunks.append(chunk)
        if end >= len(words):
            break
        # ``+ 1`` guarantees forward progress even when overlap >=
        # chunk size, which would otherwise produce an infinite loop.
        start = max(end - plan.overlap_words, start + 1)
    return chunks


def extract_pdf_text(pdf_bytes: bytes) -> list[tuple[int, str, str]]:
    """Extract ``(page_num, source_location, text)`` tuples from a PDF.

    Args:
        pdf_bytes: Raw PDF bytes.

    Returns:
        A list of ``(page_num, source_location_prefix, text)`` tuples,
        one per page.
    """
    pages: list[tuple[int, str, str]] = []
    for page_num, text in extract_pdf_pages(pdf_bytes):
        pages.append((page_num, f"page {page_num}", text))
    return pages


def extract_text_from_content(
    file_bytes: bytes,
    file_name: str,
    mime_type: str,
) -> list[tuple[int, str, str]]:
    """Extract text content from a file.

    The dispatch is intentionally coarse: PDFs go through
    :func:`extract_pdf_text`; every other supported MIME/extension
    falls back to a UTF-8 decode of the raw bytes. Use the
    :class:`ParserRegistry` when you need format-aware parsing.

    Args:
        file_bytes: Raw file contents.
        file_name: Original filename; used for extension-based dispatch.
        mime_type: MIME type from the validator.

    Returns:
        A list of ``(section_index, source_location, text)`` tuples.
        For PDFs the section index equals the page number; for all
        other formats the section index is 0 and the source location
        describes the file type (``"full file"``, ``"image"``, etc.).

    NOTE: this function does not call the parser registry — it is the
    lower-level fallback used by the chunker. Use
    :meth:`ParserRegistry.parse` for format-aware structural
    extraction.
    """
    ext = Path(file_name).suffix.lower()

    if mime_type == "application/pdf" or ext == ".pdf":
        return extract_pdf_text(file_bytes)

    text = file_bytes.decode("utf-8", errors="replace")

    if mime_type in ("text/csv",):
        return [(0, "full file", text)]

    if mime_type.startswith("text/"):
        return [(0, "full file", text)]

    if mime_type.startswith("image/"):
        return [(0, "image", text)]

    if mime_type in (
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        "application/msword",
    ):
        return [(0, "document", text)]

    if mime_type in (
        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        "application/vnd.ms-excel",
    ):
        return [(0, "spreadsheet", text)]

    if mime_type in (
        "application/vnd.openxmlformats-officedocument.presentationml.presentation",
        "application/vnd.ms-powerpoint",
    ):
        return [(0, "presentation", text)]

    return [(0, "unknown", text)]


def extract_pdf_metadata(pdf_bytes: bytes) -> dict[str, str]:
    """Extract the standard PDF metadata fields.

    Args:
        pdf_bytes: Raw PDF bytes.

    Returns:
        A dict with ``title``, ``author``, ``producer``, and
        ``creator`` keys (empty strings when missing). An empty dict
        is returned on any parse failure rather than raising.
    """
    try:
        reader = PdfReader(BytesIO(pdf_bytes))
        meta = reader.metadata
        if meta:
            return {
                "title": meta.get("/Title", ""),
                "author": meta.get("/Author", ""),
                "producer": meta.get("/Producer", ""),
                "creator": meta.get("/Creator", ""),
            }
    except Exception:
        # Defensive: malformed PDFs should never block ingestion; the
        # empty dict is harmless and the chunks still extract.
        pass
    return {}


def build_chunk_records(
    *,
    file_bytes: bytes,
    document_id: str,
    version: int,
    company: str,
    owner: str,
    department: str,
    classification: Classification,
    embedding_model: str,
    plan: ChunkingPlan,
    mime_type: str = "",
    file_name: str = "",
) -> list[ChunkRecord]:
    """Build :class:`ChunkRecord` objects for a freshly uploaded file.

    Steps:

    1. Dispatch via :func:`extract_text_from_content` to obtain
       ``(section_index, source_location, text)`` tuples.
    2. For PDFs, harvest the document-level metadata via
       :func:`extract_pdf_metadata` and attach it to every chunk.
    3. Chunk each section's text with :func:`chunk_words`.
    4. Emit one :class:`ChunkRecord` per chunk, hashing the chunk
       text for deduplication.

    Args:
        file_bytes: Raw file contents.
        document_id: Parent document id.
        version: Document version number.
        company: Tenant (company) tag.
        owner: Owning user email.
        department: Department tag (may be empty).
        classification: Sensitivity classification.
        embedding_model: Name of the embedding model that will produce
            vectors for these chunks; recorded for later re-embedding.
        plan: The :class:`ChunkingPlan` to apply.
        mime_type: MIME type from the validator.
        file_name: Original filename.

    Returns:
        A list of :class:`ChunkRecord` objects ready to be persisted
        and embedded.
    """
    records: list[ChunkRecord] = []
    parsed_sections = extract_text_from_content(file_bytes, file_name, mime_type)

    metadata: dict = {}
    # Only PDFs have a real metadata API today; other formats leave
    # the dict empty and the chunk metadata falls through to the
    # default ``{}``.
    if mime_type == "application/pdf":
        metadata.update(extract_pdf_metadata(file_bytes))

    for section_index, source_location, text in parsed_sections:
        for chunk_text in chunk_words(text, plan):
            records.append(
                ChunkRecord(
                    chunk_id=str(uuid4()),
                    document_id=document_id,
                    version=version,
                    page=section_index,
                    source_location=source_location,
                    company=company,
                    owner=owner,
                    department=department,
                    classification=classification,
                    embedding_model=embedding_model,
                    hash=sha256(chunk_text.encode("utf-8")).hexdigest(),
                    text=chunk_text,
                    metadata=metadata,
                )
            )
    return records