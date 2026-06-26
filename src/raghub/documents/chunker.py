"""Multi-format text extraction and chunking."""

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
    """Chunking configuration."""

    chunk_size_words: int = 800
    overlap_words: int = 100


def extract_pdf_pages(pdf_bytes: bytes) -> list[tuple[int, str]]:
    """Extract text per page from a PDF."""

    reader = PdfReader(BytesIO(pdf_bytes))
    pages: list[tuple[int, str]] = []
    for page_index, page in enumerate(reader.pages, start=1):
        pages.append((page_index, page.extract_text() or ""))
    return pages


def normalize_text(text: str) -> str:
    """Normalize whitespace."""

    return " ".join(text.split())


def chunk_words(text: str, plan: ChunkingPlan) -> list[str]:
    """Split normalized text into overlapping word windows."""

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
        start = max(end - plan.overlap_words, start + 1)
    return chunks


def extract_pdf_text(pdf_bytes: bytes) -> list[tuple[int, str, str]]:
    """Extract (page_num, source_location_prefix, text) tuples from a PDF."""

    pages: list[tuple[int, str, str]] = []
    for page_num, text in extract_pdf_pages(pdf_bytes):
        pages.append((page_num, f"page {page_num}", text))
    return pages


def extract_text_from_content(
    file_bytes: bytes,
    file_name: str,
    mime_type: str,
) -> list[tuple[int, str, str]]:
    """Extract text content from a file, returning (page_or_section, source_location, text) tuples.

    Returns:
        List of (section_index, source_location, text) tuples.
        For non-page formats, section_index is 0 and source_location describes the location.
    """

    ext = Path(file_name).suffix.lower()

    if mime_type == "application/pdf" or ext == ".pdf":
        return extract_pdf_text(file_bytes)

    text = file_bytes.decode("utf-8", errors="replace")

    if mime_type in ("text/csv",):
        return [(0, "full file", text)]

    if mime_type.startswith("text/"):
        lines = text.splitlines()
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


def extract_pdf_metadata(pdf_bytes: bytes) -> dict:
    """Extract metadata from a PDF file."""

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
    """Create chunk metadata records from file content."""

    records: list[ChunkRecord] = []
    parsed_sections = extract_text_from_content(file_bytes, file_name, mime_type)

    metadata: dict = {}
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
