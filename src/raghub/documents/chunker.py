"""PDF text extraction and chunking."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from io import BytesIO
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


def build_chunk_records(
    *,
    pdf_bytes: bytes,
    document_id: str,
    version: int,
    company: str,
    owner: str,
    department: str,
    classification: Classification,
    embedding_model: str,
    plan: ChunkingPlan,
) -> list[ChunkRecord]:
    """Create chunk metadata records from a PDF."""

    records: list[ChunkRecord] = []
    for page, text in extract_pdf_pages(pdf_bytes):
        for chunk_text in chunk_words(text, plan):
            records.append(
                ChunkRecord(
                    chunk_id=str(uuid4()),
                    document_id=document_id,
                    version=version,
                    page=page,
                    company=company,
                    owner=owner,
                    department=department,
                    classification=classification,
                    embedding_model=embedding_model,
                    hash=sha256(chunk_text.encode("utf-8")).hexdigest(),
                    text=chunk_text,
                )
            )
    return records
