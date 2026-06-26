from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
from uuid import uuid4

from pypdf import PdfReader


@dataclass
class Chunk:
    chunk_id: str
    document_id: str
    company: str
    filename: str
    page: int
    text: str
    offset: int


def extract_pdf_pages(pdf_bytes: bytes) -> list[tuple[int, str]]:
    reader = PdfReader(BytesIO(pdf_bytes))
    pages: list[tuple[int, str]] = []
    for page_number, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        pages.append((page_number, text))
    return pages


def chunk_text(text: str, chunk_size_words: int, overlap_words: int) -> list[tuple[int, str]]:
    words = text.split()
    if not words:
        return []
    chunks: list[tuple[int, str]] = []
    start = 0
    while start < len(words):
        end = min(start + chunk_size_words, len(words))
        chunk = " ".join(words[start:end]).strip()
        if chunk:
            chunks.append((start, chunk))
        if end >= len(words):
            break
        start = max(end - overlap_words, start + 1)
    return chunks


def build_chunks(
    pdf_bytes: bytes,
    *,
    document_id: str,
    company: str,
    filename: str,
    chunk_size_words: int,
    overlap_words: int,
) -> list[Chunk]:
    chunks: list[Chunk] = []
    for page_number, text in extract_pdf_pages(pdf_bytes):
        for offset, chunk_text_value in chunk_text(text, chunk_size_words, overlap_words):
            chunks.append(
                Chunk(
                    chunk_id=str(uuid4()),
                    document_id=document_id,
                    company=company,
                    filename=filename,
                    page=page_number,
                    text=chunk_text_value,
                    offset=offset,
                )
            )
    return chunks

