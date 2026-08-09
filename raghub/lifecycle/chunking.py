"""Word-window chunking pipeline.

This module owns the operations that turn parsed text into a list of
:class:`raghub.models.Chunk` records: whitespace normalisation, the
overlapping word-window splitter, and the one-stop factory that ties
text extraction, PDF metadata, and the per-chunk SHA-256 checksum
together. It depends on :mod:`raghub.lifecycle.state` for
:class:`ChunkingPlan` and on :mod:`raghub.lifecycle.scanner` for text
extraction, PDF metadata extraction, and the magic-byte helpers.

Public surface:

- :func:`normalize_text` — collapse runs of whitespace.
- :func:`chunk_words` — overlapping word-window splitter.
- :func:`build_chunk_records` — one-stop factory for :class:`Chunk`.
"""

from __future__ import annotations

from hashlib import sha256
from typing import Any
from uuid import uuid4

from raghub.lifecycle.scanner import (
    extract_pdf_metadata,
    extract_text,
)
from raghub.lifecycle.state import ChunkingPlan
from raghub.models import Chunk, Classification


def normalize_text(text: str) -> str:
    """Collapse any run of whitespace into a single space.

    Args:
        text: The input string.

    Returns:
        The whitespace-normalised string.

    """
    return " ".join(text.split())


def chunk_words(text: str, plan: ChunkingPlan) -> list[str]:
    """Split ``text`` into overlapping word windows.

    Args:
        text: The text to chunk.
        plan: The :class:`ChunkingPlan` to use.

    Returns:
        A list of chunk strings in source order.

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
        start = max(end - plan.overlap_words, start + 1)
    return chunks


def build_chunk_records(
    *,
    file_bytes: bytes,
    document_id: str,
    version: int,
    company: str,
    plan: ChunkingPlan,
    **attributes: Any,
) -> list[Chunk]:
    """Build :class:`Chunk` objects for a freshly uploaded file.

    Args:
        file_bytes: Raw file contents.
        document_id: Parent document id.
        version: Document version number.
        company: Owning tenant.
        plan: The :class:`ChunkingPlan` to apply.
        **attributes: ``owner=``, ``department=``,
            ``classification=``, ``embedding_model=``,
            ``mime_type=``, ``file_name=``.

    Returns:
        A list of :class:`Chunk` objects ready to be persisted
        and embedded.

    """
    owner, department, classification, embedding_model, mime_type, file_name = (
        chunk_record_attributes(attributes)
    )
    tenant_id = active_tenant_id()
    metadata = chunk_section_metadata(file_bytes, mime_type)
    parsed_sections = extract_text(file_bytes, file_name, mime_type)

    records: list[Chunk] = []
    for section_index, source_location, text in parsed_sections:
        records.extend(
            build_chunks_for_section(
                text=text,
                section_index=section_index,
                source_location=source_location,
                document_id=document_id,
                version=version,
                company=company,
                owner=owner,
                department=department,
                classification=classification,
                embedding_model=embedding_model,
                tenant_id=tenant_id,
                metadata=metadata,
                plan=plan,
            )
        )
    return records


def chunk_record_attributes(attributes: dict[str, Any]) -> tuple:
    """Return (owner, department, classification, embedding_model, mime_type, file_name)."""
    return (
        attributes.get("owner", ""),
        attributes.get("department", ""),
        attributes.get("classification", Classification.Internal),
        attributes.get("embedding_model", ""),
        attributes.get("mime_type", ""),
        attributes.get("file_name", ""),
    )


def active_tenant_id() -> str:
    """Return the current tenant id, or empty string when no tenant context."""
    from raghub.tenants import current

    ctx = current()
    return ctx.tenant_id if ctx else ""


def chunk_section_metadata(file_bytes: bytes, mime_type: str) -> dict[str, Any]:
    """Return per-section metadata: PDF metadata when applicable, else empty."""
    metadata: dict[str, Any] = {}
    if mime_type == "application/pdf":
        metadata.update(extract_pdf_metadata(file_bytes))
    return metadata


def build_chunks_for_section(
    *,
    text: str,
    section_index: int,
    source_location: str,
    document_id: str,
    version: int,
    company: str,
    owner: str,
    department: str,
    classification: Classification,
    embedding_model: str,
    tenant_id: str,
    metadata: dict[str, Any],
    plan: ChunkingPlan,
) -> list[Chunk]:
    """Build Chunk records for one parsed section by chunking its text."""
    return [
        Chunk(
            id=str(uuid4()),
            document_id=document_id,
            version=version,
            page=section_index,
            source_location=source_location,
            company=company,
            owner=owner,
            department=department,
            classification=classification,
            embedding_model=embedding_model,
            tenant_id=tenant_id,
            checksum=sha256(chunk_text.encode("utf-8", errors="surrogatepass")).hexdigest(),
            text=chunk_text,
            metadata=metadata,
        )
        for chunk_text in chunk_words(text, plan)
    ]


__all__ = [
    "build_chunk_records",
    "chunk_words",
    "normalize_text",
]