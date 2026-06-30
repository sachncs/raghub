"""Chonkie-backed chunker adapter.

Chonkie is the spec default for the ingestion stage. It supports
token-based chunking, semantic chunking, and overlap strategies out
of the box.

The import of ``chonkie`` is deferred to keep the base import graph
small. If Chonkie is not installed, :class:`ChonkieChunker.__init__`
raises :class:`raghub.exceptions.ConfigurationError`.

**API note:** Chonkie's public entry point has changed across major
versions. The adapter tries the documented classes
(:class:`chonkie.TokenChunker`, :class:`chonkie.SentenceChunker`,
:class:`chonkie.RecursiveChunker`) in order and falls back to the
first one that accepts the configured ``chunk_size`` and
``chunk_overlap``. If none of the documented chunkers match, the
adapter raises :class:`raghub.exceptions.ConfigurationError` with a
message that names the supported versions.
"""

from __future__ import annotations

import inspect
from typing import Any

from raghub.exceptions import ConfigurationError
from raghub.ingestion.chunkers.word_window import WordWindowChunker
from raghub.interfaces.chunker import Chunker
from raghub.models import Chunk

chonkie: Any

try:
    chonkie = __import__("chonkie")
    CHONKIE_AVAILABLE = True
    CHONKIE_MODULE = chonkie
    OptionalImportError: Exception | None = None
except Exception as exc:  # pragma: no cover - optional dep
    chonkie = None
    CHONKIE_MODULE = None
    CHONKIE_AVAILABLE = False
    OptionalImportError = exc


def build_chonkie_inner(*, chunk_size: int, chunk_overlap: int, tokenizer: str) -> Any:
    """Build the best available Chonkie chunker for the configuration."""
    if not CHONKIE_AVAILABLE or CHONKIE_MODULE is None:
        raise ConfigurationError(
            "chonkie is not installed; install it via `pip install chonkie` "
            "or use WordWindowChunker."
        )

    candidate_names = ("TokenChunker", "SentenceChunker", "RecursiveChunker", "Chunker")
    for name in candidate_names:
        cls = getattr(CHONKIE_MODULE, name, None)
        if cls is None:
            continue
        try:
            sig = inspect.signature(cls)
        except (TypeError, ValueError):
            sig = None
        kwargs: dict[str, Any] = {}
        if sig is not None:
            params = sig.parameters
            for key, value in (
                ("tokenizer", tokenizer),
                ("chunk_size", chunk_size),
                ("chunk_overlap", chunk_overlap),
                ("chunk_overlap_tokens", chunk_overlap),
                ("return_type", "chunks"),
            ):
                if key in params:
                    kwargs[key] = value
        try:
            return cls(**kwargs)
        except TypeError:
            continue
    raise ConfigurationError(
        "chonkie is installed but no documented chunker accepted the "
        "configuration; please check the installed chonkie version."
    )


class ChonkieChunker(Chunker):
    """Token chunker backed by Chonkie."""

    chunk_size: int
    chunk_overlap: int

    def __init__(
        self,
        *,
        chunk_size: int = 512,
        chunk_overlap: int = 64,
        tokenizer: str = "character",
    ) -> None:
        """Initialise the Chonkie chunker.

        Args:
            chunk_size: Tokens per chunk.
            chunk_overlap: Token overlap.
            tokenizer: Tokenizer name (``"character"``, ``"gpt2"``, …).
        """
        if not CHONKIE_AVAILABLE:
            raise ConfigurationError(
                "chonkie is not installed; install it via `pip install chonkie` "
                "or use WordWindowChunker."
            )
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.inner = build_chonkie_inner(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            tokenizer=tokenizer,
        )

    def chonkie_text_chunks(self, text: str) -> list[Any]:
        """Invoke the underlying Chonkie chunker; tolerate API drift."""
        try:
            return self.inner(text)
        except TypeError:
            # Older Chonkie takes a string directly via ``.chunk()`` or
            # ``.split_text()``.
            chunk = getattr(self.inner, "chunk", None) or getattr(
                self.inner, "split_text", None
            )
            if chunk is not None:
                return chunk(text)
            raise

    def chunk(self, bundle: Any) -> list[Chunk]:
        """Chunk a bundle via Chonkie.

        Args:
            bundle: The source bundle.

        Returns:
            The list of :class:`Chunk` records.
        """
        chunks: list[Chunk] = []
        for section in bundle.sections:
            for block in section.blocks:
                if not block.kind.value == "text":
                    continue
                pieces = self.chonkie_text_chunks(block.content)
                for piece in pieces:
                    text: str = getattr(piece, "text", None) or (
                        piece.get("text") if isinstance(piece, dict) else str(piece)
                    ) or ""
                    chunk_id = (
                        getattr(piece, "id", None)
                        or (piece.get("id") if isinstance(piece, dict) else None)
                        or f"{bundle.bundle_id}:{section.index}:{block.block_id}:{len(chunks)}"
                    )
                    chunks.append(
                        Chunk(
                            chunk_id=chunk_id,
                            document_id=bundle.bundle_id,
                            version=1,
                            page=(section.page_numbers[0] if section.page_numbers else section.index),
                            source_location=section.source_location or bundle.source_uri,
                            section=section.heading,
                            company="",
                            owner=bundle.metadata.get("owner", ""),
                            department=bundle.metadata.get("department", ""),
                            text=text,
                            metadata={
                                "chunker": "chonkie",
                                "section_index": section.index,
                                "block_id": block.block_id,
                            },
                        )
                    )
        return chunks

    def chunk_text(
        self,
        text: str,
        *,
        document_id: str,
        version: int = 1,
        company: str = "",
        owner: str = "",
    ) -> list[Chunk]:
        """Chunk raw ``text`` via Chonkie.

        Args:
            text: The raw text.
            document_id: Document id to install on each chunk.
            version: Document version.
            company: Tenant (company) tag; required by :class:`Chunk`.
            owner: Owning user email; required by :class:`Chunk`.

        Returns:
            The list of :class:`Chunk` records.
        """
        pieces = self.chonkie_text_chunks(text)
        chunks: list[Chunk] = []
        for i, piece in enumerate(pieces):
            text_value: str = getattr(piece, "text", None) or (
                piece.get("text") if isinstance(piece, dict) else str(piece)
            ) or ""
            chunk_id = (
                getattr(piece, "id", None)
                or (piece.get("id") if isinstance(piece, dict) else None)
                or f"{document_id}:v{version}:{i}"
            )
            chunks.append(
                Chunk(
                    chunk_id=chunk_id,
                    document_id=document_id,
                    version=version,
                    company=company,
                    owner=owner,
                    text=text_value,
                    metadata={"chunker": "chonkie"},
                )
            )
        return chunks


def build_chonkie_chunker(name: str = "auto", **kwargs: Any) -> Chunker:
    """Pick a chunker by name.

    Args:
        name: ``"chonkie"`` / ``"word_window"`` / ``"auto"``.
        **kwargs: Forwarded to the underlying constructor.

    Returns:
        A configured :class:`Chunker`.

    Raises:
        ConfigurationError: When ``name`` is unknown or chonkie is
            explicitly requested but unavailable.
    """
    if name in ("chonkie", "auto"):
        if CHONKIE_AVAILABLE:
            return ChonkieChunker(**kwargs)
        if name == "chonkie":
            raise ConfigurationError("chonkie is not installed")
    if name in ("word_window", "auto"):
        return WordWindowChunker(**kwargs)
    raise ConfigurationError(f"Unknown chunker: {name!r}")


__all__ = ["ChonkieChunker", "build_chonkie_chunker"]
