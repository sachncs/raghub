"""Chunking strategies backed by Chonkie or the built-in word window.

This module exposes the :class:`Words` overlap-aware chunker, the
:class:`Chonkie` chunker with its supported strategies (recursive,
token, sentence, semantic, late, table, code, slumber, neural), and
:func:`build_chonkie_chunker` for strategy dispatch.
"""

from __future__ import annotations

import inspect
from hashlib import sha256
from typing import Any

from raghub.errors import ConfigurationError
from raghub.lifecycle import ChunkingPlan, chunk_words, normalize_text
from raghub.llm import GenerationRequest
from raghub.models import Chunk, Chunker, deterministic_id
from raghub.runtime import capture
from raghub.types import JSONValue

__all__ = [
    "Chonkie",
    "Words",
    "build_chonkie_chunker",
]

chonkie, OptionalImportError = capture(__import__, "chonkie")
CHONKIE_AVAILABLE = OptionalImportError is None
CHONKIE_MODULE = chonkie if CHONKIE_AVAILABLE else None


class Genie:
    """Adapter bridging raghub's LLMProvider to chonkie's Genie interface."""

    def __init__(self, llm_provider: Any) -> None:
        """Wrap an LLMProvider for chonkie's Genie interface."""
        self.llm = llm_provider

    def generate(self, prompt: str) -> str:
        """Generate a chunking response for ``prompt``."""
        return str(
            self.llm.generate(
                GenerationRequest(
                    system_prompt=(
                        "You are a text chunking assistant. "
                        "Split the text at natural boundaries."
                    ),
                    conversation=[],
                    context=[],
                    question=prompt,
                )
            )
        )

    async def agenerate(self, prompt: str) -> str:
        """Generate a chunking response asynchronously."""
        return str(self.generate(prompt))


def build_refinery(context_size: int = 128, tokenizer: str = "character") -> Any:
    """Build an overlap refinery when supported."""
    if CHONKIE_MODULE is None:
        return None
    cls = getattr(CHONKIE_MODULE, "OverlapRefinery", None)
    if cls is None:
        return None
    refinery, error = capture(
        cls, tokenizer=tokenizer, context_size=context_size, merge=True, inplace=True
    )
    return None if isinstance(error, TypeError) else refinery


def apply_refinery(pieces: list[Any], refinery: Any) -> list[Any]:
    """Apply an available refinery to chunks."""
    if refinery is None or not pieces:
        return pieces
    result, error = capture(refinery, pieces)
    return pieces if error is not None else list(result)


def build_chonkie_inner(
    *,
    chunk_size: int,
    chunk_overlap: int,
    **options: JSONValue,
) -> Any:
    """Build the best available Chonkie chunker for the configuration.

    Args:
        chunk_size: Tokens per chunk.
        chunk_overlap: Token overlap.
        **options: ``tokenizer=``, ``chunker_name=``,
            ``embedding_model=``, ``language=``, ``genie=``.

    """
    tokenizer, chunker_name, embedding_model, language, genie = chonkie_options(options)
    if not CHONKIE_AVAILABLE or CHONKIE_MODULE is None:
        raise ConfigurationError(
            "chonkie is not installed; install it via `pip install chonkie` or use Words."
        )

    chunker_builders = chonkie_chunk_builders(
        tokenizer, embedding_model, language, chunk_size, chunk_overlap, genie
    )

    if chunker_name == "auto":
        return auto_select_chunker(
            auto_probe=("RecursiveChunker", "TokenChunker", "SentenceChunker"),
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            tokenizer=tokenizer,
        )

    return instantiate_chonkie_chunker(chunker_name, chunker_builders, chunk_size, chunk_overlap)


def chonkie_options(options: dict[str, JSONValue]) -> tuple[str, str, str, str, Any]:
    """Return (tokenizer, chunker_name, embedding_model, language, genie) from options."""
    return (
        options.get("tokenizer", "character"),
        options.get("chunker_name", "recursive"),
        options.get("embedding_model", "minishlab/potion-base-8M"),
        options.get("language", "auto"),
        options.get("genie"),
    )


def chonkie_chunk_builders(
    tokenizer: str,
    embedding_model: str,
    language: str,
    chunk_size: int,
    chunk_overlap: int,
    genie: Any,
) -> dict[str, tuple[str, dict[str, Any]]]:
    """Return the mapping from chunker-name -> (Chonkie class, kwargs)."""
    return {
        "token": (
            "TokenChunker",
            {"tokenizer": tokenizer, "chunk_size": chunk_size, "chunk_overlap": chunk_overlap},
        ),
        "sentence": (
            "SentenceChunker",
            {"tokenizer": tokenizer, "chunk_size": chunk_size, "chunk_overlap": chunk_overlap},
        ),
        "recursive": ("RecursiveChunker", {"tokenizer": tokenizer, "chunk_size": chunk_size}),
        "semantic": (
            "SemanticChunker",
            {"embedding_model": embedding_model, "chunk_size": chunk_size, "threshold": 0.8},
        ),
        "late": ("LateChunker", {"embedding_model": embedding_model, "chunk_size": chunk_size}),
        "table": ("TableChunker", {"tokenizer": "row", "chunk_size": max(1, chunk_size // 100)}),
        "code": ("CodeChunker", {"language": language, "chunk_size": chunk_size}),
        "neural": ("NeuralChunker", {"min_characters_per_chunk": 24}),
        "slumber": (
            "SlumberChunker",
            {"genie": genie, "chunk_size": chunk_size, "candidate_size": 128},
        ),
    }


def instantiate_chonkie_chunker(
    chunker_name: str,
    chunker_builders: dict[str, tuple[str, dict[str, Any]]],
    chunk_size: int,
    chunk_overlap: int,
) -> Any:
    """Instantiate the requested Chonkie chunker by name, raising on missing deps."""
    if chunker_name not in chunker_builders:
        raise ConfigurationError(f"Unknown chonkie chunker strategy: {chunker_name!r}")
    cls_name, kwargs = chunker_builders[chunker_name]
    cls = getattr(CHONKIE_MODULE, cls_name, None)
    if cls is None:
        raise ConfigurationError(
            f"chonkie chunker {cls_name!r} not available; "
            "install the required extra (e.g. `pip install chonkie[semantic]`)"
        )
    inner, initialization_error = capture(cls, **kwargs)
    if initialization_error is None:
        return inner
    if isinstance(initialization_error, ConfigurationError):
        raise initialization_error
    raise ConfigurationError(
        f"chonkie {cls_name} failed to initialize: {initialization_error}"
    ) from initialization_error


def auto_select_chunker(
    *,
    auto_probe: tuple[str, ...],
    chunk_size: int,
    chunk_overlap: int,
    tokenizer: str,
) -> Any:
    """Pick the first Chonkie chunker that accepts the auto-probed kwargs."""
    for cls_name in auto_probe:
        cls = getattr(CHONKIE_MODULE, cls_name, None)
        if cls is None:
            continue
        sig, signature_error = capture(inspect.signature, cls)
        if isinstance(signature_error, (TypeError, ValueError)):
            sig = None
        kwargs: dict[str, Any] = {}
        if sig is not None:
            params = sig.parameters
            for key, value in (
                ("tokenizer", tokenizer),
                ("tokenizer_or_token_counter", tokenizer),
                ("chunk_size", chunk_size),
                ("chunk_overlap", chunk_overlap),
                ("return_type", "chunks"),
            ):
                if key in params:
                    kwargs[key] = value
        inner, initialization_error = capture(cls, **kwargs)
        if initialization_error is None:
            return inner
        if not isinstance(initialization_error, TypeError):
            raise initialization_error
    raise ConfigurationError(
        "chonkie is installed but no documented chunker accepted the "
        "configuration; please check the installed chonkie version."
    )


class Chonkie(Chunker):
    """Chonkie-backed chunker supporting all strategies."""

    chunk_size: int
    chunk_overlap: int

    def __init__(
        self,
        *,
        chunk_size: int = 512,
        chunk_overlap: int = 64,
        **options: JSONValue,
    ) -> None:
        """Initialise the Chonkie chunker.

        Args:
            chunk_size: Tokens per chunk.
            chunk_overlap: Token overlap.
            **options: Optional overrides — ``tokenizer=``,
                ``chunker_name=``, ``embedding_model=``,
                ``language=``, ``llm_provider=`` (for
                SlumberChunker).

        """
        if not CHONKIE_AVAILABLE:
            raise ConfigurationError(
                "chonkie is not installed; install it via `pip install chonkie` or use Words."
            )
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        chunker_name = options.get("chunker_name", "recursive")
        genie = None
        if chunker_name == "slumber":
            llm_provider = options.get("llm_provider")
            if llm_provider is None:
                raise ConfigurationError(
                    "SlumberChunker requires an LLM provider; pass llm_provider="
                )
            genie = Genie(llm_provider)

        self.inner = build_chonkie_inner(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            tokenizer=options.get("tokenizer", "character"),
            chunker_name=options.get("chunker_name", "recursive"),
            embedding_model=options.get("embedding_model", "minishlab/potion-base-8M"),
            language=options.get("language", "auto"),
            genie=genie,
        )
        self.refinery = build_refinery(
            context_size=chunk_overlap, tokenizer=options.get("tokenizer", "character")
        )

    def chonkie_text_chunks(self, text: str) -> list[Any]:
        """Invoke the underlying Chonkie chunker; tolerate API drift."""
        pieces, invocation_error = capture(self.inner, text)
        if isinstance(invocation_error, TypeError):
            chunk = getattr(self.inner, "chunk", None) or getattr(self.inner, "split_text", None)
            if chunk is None:
                raise invocation_error
            pieces = chunk(text)
        elif invocation_error is not None:
            raise invocation_error
        return apply_refinery(pieces, self.refinery)

    def chonkie_batch_chunks(self, texts: list[str]) -> list[list[Any]]:
        """Chunk multiple texts at once via chonkie.Pipeline when available."""
        if not texts:
            return []
        return [self.chonkie_text_chunks(text) for text in texts]

    def chunk(self, bundle: Any) -> list[Chunk]:
        """Chunk a bundle via Chonkie."""
        from raghub.tenants import current

        ctx = current()
        tenant_id = ctx.tenant_id if ctx else ""
        chunks: list[Chunk] = []
        for section in bundle.sections:
            for block in section.blocks:
                if block.kind.value != "text":
                    continue
                pieces = self.chonkie_text_chunks(block.content)
                for piece in pieces:
                    text: str = (
                        getattr(piece, "text", None)
                        or (piece.get("text") if isinstance(piece, dict) else str(piece))
                        or ""
                    )
                    chunk_id = (
                        getattr(piece, "id", None)
                        or (piece.get("id") if isinstance(piece, dict) else None)
                        or f"{bundle.bundle_id}:{section.index}:{block.block_id}:{len(chunks)}"
                    )
                    chunks.append(
                        Chunk(
                            id=chunk_id,
                            document_id=bundle.bundle_id,
                            version=1,
                            page=(
                                section.page_numbers[0] if section.page_numbers else section.index
                            ),
                            source_location=section.source_location or bundle.source_uri,
                            section=section.heading,
                            company="",
                            owner=bundle.metadata.get("owner", ""),
                            department=bundle.metadata.get("department", ""),
                            tenant_id=tenant_id,
                            text=text,
                            checksum=sha256(text.encode("utf-8", errors="surrogatepass")).hexdigest(),
                            metadata={
                                "chunker": "chonkie",
                                "strategy": getattr(self.inner, "__class__", type(None)).__name__,
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
        """Chunk raw ``text`` via Chonkie."""
        from raghub.tenants import current

        ctx = current()
        tenant_id = ctx.tenant_id if ctx else ""
        pieces = self.chonkie_text_chunks(text)
        chunks: list[Chunk] = []
        for i, piece in enumerate(pieces):
            text_value: str = (
                getattr(piece, "text", None)
                or (piece.get("text") if isinstance(piece, dict) else str(piece))
                or ""
            )
            chunk_id = (
                getattr(piece, "id", None)
                or (piece.get("id") if isinstance(piece, dict) else None)
                or f"{document_id}:v{version}:{i}"
            )
            chunks.append(
                Chunk(
                    id=chunk_id,
                    document_id=document_id,
                    version=version,
                    company=company,
                    owner=owner,
                    tenant_id=tenant_id,
                    text=text_value,
                    checksum=sha256(text_value.encode("utf-8", errors="surrogatepass")).hexdigest(),
                    metadata={
                        "chunker": "chonkie",
                        "strategy": getattr(self.inner, "__class__", type(None)).__name__,
                    },
                )
            )
        return chunks


class Words(Chunker):
    """Overlap-aware word-window chunker."""

    chunk_size: int
    chunk_overlap: int

    def __init__(
        self,
        *,
        chunk_size: int = 800,
        chunk_overlap: int = 100,
    ) -> None:
        """Initialise the chunker.

        Args:
            chunk_size: Number of words per chunk.
            chunk_overlap: Overlap between consecutive chunks.

        """
        if chunk_size < 1:
            raise ValueError("chunk_size must be >= 1")
        if chunk_overlap < 0 or chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must satisfy 0 <= overlap < chunk_size")
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.plan = ChunkingPlan(chunk_size_words=chunk_size, overlap_words=chunk_overlap)

    def chunk(self, bundle: Any) -> list[Chunk]:
        """Chunk ``bundle`` into overlapping windows."""
        from raghub.tenants import current

        ctx = current()
        tenant_id = ctx.tenant_id if ctx else ""
        chunks: list[Chunk] = []
        for section in bundle.sections:
            for block in section.blocks:
                if block.kind.value != "text":
                    continue
                for text in self.word_window_chunks(block.content):
                    chunk_id = deterministic_id(
                        "chunk",
                        bundle.source_uri,
                        str(section.index),
                        block.block_id,
                        text[:64],
                    )
                    chunks.append(
                        Chunk(
                            id=chunk_id,
                            document_id=bundle.bundle_id,
                            version=1,
                            page=(
                                section.page_numbers[0] if section.page_numbers else section.index
                            ),
                            source_location=section.source_location or bundle.source_uri,
                            section=section.heading,
                            company="",
                            owner=bundle.metadata.get("owner", ""),
                            department=bundle.metadata.get("department", ""),
                            tenant_id=tenant_id,
                            text=text,
                            checksum=sha256(text.encode("utf-8", errors="surrogatepass")).hexdigest(),
                            metadata={
                                "block_kind": "text",
                                "block_id": block.block_id,
                                "section_index": section.index,
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
        """Chunk raw ``text`` (no bundle)."""
        from raghub.tenants import current

        ctx = current()
        tenant_id = ctx.tenant_id if ctx else ""
        result: list[Chunk] = []
        for chunk_text in self.word_window_chunks(text):
            chunk_id = deterministic_id(
                "chunk",
                document_id,
                str(version),
                chunk_text[:64],
            )
            result.append(
                Chunk(
                    id=chunk_id,
                    document_id=document_id,
                    version=version,
                    company=company,
                    owner=owner,
                    tenant_id=tenant_id,
                    text=chunk_text,
                    checksum=sha256(chunk_text.encode("utf-8", errors="surrogatepass")).hexdigest(),
                )
            )
        return result

    def word_window_chunks(self, text: str) -> list[str]:
        """Split ``text`` into overlapping windows."""
        return chunk_words(normalize_text(text), self.plan)


def build_chonkie_chunker(name: str = "auto", **kwargs: JSONValue) -> Chunker:
    """Pick a chunker by name.

    Args:
        name: Chunker strategy.
        **kwargs: Forwarded to the underlying constructor.

    Returns:
        A configured :class:`Chunker`.

    Raises:
        ConfigurationError: When ``name`` is unknown or chonkie is
            explicitly requested but unavailable.

    """
    chonkie_names = {
        "auto",
        "recursive",
        "token",
        "sentence",
        "semantic",
        "late",
        "table",
        "code",
        "slumber",
        "neural",
    }
    if name in chonkie_names:
        if CHONKIE_AVAILABLE:
            return Chonkie(chunker_name=name, **kwargs)
        if name != "auto":
            raise ConfigurationError("chonkie is not installed")
    if name in {"chonkie", "word_window", "auto"}:
        if name == "chonkie":
            if CHONKIE_AVAILABLE:
                return Chonkie(**kwargs)
            raise ConfigurationError("chonkie is not installed")
        return Words(**kwargs)
    raise ConfigurationError(f"Unknown chunker: {name!r}")
