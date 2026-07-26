"""Chonkie-backed chunker adapter.

Chonkie is the spec default for the ingestion stage. It supports
token-based chunking, semantic chunking, and overlap strategies out
of the box.

The import of ``chonkie`` is deferred to keep the base import graph
small. If Chonkie is not installed, :class:`ChonkieChunker.__init__`
raises :class:`raghub.exceptions.ConfigurationError`.

Supported chunker strategies: recursive, token, sentence, word,
semantic, late, table, code, slumber, neural.
"""

from __future__ import annotations

import inspect
from typing import Any

from raghub.exceptions import ConfigurationError
from raghub.ingestion.chunkers.word_window import WordWindowChunker
from raghub.interfaces.chunker import Chunker
from raghub.models import Chunk
from raghub.utils.execution import capture

chonkie, OptionalImportError = capture(__import__, "chonkie")
CHONKIE_AVAILABLE = OptionalImportError is None
CHONKIE_MODULE = chonkie if CHONKIE_AVAILABLE else None


class RAGHubGenie:
    """Adapter bridging raghub's LLMProvider to chonkie's Genie interface.

    Chonkie's SlumberChunker expects a ``Genie`` with a ``generate(prompt) -> str``
    method. This thin wrapper delegates to whatever raghub LLM provider is configured.
    """

    def __init__(self, llm_provider: Any) -> None:
        self.llm = llm_provider

    def generate(self, prompt: str) -> str:
        """Generate a chunking response for ``prompt``."""
        return str(self.llm.generate(
            system_prompt="You are a text chunking assistant. Split the text at natural boundaries.",
            conversation=[],
            context=[],
            question=prompt,
        ))

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
    tokenizer: str = "character",
    chunker_name: str = "recursive",
    embedding_model: str = "minishlab/potion-base-8M",
    language: str = "auto",
    genie: Any = None,
) -> Any:
    """Build the best available Chonkie chunker for the configuration."""
    if not CHONKIE_AVAILABLE or CHONKIE_MODULE is None:
        raise ConfigurationError(
            "chonkie is not installed; install it via `pip install chonkie` "
            "or use WordWindowChunker."
        )

    chunker_builders: dict[str, tuple[str, dict[str, Any]]] = {
        "token": ("TokenChunker", {"tokenizer": tokenizer, "chunk_size": chunk_size, "chunk_overlap": chunk_overlap}),
        "sentence": ("SentenceChunker", {"tokenizer": tokenizer, "chunk_size": chunk_size, "chunk_overlap": chunk_overlap}),
        "recursive": ("RecursiveChunker", {"tokenizer": tokenizer, "chunk_size": chunk_size}),
        "semantic": ("SemanticChunker", {"embedding_model": embedding_model, "chunk_size": chunk_size, "threshold": 0.8}),
        "late": ("LateChunker", {"embedding_model": embedding_model, "chunk_size": chunk_size}),
        "table": ("TableChunker", {"tokenizer": "row", "chunk_size": max(1, chunk_size // 100)}),
        "code": ("CodeChunker", {"language": language, "chunk_size": chunk_size}),
        "neural": ("NeuralChunker", {"min_characters_per_chunk": 24}),
        "slumber": ("SlumberChunker", {"genie": genie, "chunk_size": chunk_size, "candidate_size": 128}),
    }

    auto_probe = ("RecursiveChunker", "TokenChunker", "SentenceChunker")

    if chunker_name == "auto":
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


class ChonkieChunker(Chunker):
    """Chonkie-backed chunker supporting all strategies."""

    chunk_size: int
    chunk_overlap: int

    def __init__(
        self,
        *,
        chunk_size: int = 512,
        chunk_overlap: int = 64,
        tokenizer: str = "character",
        chunker_name: str = "recursive",
        embedding_model: str = "minishlab/potion-base-8M",
        language: str = "auto",
        llm_provider: Any = None,
    ) -> None:
        """Initialise the Chonkie chunker.

        Args:
            chunk_size: Tokens per chunk.
            chunk_overlap: Token overlap.
            tokenizer: Tokenizer name (``"character"``, ``"gpt2"``, …).
            chunker_name: Chunking strategy (``"recursive"``, ``"token"``,
                ``"sentence"``, ``"semantic"``, ``"late"``, ``"table"``,
                ``"code"``, ``"word"``, ``"slumber"``, ``"neural"``,
                ``"auto"``).
            embedding_model: Model for semantic/late chunkers.
            language: Language for CodeChunker.
            llm_provider: raghub LLM provider for SlumberChunker.
        """
        if not CHONKIE_AVAILABLE:
            raise ConfigurationError(
                "chonkie is not installed; install it via `pip install chonkie` "
                "or use WordWindowChunker."
            )
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        genie = None
        if chunker_name == "slumber":
            if llm_provider is None:
                raise ConfigurationError(
                    "SlumberChunker requires an LLM provider; pass llm_provider="
                )
            genie = RAGHubGenie(llm_provider)

        self.inner = build_chonkie_inner(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            tokenizer=tokenizer,
            chunker_name=chunker_name,
            embedding_model=embedding_model,
            language=language,
            genie=genie,
        )
        self.refinery = build_refinery(context_size=chunk_overlap, tokenizer=tokenizer)

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
                            chunk_id=chunk_id,
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
                            text=text,
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
                    chunk_id=chunk_id,
                    document_id=document_id,
                    version=version,
                    company=company,
                    owner=owner,
                    text=text_value,
                    metadata={
                        "chunker": "chonkie",
                        "strategy": getattr(self.inner, "__class__", type(None)).__name__,
                    },
                )
            )
        return chunks


def build_chonkie_chunker(name: str = "auto", **kwargs: Any) -> Chunker:
    """Pick a chunker by name.

    Args:
        name: Chunker strategy (``"auto"``, ``"recursive"``, ``"token"``,
            ``"sentence"``, ``"semantic"``, ``"late"``, ``"table"``,
            ``"code"``, ``"slumber"``, ``"neural"``,
            ``"word_window"``).
        **kwargs: Forwarded to the underlying constructor.

    Returns:
        A configured :class:`Chunker`.

    Raises:
        ConfigurationError: When ``name`` is unknown or chonkie is
            explicitly requested but unavailable.
    """
    chonkie_names = {
        "auto", "recursive", "token", "sentence",
        "semantic", "late", "table", "code",
        "slumber", "neural",
    }
    if name in chonkie_names:
        if CHONKIE_AVAILABLE:
            return ChonkieChunker(chunker_name=name, **kwargs)
        if name != "auto":
            raise ConfigurationError("chonkie is not installed")
    if name in ("chonkie", "word_window", "auto"):
        if name == "chonkie":
            if CHONKIE_AVAILABLE:
                return ChonkieChunker(**kwargs)
            raise ConfigurationError("chonkie is not installed")
        return WordWindowChunker(**kwargs)
    raise ConfigurationError(f"Unknown chunker: {name!r}")


def __getattr__(name: str) -> Any:
    """Resolve renamed refinery helpers for compatibility."""
    if name == "_build_refinery":
        return build_refinery
    if name == "_apply_refinery":
        return apply_refinery
    raise AttributeError(name)


__all__ = [
    "CHONKIE_AVAILABLE",
    "ChonkieChunker",
    "RAGHubGenie",
    "apply_refinery",
    "build_chonkie_chunker",
    "build_chonkie_inner",
    "build_refinery",
]
