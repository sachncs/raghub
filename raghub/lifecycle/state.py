"""Lifecycle state machine and document-block state machine.

This module owns the in-memory state transitions of a :class:`Document`
plus the Markdown → :class:`DocumentBlock` state machine that produces
:class:`raghub.models.DocumentSection` content. It deliberately knows
nothing about file scanning, MIME detection, or third-party converters.

Public surface:

- :class:`Lifecycle` — validate and apply status transitions.
- :func:`new_version` — mint the next :class:`Document` revision.
- :func:`datetime_now_utc` — injectable UTC clock.
- :class:`ChunkingPlan` — word-window chunking configuration.
- :class:`Section` — Markdown → :class:`DocumentBlock` state machine.
- :func:`md_to_blocks` — convenience wrapper around :class:`Section`.
- :func:`normalise_markdown` — Markdown → :class:`Bundle`.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from raghub.constants import DEFAULT_CHUNK_SIZE_WORDS
from raghub.core import DocumentStateMachine
from raghub.models import (
    BlockKind,
    Bundle,
    Document,
    DocumentBlock,
    DocumentLifecycleStatus,
    DocumentSection,
    deterministic_id,
)
from raghub.types import JSONValue

# ---------------------------------------------------------------------------
# Lifecycle
# ---------------------------------------------------------------------------


@dataclass
class Lifecycle:
    """Validate and apply document-status transitions.

    Attributes:
        machine: The :class:`raghub.core.DocumentStateMachine` used to
            validate transitions. Defaults to a fresh instance.

    """

    machine: Any = field(default_factory=lambda: None)

    def __post_init__(self) -> None:
        """Initialise the default state machine when none was supplied."""
        if self.machine is None:
            self.machine = DocumentStateMachine()

    def transition(self, document: Document, status: DocumentLifecycleStatus) -> Document:
        """Return a new :class:`Document` whose ``status`` is ``status``.

        The input document is not mutated. Callers that held a
        reference should keep the returned instance instead.

        Args:
            document: The :class:`Document` to transition.
            status: The target lifecycle status.

        Returns:
            A new :class:`Document` with ``status`` updated.

        Raises:
            ValueError: If the transition is not in the state machine's
                allow table. Idempotent transitions to the same status
                are accepted as no-ops.

        """
        if not self.machine.can_transition(document.status, status) and status != document.status:
            raise ValueError(f"Illegal transition from {document.status} to {status}")
        return document.copy(status=status)


# ---------------------------------------------------------------------------
# Versioning
# ---------------------------------------------------------------------------


def datetime_now_utc() -> Any:
    """Return the current UTC datetime.

    Thin wrapper around :func:`datetime.datetime.now` so callers can
    inject a fake clock in tests without monkey-patching the global
    ``datetime`` module.
    """
    from datetime import UTC, datetime

    return datetime.now(UTC)


def new_version(previous: Document | None, **overrides: Any) -> Document:
    """Build a new :class:`Document` from a previous record.

    Args:
        previous: The prior version, or ``None`` for a brand-new document.
        **overrides: Field overrides applied after the clone.

    Returns:
        A fully-typed :class:`Document` ready for persistence.

    """
    version_number = 1 if previous is None else previous.version + 1
    baseline = Document() if previous is None else previous
    return baseline.copy(
        version=version_number,
        status=DocumentLifecycleStatus.New,
        updated_at=datetime_now_utc(),
        **overrides,
    )


# ---------------------------------------------------------------------------
# Chunking configuration
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ChunkingPlan:
    """Configuration for the word-window chunker.

    Attributes:
        chunk_size_words: Target number of words per chunk.
        overlap_words: Number of words carried over from one chunk to
            the next.

    """

    chunk_size_words: int = DEFAULT_CHUNK_SIZE_WORDS
    overlap_words: int = 100


# ---------------------------------------------------------------------------
# Markdown → DocumentBlock state machine
# ---------------------------------------------------------------------------


HEADING_RE = re.compile(r"^(#{1,6})\s+(.*)$")
TABLE_LINE_RE = re.compile(r"^\s*\|.*\|\s*$")
FENCE_RE = re.compile(r"^(```|~~~)\s*(\S+)?\s*$")
IMAGE_RE = re.compile(r"!\[([^\]]*)\]\(([^)]+)\)")
EQUATION_RE = re.compile(r"^\$\$(.*)\$\$\s*$", re.DOTALL)
INLINE_EQUATION_RE = re.compile(r"\$([^$\n]+)\$")


class Section:
    """State machine that turns Markdown text into a list of :class:`DocumentBlock`.

    The class encapsulates the per-line scanner so the public
    :meth:`parse` entry point stays a tight state machine loop. Each
    helper is a regular public method; the chain of calls inside
    :meth:`parse` is the only intended caller.
    """

    def __init__(self) -> None:
        """Initialise the scanner with an empty block list and buffer."""
        self.blocks: list[DocumentBlock] = []
        self.text_buf: list[str] = []
        self.in_fence: bool = False
        self.fence_marker: str = ""
        self.fence_lang: str = ""

    def parse(self, markdown: str) -> tuple[list[DocumentBlock], str]:
        """Return ``(blocks, trailing_text)`` for a Markdown snippet.

        Args:
            markdown: The Markdown body.

        Returns:
            A list of structured blocks plus any un-emitted text.

        """
        self.blocks = []
        self.text_buf = []
        self.in_fence = False
        self.fence_marker = ""
        self.fence_lang = ""

        for raw_line in markdown.splitlines():
            if self.in_fence:
                self.in_fence = self.handle_fenced_code(raw_line)
                continue

            fence_match = FENCE_RE.match(raw_line.strip())
            if fence_match:
                self.fence_marker, self.fence_lang = self.open_fence(fence_match)
                self.in_fence = True
                continue

            if TABLE_LINE_RE.match(raw_line):
                self.flush_text_buffer()
                self.blocks.append(DocumentBlock(kind=BlockKind.Table, content=raw_line.strip()))
                continue

            equation_match = EQUATION_RE.match(raw_line.strip())
            if equation_match:
                self.flush_text_buffer()
                self.blocks.append(
                    DocumentBlock(kind=BlockKind.Equation, content=equation_match.group(1).strip())
                )
                continue

            self.text_buf.append(raw_line)

        if self.text_buf:
            trailing = "\n".join(self.text_buf).rstrip("\n")
            self.flush_images_and_text(trailing)

        return self.blocks, ""

    def flush_text_buffer(self) -> None:
        """Append a TEXT block from ``text_buf`` and reset the buffer."""
        if not self.text_buf:
            return
        self.blocks.append(
            DocumentBlock(kind=BlockKind.Text, content="\n".join(self.text_buf).rstrip("\n"))
        )
        self.text_buf.clear()

    def handle_fenced_code(self, raw_line: str) -> bool:
        """Return updated ``in_fence`` after handling a line inside (or closing) a fence."""
        if raw_line.strip() == self.fence_marker:
            self.blocks.append(
                DocumentBlock(
                    kind=BlockKind.Code,
                    content="\n".join(self.text_buf).rstrip("\n"),
                    metadata={"language": self.fence_lang},
                )
            )
            self.text_buf.clear()
            return False
        self.text_buf.append(raw_line)
        return self.in_fence

    def open_fence(self, fence_match: re.Match[str]) -> tuple[str, str]:
        """Open a new code fence; return ``(fence_marker, fence_lang)``."""
        self.flush_text_buffer()
        return fence_match.group(1), fence_match.group(2) or ""

    def flush_images_and_text(self, trailing: str) -> None:
        """Extract image / inline-equation blocks from trailing text and append a TEXT block."""
        for raw_image in IMAGE_RE.finditer(trailing):
            caption, uri = raw_image.group(1), raw_image.group(2)
            self.blocks.append(
                DocumentBlock(
                    kind=BlockKind.Image,
                    content=uri,
                    metadata={"caption": caption, "source": uri},
                )
            )
        trailing = IMAGE_RE.sub("", trailing)
        if trailing.strip():
            self.blocks.append(
                DocumentBlock(
                    kind=BlockKind.Text,
                    content=INLINE_EQUATION_RE.sub(lambda m: f"\\({m.group(1)}\\)", trailing),
                )
            )


def md_to_blocks(markdown: str) -> tuple[list[DocumentBlock], str]:
    """Return ``(blocks, trailing_text)`` for a Markdown snippet.

    Thin convenience wrapper around :class:`Section`; see the
    class docstring for the block-construction semantics.
    """
    return Section().parse(markdown)


def normalise_markdown(
    markdown: str,
    *,
    source_uri: str,
    **options: JSONValue,
) -> Bundle:
    """Convert ``markdown`` to a single-section :class:`Bundle`.

    Args:
        markdown: Markdown source.
        source_uri: Stable identifier for the source.
        **options: Optional ``mime_type=``, ``language=``,
            ``metadata=``, ``page_numbers=`` overrides.

    Returns:
        The canonical :class:`Bundle`.

    """
    mime_type: str = options.get("mime_type", "")
    language: str = options.get("language", "")
    metadata: dict[str, Any] | None = options.get("metadata")
    page_numbers: list[int] | None = options.get("page_numbers")
    metadata = metadata or {}
    page_numbers = page_numbers or []
    blocks, flat = md_to_blocks(markdown)

    if not blocks and flat:
        blocks = [DocumentBlock(kind=BlockKind.Text, content=flat)]

    section = DocumentSection(
        section_id=deterministic_id("section", source_uri, "auto"),
        index=0,
        heading="",
        blocks=blocks,
        page_numbers=page_numbers,
        source_location=f"{source_uri}#0",
    )

    return Bundle(
        bundle_id=deterministic_id("bundle", source_uri),
        source_uri=source_uri,
        mime_type=mime_type,
        language=language,
        metadata=metadata,
        sections=[section],
    )


__all__ = [
    "ChunkingPlan",
    "Lifecycle",
    "Section",
    "datetime_now_utc",
    "md_to_blocks",
    "new_version",
    "normalise_markdown",
]
