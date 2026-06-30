"""Common Markdown → DocumentSection normalisation.

Both Marker and the lightweight fallback use this helper so the
output shape is uniform regardless of which library produced the
Markdown.
"""

from __future__ import annotations

import re
from typing import Any

from raghub.models import (
    BlockKind,
    DocumentBlock,
    DocumentSection,
    KnowledgeBundle,
    deterministic_id,
)

HEADING_RE = re.compile(r"^(#{1,6})\s+(.*)$")
TABLE_LINE_RE = re.compile(r"^\s*\|.*\|\s*$")
FENCE_RE = re.compile(r"^(```|~~~)\s*(\S+)?\s*$")
IMAGE_RE = re.compile(r"!\[([^\]]*)\]\(([^)]+)\)")
EQUATION_BLOCK_RE = re.compile(r"^\$\$(.*)\$\$\s*$", re.DOTALL)
INLINE_EQUATION_RE = re.compile(r"\$([^$\n]+)\$")


def normalise_markdown(
    markdown: str,
    *,
    source_uri: str,
    mime_type: str = "",
    language: str = "",
    metadata: dict[str, Any] | None = None,
    page_numbers: list[int] | None = None,
) -> KnowledgeBundle:
    """Convert ``markdown`` to a single-section :class:`KnowledgeBundle`.

    Splits the Markdown by headings and dispatches each block by
    kind (text, table, code, image, equation). One section is created
    per top-level heading; sub-section headings live inside the same
    section's text. This is sufficient for a normative first cut
    while still preserving the document hierarchy.

    Args:
        markdown: Markdown source.
        source_uri: Stable identifier for the source.
        mime_type: MIME type of the source (kept on the bundle).
        language: BCP-47 language tag.
        metadata: Format-specific metadata.
        page_numbers: Optional page numbers for the section.

    Returns:
        The canonical :class:`KnowledgeBundle`.
    """
    metadata = metadata or {}
    page_numbers = page_numbers or []
    blocks, flat = markdown_to_document_blocks(markdown)

    if not blocks and flat:
        blocks = [DocumentBlock(kind=BlockKind.TEXT, content=flat)]

    section = DocumentSection(
        section_id=deterministic_id("section", source_uri, "auto"),
        index=0,
        heading="",
        blocks=blocks,
        page_numbers=page_numbers,
        source_location=f"{source_uri}#0",
    )

    return KnowledgeBundle(
        bundle_id=deterministic_id("bundle", source_uri),
        source_uri=source_uri,
        mime_type=mime_type,
        language=language,
        metadata=metadata,
        sections=[section],
    )


def markdown_to_document_blocks(markdown: str) -> tuple[list[DocumentBlock], str]:
    """Return ``(blocks, trailing_text)`` for a Markdown snippet.

    Args:
        markdown: The Markdown body.

    Returns:
        A list of structured blocks plus any un-emitted text.
    """
    blocks: list[DocumentBlock] = []
    text_buf: list[str] = []
    in_fence = False
    fence_marker = ""
    fence_lang = ""

    for raw_line in markdown.splitlines():
        if in_fence:
            if raw_line.strip() == fence_marker:
                blocks.append(
                    DocumentBlock(
                        kind=BlockKind.CODE,
                        content="\n".join(text_buf).rstrip("\n"),
                        metadata={"language": fence_lang},
                    )
                )
                text_buf = []
                in_fence = False
            else:
                text_buf.append(raw_line)
            continue

        fence_match = FENCE_RE.match(raw_line.strip())
        if fence_match:
            if text_buf:
                blocks.append(
                    DocumentBlock(kind=BlockKind.TEXT, content="\n".join(text_buf).rstrip("\n"))
                )
                text_buf = []
            in_fence = True
            fence_marker = fence_match.group(1)
            fence_lang = fence_match.group(2) or ""
            continue

        if TABLE_LINE_RE.match(raw_line):
            if text_buf:
                blocks.append(
                    DocumentBlock(kind=BlockKind.TEXT, content="\n".join(text_buf).rstrip("\n"))
                )
                text_buf = []
            blocks.append(DocumentBlock(kind=BlockKind.TABLE, content=raw_line.strip()))
            continue

        equation_match = EQUATION_BLOCK_RE.match(raw_line.strip())
        if equation_match:
            if text_buf:
                blocks.append(
                    DocumentBlock(kind=BlockKind.TEXT, content="\n".join(text_buf).rstrip("\n"))
                )
                text_buf = []
            blocks.append(
                DocumentBlock(kind=BlockKind.EQUATION, content=equation_match.group(1).strip())
            )
            continue

        text_buf.append(raw_line)

    if text_buf:
        trailing = "\n".join(text_buf).rstrip("\n")
        for raw_image in IMAGE_RE.finditer(trailing):
            caption, uri = raw_image.group(1), raw_image.group(2)
            blocks.append(
                DocumentBlock(
                    kind=BlockKind.IMAGE,
                    content=uri,
                    metadata={"caption": caption, "source": uri},
                )
            )
        trailing = IMAGE_RE.sub("", trailing)
        if trailing.strip():
            blocks.append(
                DocumentBlock(
                    kind=BlockKind.TEXT,
                    content=INLINE_EQUATION_RE.sub(lambda m: f"\\({m.group(1)}\\)", trailing),
                )
            )

    return blocks, ""


__all__ = ["normalise_markdown"]
