"""Lightweight, dependency-free fallback converter.

Used when Marker isn't installed or the input is plain text. Accepts
anything that decodes as UTF-8 and produces a single-section bundle.
"""

from __future__ import annotations

from raghub.converters.markdown import normalise_markdown
from raghub.interfaces.converter import DocumentConverter
from raghub.models import KnowledgeBundle


class PlainTextConverter(DocumentConverter):
    """Convert plain text into a :class:`KnowledgeBundle`.

    The text is wrapped in a Markdown paragraph and normalised via
    :func:`raghub.converters.markdown.normalise_markdown`. There is no
    structure to preserve; callers wanting headings should pre-Markdown
    the input themselves.
    """

    def convert(
        self,
        *,
        source_uri: str,
        file_bytes: bytes,
        mime_type: str = "",
        language: str = "",
        metadata: dict | None = None,
    ) -> KnowledgeBundle:
        """Convert plain text to a single-section bundle.

        Args:
            source_uri: Stable source identifier.
            file_bytes: Raw bytes.
            mime_type: MIME hint.
            language: BCP-47 language tag.
            metadata: Extra metadata.

        Returns:
            The canonical bundle.
        """
        text = file_bytes.decode("utf-8", errors="replace")
        return normalise_markdown(
            text,
            source_uri=source_uri,
            mime_type=mime_type or "text/plain",
            language=language,
            metadata=metadata or {},
        )


__all__ = ["PlainTextConverter"]
