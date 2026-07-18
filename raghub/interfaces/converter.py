"""Document converter contract.

The contract normalisation layer between source documents and the
canonical :class:`raghub.models.KnowledgeBundle`. Concrete
implementations include :class:`raghub.converters.marker.MarkerConverter`
(default) and :class:`raghub.converters.directory.DirectoryConverter`.
"""

from __future__ import annotations

from typing import Protocol

from raghub.models import (
    KnowledgeBundle,
)


class DocumentConverter(Protocol):
    """Converts source bytes to a :class:`KnowledgeBundle`.

    The Markdown produced by the converter is normalised into OKF
    shape (sections, blocks, metadata) and then persisted as the
    canonical knowledge representation.
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
        """Convert ``file_bytes`` into a :class:`KnowledgeBundle`.

        Args:
            source_uri: Stable source identifier (file path, URL).
            file_bytes: Raw content.
            mime_type: Optional MIME hint.
            language: Optional BCP-47 language tag.
            metadata: Optional extra metadata.

        Returns:
            The canonical :class:`KnowledgeBundle`.
        """
