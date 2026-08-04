"""Optional ColBERT late-interaction adapter.

:class:`Colbert` wraps the ``ragatouille`` ColBERT backend and exposes a
:func:`score` interface that can be plugged into the three-channel
hybrid retrieval pipeline.
"""

from __future__ import annotations

from typing import Any

from raghub.errors import GraphUnavailableError


class Colbert:
    """Adapter for the optional :mod:`ragatouille` ColBERT backend."""

    name = "colbert"

    def __init__(self, config: Any | None = None) -> None:
        """Initialise the adapter.

        Args:
            config: Optional :class:`HybridConfig` carrying the
                ``colbert_enabled`` flag. ``None`` defaults to disabled.

        """
        self.config = config
        self.enabled = bool(getattr(config, "colbert_enabled", False))
        self.index: Any | None = None

    def is_available(self) -> bool:
        """Return ``True`` when ColBERT is enabled and importable."""
        if not self.enabled:
            return False
        import importlib.util

        return importlib.util.find_spec("ragatouille") is not None

    def score(self, query: str, doc_texts: list[str]) -> list[float]:
        """Return ColBERT relevance scores parallel to ``doc_texts``.

        Raises:
            GraphUnavailableError: When ``colbert_enabled`` is ``True``
                but the dependency is missing.

        """
        if not doc_texts:
            return []
        if not self.is_available():
            if self.enabled:
                raise GraphUnavailableError(
                    "colbert_enabled is True but ragatouille is not installed; "
                    "pip install 'raghub[colbert]' to enable ColBERT late-interaction"
                )
            return []
        from ragatouille import RAGPretrainedModel

        if self.index is None:
            self.index = RAGPretrainedModel.from_pretrained("colbert-ir/colbertv2.0")
        return list(self.index.rerank(query=query, documents=doc_texts))


__all__ = ["Colbert"]
