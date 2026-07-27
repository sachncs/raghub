"""ColBERT late-interaction adapter (Phase 3.4).

Adapter over the optional :mod:`ragatouille` package — at runtime
the ``ragatouille`` import is guarded so the rest of the package
stays usable when ColBERT is not installed. The adapter exposes
:meth:`is_available` and :meth:`score`; the surrounding retrieval
pipeline consults both before adding ColBERT to the hybrid fusion.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from raghub.exceptions import GraphUnavailableError

if TYPE_CHECKING:
    from raghub.config import HybridConfig


class ColbertLateInteraction:
    """Adapter for the :mod:`ragatouille` ColBERT backend.

    Attributes:
        name: Always ``"colbert"``.
    """

    name = "colbert"

    def __init__(self, config: HybridConfig | None = None) -> None:
        """Initialise the adapter.

        Args:
            config: Optional :class:`HybridConfig` carrying the
                ``colbert_enabled`` flag. ``None`` defaults to
                ``colbert_enabled=False``.
        """
        self.config = config
        self.enabled = bool(getattr(config, "colbert_enabled", False))
        self.index: Any | None = None

    def is_available(self) -> bool:
        """Return ``True`` when ColBERT is enabled and importable.

        Returns:
            ``True`` only when both the ``colbert_enabled`` setting is
            on and the :mod:`ragatouille` package can be imported.
        """
        if not self.enabled:
            return False
        import importlib.util

        return importlib.util.find_spec("ragatouille") is not None

    def score(self, query: str, doc_texts: list[str]) -> list[float]:
        """Return ColBERT relevance scores for each document.

        Args:
            query: The user's question.
            doc_texts: One entry per candidate document.

        Returns:
            A list of float scores (higher = more relevant), parallel
            to ``doc_texts``. When ColBERT is unavailable an empty
            list is returned and the surrounding hybrid path skips
            the channel.

        Raises:
            GraphUnavailableError: When ``colbert_enabled`` is ``True``
                but the dependency is missing — the operator
                configured a feature that cannot run.
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


