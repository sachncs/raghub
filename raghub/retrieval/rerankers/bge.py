"""BGE / sentence-transformers cross-encoder reranker.

Local model; no API key required but downloads ~2 GB on first use.
Optional dependency (``sentence-transformers``); imported lazily.
"""

from __future__ import annotations

import time
from collections.abc import Sequence

from raghub.exceptions import RerankerError
from raghub.models import RetrievalHit


def record_latency(provider: str, seconds: float) -> None:
    """Push a histogram observation when Prometheus is wired up.

    Args:
        provider: The reranker provider label (``"bge"`` here).
        seconds: Observed wall-clock latency.
    """
    try:
        from raghub.observability.metrics import record_rerank_latency

        record_rerank_latency(provider, seconds)
    except Exception:
        pass


class BgeReranker:
    """Local cross-encoder reranker (default: ``BAAI/bge-reranker-v2-m3``).

    Attributes:
        name: ``"bge"``.
    """

    name = "bge"

    def __init__(
        self,
        *,
        model: str = "BAAI/bge-reranker-v2-m3",
        top_k: int = 20,
        encoder: object | None = None,
    ) -> None:
        """Initialise the reranker.

        Args:
            model: HuggingFace model id or local path.
            top_k: Maximum candidates the reranker is asked to score.
            encoder: Optional pre-built ``CrossEncoder`` instance.
                When supplied the model is not loaded — useful for
                tests and for sharing an encoder across rerankers.
        """
        self.model = model
        self.top_k = top_k
        self.encoder = encoder

    def ensure_encoder(self) -> object:
        """Return the underlying ``CrossEncoder``, loading it lazily.

        Returns:
            The :class:`sentence_transformers.CrossEncoder` instance.

        Raises:
            RerankerError: When the optional ``sentence-transformers``
                package is not installed.
        """
        if self.encoder is not None:
            return self.encoder
        try:
            from sentence_transformers import CrossEncoder  # type: ignore[import-not-found]
        except ImportError as exc:
            raise RerankerError(
                "sentence-transformers not installed; pip install 'raghub[rerank]'"
            ) from exc
        self.encoder = CrossEncoder(self.model)
        return self.encoder

    def rerank(
        self,
        *,
        question: str,
        hits: Sequence[RetrievalHit],
    ) -> list[RetrievalHit]:
        """Return the hits reordered by the cross-encoder's relevance score.

        Args:
            question: User query.
            hits: Candidate hits from the retriever.

        Returns:
            The same hits reordered by descending cross-encoder score.
        """
        if not hits:
            return []
        started = time.perf_counter()
        ordered = self.do_rerank(question, hits)
        record_latency(self.name, time.perf_counter() - started)
        return ordered

    def do_rerank(
        self, question: str, hits: Sequence[RetrievalHit]
    ) -> list[RetrievalHit]:
        """Score every ``(question, chunk)`` pair and sort by the score.

        Args:
            question: The user query.
            hits: The candidates to rerank.

        Returns:
            The hits reordered to match the cross-encoder's scores.
        """
        encoder = self.ensure_encoder()
        pairs = [(question, hit.chunk.text) for hit in hits]
        try:
            scores = list(encoder.predict(pairs))
        except Exception as exc:
            raise RerankerError(f"BGE rerank failed: {exc}") from exc
        ordered = sorted(
            zip(scores, hits, strict=True),
            key=lambda pair: pair[0],
            reverse=True,
        )
        return [hit for _, hit in ordered[: self.top_k]]


__all__ = ["BgeReranker", "record_latency"]