"""Cohere cross-encoder reranker.

Uses the Cohere SDK; ``cohere`` is a required runtime dependency.
The client is constructed on first use.
"""

from __future__ import annotations

import os
import time
from collections.abc import Sequence
from typing import cast

import cohere
from pydantic import SecretStr

from raghub.exceptions import RerankerError
from raghub.models import RetrievalHit
from raghub.observability import record_rerank_latency


def record_latency(provider: str, seconds: float) -> None:
    """Push a histogram observation when Prometheus is wired up.

    Args:
        provider: The reranker provider label (``"cohere"`` here).
        seconds: Observed wall-clock latency.
    """
    record_rerank_latency(provider, seconds)


class CohereReranker:
    """Cross-encoder reranker backed by the Cohere API.

    Attributes:
        name: ``"cohere"``.
    """

    name = "cohere"

    def __init__(
        self,
        api_key: str | SecretStr | None = None,
        *,
        model: str = "rerank-english-v3.0",
        top_k: int = 20,
        client: object | None = None,
    ) -> None:
        """Initialise the reranker.

        Args:
            api_key: Cohere API key. Defaults to ``COHERE_API_KEY``
                env var when ``None``. Wrapped in :class:`SecretStr`
                to avoid leaking into logs.
            model: Cohere rerank model name.
            top_k: Maximum candidates the reranker scores.
            client: Optional pre-built ``cohere.Client`` instance. When
                supplied, ``api_key`` is ignored — useful for tests.

        Raises:
            RerankerError: When no API key is available.
        """
        resolved = api_key
        if resolved is None:
            env = os.getenv("COHERE_API_KEY")
            if not env:
                raise RerankerError(
                    "CohereReranker requires COHERE_API_KEY or an explicit api_key"
                )
            resolved = env
        self.api_key = resolved if isinstance(resolved, SecretStr) else SecretStr(resolved)
        self.model = model
        self.top_k = top_k
        self.client = client

    def ensure_client(self) -> cohere.Client:
        """Return the underlying Cohere client, building it on first use.

        Returns:
            The :class:`cohere.Client` instance.
        """
        if self.client is None:
            self.client = cohere.Client(api_key=self.api_key.get_secret_value())
        return cast(cohere.Client, self.client)

    def rerank(
        self,
        *,
        question: str,
        hits: Sequence[RetrievalHit],
    ) -> list[RetrievalHit]:
        """Return the hits reordered by Cohere's relevance score.

        Args:
            question: User query.
            hits: Candidate hits from the retriever.

        Returns:
            The same hits reordered by descending Cohere score. Hits
            the API rejects (rare) are dropped. ``hits`` unchanged when
            empty.
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
        """Call the Cohere API and reorder ``hits`` by its output.

        Args:
            question: The user query.
            hits: The candidates to rerank.

        Returns:
            The hits reordered to match the API response. Hits the
            API doesn't include (rare) are dropped.
        """
        client = self.ensure_client()
        documents = [hit.chunk.text for hit in hits]
        response = client.rerank(
            model=self.model,
            query=question,
            documents=documents,
            top_n=min(self.top_k, len(documents)),
        )
        ordered: list[RetrievalHit] = []
        for result in getattr(response, "results", []):
            idx = getattr(result, "index", None)
            if idx is None or idx < 0 or idx >= len(hits):
                continue
            ordered.append(hits[idx])
        return ordered


