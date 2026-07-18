"""LiteLLM-backed embedding adapter.

Uses ``litellm.embedding`` to compute vectors across many providers
(OpenAI, NVIDIA, Cohere, Bedrock, …).
"""

from __future__ import annotations

from typing import Any

from raghub.embeddings.base import BaseEmbeddingProvider
from raghub.exceptions import ConfigurationError

litellm: Any

try:
    import litellm

    LITELLM_AVAILABLE = True
    OptionalImportError: Exception | None = None
except Exception as exc:  # pragma: no cover - optional dep
    litellm = None
    LITELLM_AVAILABLE = False
    OptionalImportError = exc


class LiteLLMEmbeddingProvider(BaseEmbeddingProvider):
    """Embedding provider backed by LiteLLM."""

    model_name: str

    def __init__(
        self,
        model: str = "text-embedding-3-small",
        *,
        api_key: str | None = None,
        api_base: str | None = None,
    ) -> None:
        """Initialise the provider.

        Args:
            model: LiteLLM model name (provider-prefixed when needed).
            api_key: Optional API key override.
            api_base: Optional API base override.

        Raises:
            ConfigurationError: When ``litellm`` is not installed.
        """
        if not LITELLM_AVAILABLE:
            raise ConfigurationError("litellm is not installed; run `pip install litellm`.")
        self.model_name = model
        self.api_key = api_key
        self.api_base = api_base

    def embed_text(self, text: str) -> list[float]:
        """Embed a single string.

        Args:
            text: The input text.

        Returns:
            A float vector.
        """
        return self.embed_texts([text])[0]

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Embed multiple strings in one LiteLLM call.

        Args:
            texts: The list of input strings.

        Returns:
            A list of float vectors.
        """
        if not texts:
            return []
        kwargs: dict[str, Any] = {"model": self.model_name, "input": texts}
        if self.api_key:
            kwargs["api_key"] = self.api_key
        if self.api_base:
            kwargs["api_base"] = self.api_base
        response = litellm.embedding(**kwargs)
        data = response.get("data", []) if isinstance(response, dict) else response.data
        return [
            list(item["embedding"]) if isinstance(item, dict) else list(item.embedding)
            for item in data
        ]


__all__ = ["LiteLLMEmbeddingProvider"]
