"""LiteLLM-backed embedding adapter.

Uses ``litellm.embedding`` to compute vectors across many providers
(OpenAI, NVIDIA, Cohere, Bedrock, …).
"""

from __future__ import annotations

from raghub.embeddings.base import BaseEmbeddingProvider
from raghub.exceptions import ConfigurationError

try:
    import litellm  # type: ignore
    _LITELLM_AVAILABLE = True
    _ImportError: Exception | None = None
except Exception as exc:  # pragma: no cover - optional dep
    litellm = None
    _LITELLM_AVAILABLE = False
    _ImportError = exc


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
        if not _LITELLM_AVAILABLE:
            raise ConfigurationError(
                "litellm is not installed; run `pip install litellm`."
            )
        self.model_name = model
        self._api_key = api_key
        self._api_base = api_base

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
        kwargs = {"model": self.model_name, "input": texts}
        if self._api_key:
            kwargs["api_key"] = self._api_key
        if self._api_base:
            kwargs["api_base"] = self._api_base
        response = litellm.embedding(**kwargs)
        data = response.get("data", []) if isinstance(response, dict) else response.data
        return [list(item["embedding"]) if isinstance(item, dict) else list(item.embedding) for item in data]


__all__ = ["LiteLLMEmbeddingProvider"]
