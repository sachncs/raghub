"""NV-Embed-QA embedding provider via langchain-nvidia-ai-endpoints."""

from __future__ import annotations

from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings


class NvidiaEmbeddingProvider:
    """NV-Embed-QA embedding provider via langchain-nvidia-ai-endpoints."""

    def __init__(
        self,
        model: str = "nvidia/nv-embed-qa",
        dimension: int = 768,
        api_key: str | None = None,
    ) -> None:
        self.model_name = model
        kwargs = {"model": model, "dims": dimension}
        if api_key is not None:
            kwargs["api_key"] = api_key
        self._client = NVIDIAEmbeddings(**kwargs)

    def embed(self, texts: list[str]) -> list[list[float]]:
        return self._client.embed_documents(texts)

    def embed_query(self, text: str) -> list[float]:
        return self._client.embed_query(text)

    @property
    def dimension(self) -> int:
        return self._client.dims
