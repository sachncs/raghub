"""LLM provider implementations."""

from .base import BaseLLMProvider
from .heuristic import HeuristicLLMProvider
from .nvidia import NvidiaLLMProvider


def build_llm_provider(
    model_name: str,
    api_key: str | None = None,
) -> NvidiaLLMProvider | HeuristicLLMProvider:
    if "nvidia" in model_name.lower() or api_key is not None:
        return NvidiaLLMProvider(model=model_name, api_key=api_key)
    return HeuristicLLMProvider(model_name=model_name)


__all__ = [
    "BaseLLMProvider",
    "HeuristicLLMProvider",
    "NvidiaLLMProvider",
    "build_llm_provider",
]
