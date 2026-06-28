"""LLM provider implementations.

This package ships:

* :class:`BaseLLMProvider` — abstract base class.
* :class:`NvidiaLLMProvider` — production LLM via NVIDIA's hosted
  endpoints, backed by ``langchain-nvidia-ai-endpoints``.
* :class:`HeuristicLLMProvider` — deterministic offline fallback.

:func:`build_llm_provider` chooses the right implementation based on
the model's name; any name containing ``"nvidia"`` resolves to
:class:`NvidiaLLMProvider`, everything else to the heuristic.
"""

from .base import BaseLLMProvider
from .heuristic import HeuristicLLMProvider
from .nvidia import NvidiaLLMProvider


def build_llm_provider(
    model_name: str,
    api_key: str | None = None,
) -> NvidiaLLMProvider | HeuristicLLMProvider:
    """Construct the appropriate LLM provider for ``model_name``.

    Args:
        model_name: The model identifier. A case-insensitive substring
            check for ``"nvidia"`` routes to
            :class:`NvidiaLLMProvider`; everything else falls through
            to :class:`HeuristicLLMProvider`.
        api_key: Optional API key passed through to
            :class:`NvidiaLLMProvider`. Ignored for the heuristic.

    Returns:
        A ready-to-use provider instance.
    """
    if "nvidia" in model_name.lower():
        return NvidiaLLMProvider(model=model_name, api_key=api_key)
    return HeuristicLLMProvider(model_name=model_name)


__all__ = [
    "BaseLLMProvider",
    "HeuristicLLMProvider",
    "NvidiaLLMProvider",
    "build_llm_provider",
]