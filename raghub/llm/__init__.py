"""LLM provider implementations.

This package ships:

* :class:`BaseLLMProvider` — abstract base class.
* :class:`LiteLLMProvider` — production LLM, backed by LiteLLM (any
  provider: OpenAI, NVIDIA, Anthropic, Bedrock, …).
* :class:`HeuristicLLMProvider` — deterministic offline fallback.

:func:`build_llm_provider` selects the implementation by model name
heuristics: empty / ``heuristic`` / unknown names resolve to
:class:`HeuristicLLMProvider`; everything else resolves to
:class:`LiteLLMProvider`.
"""

from typing import TYPE_CHECKING, Any

from .base import BaseLLMProvider
from .heuristic import HeuristicLLMProvider

if TYPE_CHECKING:
    from .litellm import LiteLLMProvider as LiteLLMProvider


def __getattr__(name: str) -> Any:
    """Lazily expose :class:`LiteLLMProvider`."""
    if name == "LiteLLMProvider":
        from .litellm import LiteLLMProvider as _LiteLLM

        return _LiteLLM
    raise AttributeError(f"module 'raghub.llm' has no attribute {name!r}")


def build_llm_provider(
    model_name: str,
    api_key: str | None = None,
) -> Any:
    """Construct the appropriate LLM provider for ``model_name``.

    Args:
        model_name: The model identifier. Empty / ``"heuristic"`` /
            unknown names resolve to :class:`HeuristicLLMProvider`;
            anything else resolves to :class:`LiteLLMProvider`.
        api_key: Optional API key passed through to
            :class:`LiteLLMProvider`. Ignored by the heuristic.

    Returns:
        A ready-to-use provider instance.
    """
    name = (model_name or "").lower().strip()
    if not name or name == "heuristic-llm" or name == "heuristic":
        return HeuristicLLMProvider(model_name=model_name)
    from .litellm import LiteLLMProvider

    return LiteLLMProvider(model=model_name, api_key=api_key)


__all__ = [
    "BaseLLMProvider",
    "HeuristicLLMProvider",
    "LiteLLMProvider",
    "build_llm_provider",
]
