"""LLM provider implementations.

This package ships:

* :class:`BaseLLMProvider` — abstract base class.
* :class:`LiteLLMProvider` — production LLM, backed by LiteLLM (any
  provider: OpenAI, NVIDIA, Anthropic, Bedrock, …).
* :class:`HeuristicLLMProvider` — deterministic offline fallback.

:func:`build_llm_provider` selects the implementation by model name
heuristics: empty / ``heuristic`` / unknown names resolve to
:class:`HeuristicLLMProvider`; everything else resolves to
:class:`LiteLLMProvider` — **but only if an LLM API key is present
in the environment**. Without a key, the function falls back to the
heuristic provider so the framework always runs offline.
"""

from __future__ import annotations

import os

from . import litellm as litellm_provider
from .base import BaseLLMProvider
from .heuristic import HeuristicLLMProvider

LiteLLMProvider = litellm_provider.LiteLLMProvider


# Environment variable names whose presence indicates the operator
# has credentials for at least one LLM provider. When none of these
# are set, :func:`build_llm_provider` falls back to the deterministic
# heuristic so the framework remains usable offline.
LLM_API_KEY_ENV_VARS: tuple[str, ...] = (
    "OPENAI_API_KEY",
    "ANTHROPIC_API_KEY",
    "NVIDIA_API_KEY",
    "GROQ_API_KEY",
    "LITELLM_API_KEY",
    "COHERE_API_KEY",
    "VOYAGE_API_KEY",
    "AZURE_API_KEY",
    "AWS_ACCESS_KEY_ID",
)


def any_llm_api_key_present() -> bool:
    """Return ``True`` when at least one LLM credential env var is set.

    Returns:
        ``True`` if any of the recognised LLM API-key environment
        variables is set in the process environment; ``False``
        otherwise.
    """
    return any(os.getenv(name) for name in LLM_API_KEY_ENV_VARS)


def __getattr__(name: str) -> object:
    """Return the hoisted provider export for legacy dynamic access."""
    if name == "LiteLLMProvider":
        return LiteLLMProvider
    raise AttributeError(f"module 'raghub.llm' has no attribute {name!r}")


def build_llm_provider(
    model_name: str,
    api_key: str | None = None,
) -> BaseLLMProvider:
    """Construct the appropriate LLM provider for ``model_name``.

    Selection rules (highest priority first):

    1. If ``model_name`` is empty, ``"heuristic"``, or
       ``"heuristic-llm"`` → :class:`HeuristicLLMProvider`.
    2. If no LLM API key is present in the environment *and* no
       ``api_key`` was passed in → :class:`HeuristicLLMProvider`
       (so the framework remains usable offline).
    3. Otherwise → :class:`LiteLLMProvider`.

    Args:
        model_name: The model identifier. Empty / ``"heuristic"`` /
            unknown names resolve to :class:`HeuristicLLMProvider`.
        api_key: Optional API key passed through to
            :class:`LiteLLMProvider`. When provided, the key counts
            as a present credential even if the env vars are unset.

    Returns:
        A ready-to-use provider instance.
    """
    name = (model_name or "").lower().strip()
    if not name or name == "heuristic-llm" or name == "heuristic":
        return HeuristicLLMProvider(model_name=model_name or "heuristic-llm")
    if not api_key and not any_llm_api_key_present():
        return HeuristicLLMProvider(model_name=model_name)
    return litellm_provider.LiteLLMProvider(model=model_name, api_key=api_key)


__all__ = [
    "LLM_API_KEY_ENV_VARS",
    "BaseLLMProvider",
    "HeuristicLLMProvider",
    "LiteLLMProvider",
    "any_llm_api_key_present",
    "build_llm_provider",
]
