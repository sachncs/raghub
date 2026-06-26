"""ChatNVIDIA adapter.

This module wraps `ChatNVIDIA` behind a tiny interface so the rest of the app
does not depend on the vendor SDK directly.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
import logging
import os
from typing import Any


LOGGER = logging.getLogger(__name__)


class LLM(ABC):
    """Chat model interface."""

    @abstractmethod
    def chat(self, messages: list[dict[str, str]]) -> str:
        """Generate a reply from chat messages."""


class NvidiaLLM(LLM):
    """ChatNVIDIA adapter for MiniMax-M3."""

    def __init__(
        self,
        model: str,
        api_key: str | None = None,
        temperature: float = 0.1,
        top_p: float = 0.95,
        max_completion_tokens: int = 8192,
    ) -> None:
        self._model = model
        self._api_key = api_key or os.getenv("NVIDIA_API_KEY")
        self._temperature = temperature
        self._top_p = top_p
        self._max_completion_tokens = max_completion_tokens

    def chat(self, messages: list[dict[str, str]]) -> str:
        """Invoke ChatNVIDIA and return the textual response."""

        try:
            from langchain_nvidia_ai_endpoints import ChatNVIDIA  # type: ignore[import-not-found]
        except ImportError as exc:  # pragma: no cover - dependency missing in dev env
            raise RuntimeError("langchain_nvidia_ai_endpoints is required for ChatNVIDIA") from exc

        if not self._api_key:
            raise RuntimeError("NVIDIA_API_KEY is not configured")

        client = ChatNVIDIA(
            model=self._model,
            api_key=self._api_key,
            temperature=self._temperature,
            top_p=self._top_p,
            max_completion_tokens=self._max_completion_tokens,
        )
        response: Any = client.invoke(messages)
        return str(getattr(response, "content", response))
