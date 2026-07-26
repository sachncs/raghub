"""Shared fakes for query-transform tests (Phase 2)."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any


class FakeLLM:
    """Minimal async LLM stand-in.

    Args:
        responses: Cycled through one-per-call to ``async_generate``.
            When exhausted, the last response is reused.
    """

    def __init__(self, responses: list[str]) -> None:
        self.responses = list(responses)
        self.calls: list[dict[str, Any]] = []
        self._cursor = 0

    @property
    def model_name(self) -> str:
        return "fake-llm"

    async def async_generate(
        self,
        *,
        system_prompt: str,
        conversation: Sequence = (),
        context: Sequence[str] = (),
        question: str,
        image_paths: list[str] | None = None,
        session_history: list[dict] | None = None,
    ) -> str:
        self.calls.append({"question": question, "system": system_prompt})
        if self._cursor < len(self.responses):
            value = self.responses[self._cursor]
            self._cursor += 1
            return value
        return self.responses[-1] if self.responses else ""


class RaisingLLM:
    """LLM that always raises; useful for error-path tests."""

    @property
    def model_name(self) -> str:
        return "raising-llm"

    async def async_generate(self, **_: Any) -> str:
        raise RuntimeError("LLM down")