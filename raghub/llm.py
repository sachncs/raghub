"""LLM provider implementations.

This module ships:

* :class:`BaseLLMProvider` — abstract base class.
* :class:`LiteLLMProvider` — production LLM, backed by LiteLLM (any
  provider: OpenAI, NVIDIA, Anthropic, Bedrock, …).
* :class:`HeuristicLLMProvider` — deterministic offline fallback.
* :func:`build_llm_provider` — selects an implementation by model
  name and credential availability.

:func:`build_llm_provider` resolves to :class:`HeuristicLLMProvider`
when the model name is empty / ``"heuristic"`` *or* when no LLM API
key is present in the environment, so the framework always runs
offline.
"""

from __future__ import annotations

import asyncio
import base64
import mimetypes
import os
from abc import ABC, abstractmethod
from collections.abc import AsyncIterator, Sequence
from types import TracebackType
from typing import Any, Literal, Self

import litellm

from raghub.exceptions import ConfigurationError, LLMError
from raghub.models import ConversationTurn

# Module-level flag retained so existing tests that patch
# ``raghub.llm.LITELLM_AVAILABLE = False`` can simulate a missing
# optional dependency even though the package is now required.
LITELLM_AVAILABLE = True


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


class BaseLLMProvider(ABC):
    """Abstract LLM provider.

    All concrete providers (NVIDIA, heuristic, …) implement
    :meth:`generate`. The interface is intentionally narrow: the
    caller assembles the prompt and passes the components in. The
    provider's job is to call its backing SDK and return a string.
    """

    model_name: str

    @abstractmethod
    def generate(
        self,
        *,
        system_prompt: str,
        conversation: Sequence[ConversationTurn],
        context: Sequence[str],
        question: str,
        image_paths: list[str] | None = None,
        session_history: list[dict[str, Any]] | None = None,
    ) -> str:
        """Generate an answer from a fully-constructed prompt.

        Args:
            system_prompt: The system-level instructions, including
                tenant-specific formatting guidance.
            conversation: Recent in-window turns from the conversation
                manager.
            context: Retrieved source chunks (already RBAC-filtered).
            question: The user's most recent question.
            image_paths: Optional list of on-disk image paths to attach
                to the final user message (vision-capable providers only).
            session_history: Optional prior turns from the persistent
                session store. Format mirrors
                :class:`raghub.models.ConversationTurn` dicts.

        Returns:
            The provider-generated answer as a plain string.
        """

    async def async_generate(
        self,
        *,
        system_prompt: str,
        conversation: Sequence[ConversationTurn] = (),
        context: Sequence[str] = (),
        question: str,
        image_paths: list[str] | None = None,
        session_history: list[dict[str, Any]] | None = None,
    ) -> str:
        """Generate without blocking the event loop."""
        return await asyncio.to_thread(
            self.generate,
            system_prompt=system_prompt,
            conversation=conversation,
            context=context,
            question=question,
            image_paths=image_paths,
            session_history=session_history,
        )


class HeuristicLLMProvider(BaseLLMProvider):
    """Composes an answer from retrieved context without any model call."""

    def __init__(self, model_name: str = "heuristic-llm") -> None:
        """Initialise the heuristic provider.

        Args:
            model_name: Stable identifier surfaced as
                :pyattr:`model_name`. Defaults to ``"heuristic-llm"``.
        """
        self.model_name = model_name

    def generate(
        self,
        *,
        system_prompt: str,
        conversation: Sequence[ConversationTurn],
        context: Sequence[str],
        question: str,
        image_paths: list[str] | None = None,
        session_history: list[dict[str, Any]] | None = None,
    ) -> str:
        """Return a fixed prefix built from the top context fragments.

        Args:
            system_prompt: Ignored; kept for interface symmetry.
            conversation: Ignored; kept for interface symmetry.
            context: The retrieved chunks to summarise. At most the
                first three non-empty fragments are consulted.
            question: Ignored; kept for interface symmetry.
            image_paths: Ignored; the heuristic does not handle images.
            session_history: Ignored; the heuristic does not use
                history.

        Returns:
            A ``"<fragment1> <fragment2> <fragment3>"``-style prefix,
            truncated to 1000 characters. The literal string
            ``"No accessible source chunks were found for this question."``
            is returned when ``context`` is empty.
        """
        if not context:
            return "No accessible source chunks were found for this question."
        prefix = " ".join(fragment.strip() for fragment in context[:3] if fragment.strip())
        return prefix[:1000]


class LLMValueErrorBoundary:
    """Preserve legacy ``ValueError`` translation without catching provider errors."""

    def __init__(self, message: str) -> None:
        """Store the domain-error message prefix."""
        self.message = message

    def __enter__(self) -> Self:
        """Enter the error boundary."""
        return self

    def __exit__(
        self,
        exception_type: type[BaseException] | None,
        exception: BaseException | None,
        traceback: TracebackType | None,
    ) -> Literal[False]:
        """Translate direct ``ValueError`` failures and propagate all others."""
        if exception_type is ValueError and exception is not None:
            raise LLMError(f"{self.message}: {exception}") from exception
        return False


class LiteLLMProvider(BaseLLMProvider):
    """LLM provider backed by LiteLLM.

    The provider is API-compatible with every LLM endpoint that
    LiteLLM supports: OpenAI, NVIDIA, Anthropic, Bedrock, etc.
    """

    model_name: str

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        *,
        api_key: str | None = None,
        api_base: str | None = None,
        temperature: float = 0.2,
        timeout_seconds: float | None = None,
    ) -> None:
        """Initialise the provider.

        Args:
            model: LiteLLM model name.
            api_key: Optional API key override.
            api_base: Optional API base override.
            temperature: Sampling temperature for ``completion``.
            timeout_seconds: Optional LiteLLM request timeout.

        Raises:
            ConfigurationError: When ``litellm`` is not installed.
        """
        if not LITELLM_AVAILABLE:
            raise ConfigurationError("litellm is not installed; run `pip install litellm`.")
        if timeout_seconds is not None and timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be greater than zero")
        self.model_name = model
        self.api_key = api_key
        self.api_base = api_base
        self.temperature = temperature
        self.timeout_seconds = timeout_seconds
        self.last_usage: dict[str, Any] | None = None

    def require_litellm(self) -> None:
        """Raise a clear error if LiteLLM is not installed."""
        if not LITELLM_AVAILABLE:
            raise ConfigurationError("litellm is not installed; run `pip install litellm`.")

    def build_messages(
        self,
        *,
        system_prompt: str,
        conversation: Sequence[ConversationTurn] = (),
        context: Sequence[str] = (),
        question: str,
        image_paths: list[str] | None = None,
        session_history: list[dict[str, Any]] | None = None,
    ) -> list[dict[str, Any]]:
        """Assemble an OpenAI-style message list.

        Args:
            system_prompt: System instructions; becomes the first
                ``system`` message.
            conversation: Recent in-window question/answer turns.
            context: Retrieved chunks; joined into a single system
                message labelled ``"Context:"``.
            question: The latest user question. When ``image_paths``
                is empty the question is a plain string; otherwise it
                is a content array with one ``image_url`` entry per
                file.
            image_paths: Optional list of on-disk image paths.
            session_history: Optional prior turns; ``role`` maps to
                ``user`` / ``assistant`` / ``system``.

        Returns:
            A list of OpenAI-style message dicts in the order they
            should be sent to the model.
        """
        messages: list[dict[str, Any]] = [{"role": "system", "content": system_prompt}]

        if not session_history:
            for turn in conversation:
                messages.append({"role": "user", "content": turn.question})
                messages.append({"role": "assistant", "content": turn.answer})

        if session_history:
            for session_item in session_history:
                role = session_item.get("role", "user")
                if role not in {"user", "assistant", "system"}:
                    role = "user"
                messages.append({"role": role, "content": session_item.get("content", "")})

        if context:
            formatted_context = "\n\n---\n\n".join(context)
            messages.append({"role": "system", "content": f"Context:\n{formatted_context}"})

        if image_paths:
            human_content: list[dict[str, Any]] = [{"type": "text", "text": question}]
            for path in image_paths:
                with open(path, "rb") as f:
                    encoded = base64.b64encode(f.read()).decode("utf-8")
                mime_type, _ = mimetypes.guess_type(path)
                if mime_type is None:
                    mime_type = "image/png"
                human_content.append(
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:{mime_type};base64,{encoded}"},
                    }
                )
            messages.append({"role": "user", "content": human_content})
        else:
            messages.append({"role": "user", "content": question})

        return messages

    def generate(
        self,
        *,
        system_prompt: str,
        conversation: Sequence[ConversationTurn] = (),
        context: Sequence[str] = (),
        question: str,
        image_paths: list[str] | None = None,
        session_history: list[dict[str, Any]] | None = None,
    ) -> str:
        """Generate a final string answer.

        Also populates ``self.last_usage`` with a dict of token
        counts so the RAG facade can record them to telemetry.
        """
        messages = self.build_messages(
            system_prompt=system_prompt,
            conversation=conversation,
            context=context,
            question=question,
            image_paths=image_paths,
            session_history=session_history,
        )
        self.require_litellm()
        options = {
            "model": self.model_name,
            "messages": messages,
            "temperature": self.temperature,
            "api_key": self.api_key,
            "api_base": self.api_base,
        }
        if self.timeout_seconds is not None:
            options["timeout"] = self.timeout_seconds
        with LLMValueErrorBoundary("LiteLLM completion failed"):
            response = litellm.completion(**options)

        choice = response["choices"][0] if isinstance(response, dict) else response.choices[0]
        message = choice["message"] if isinstance(choice, dict) else choice.message
        self.record_usage(response)
        content = message["content"] if isinstance(message, dict) else message.content
        return str(content or "")

    async def async_generate(
        self,
        *,
        system_prompt: str,
        conversation: Sequence[ConversationTurn] = (),
        context: Sequence[str] = (),
        question: str,
        image_paths: list[str] | None = None,
        session_history: list[dict[str, Any]] | None = None,
    ) -> str:
        """Generate a final answer with LiteLLM's native async client."""
        messages = self.build_messages(
            system_prompt=system_prompt,
            conversation=conversation,
            context=context,
            question=question,
            image_paths=image_paths,
            session_history=session_history,
        )
        self.require_litellm()
        options = {
            "model": self.model_name,
            "messages": messages,
            "temperature": self.temperature,
            "api_key": self.api_key,
            "api_base": self.api_base,
        }
        if self.timeout_seconds is not None:
            options["timeout"] = self.timeout_seconds
        with LLMValueErrorBoundary("LiteLLM async completion failed"):
            response = await litellm.acompletion(**options)
        choice = response["choices"][0] if isinstance(response, dict) else response.choices[0]
        message = choice["message"] if isinstance(choice, dict) else choice.message
        self.record_usage(response)
        content = message["content"] if isinstance(message, dict) else message.content
        return str(content or "")

    def record_usage(self, response: Any) -> None:
        """Populate ``self.last_usage`` from a LiteLLM response.

        Args:
            response: The raw LiteLLM response.
        """
        usage: Any = None
        if isinstance(response, dict):
            usage = response.get("usage")
        else:
            usage = getattr(response, "usage", None)
        if usage is None:
            return
        if isinstance(usage, dict):
            prompt = usage.get("prompt_tokens") or usage.get("input_tokens") or 0
            completion = usage.get("completion_tokens") or usage.get("output_tokens") or 0
        else:
            prompt = getattr(usage, "prompt_tokens", 0) or 0
            completion = getattr(usage, "completion_tokens", 0) or 0
        self.last_usage = {
            "prompt": int(prompt),
            "completion": int(completion),
            "model": self.model_name,
        }

    async def astream(
        self,
        *,
        system_prompt: str,
        conversation: Sequence[ConversationTurn] = (),
        context: Sequence[str] = (),
        question: str,
        image_paths: list[str] | None = None,
        session_history: list[dict[str, Any]] | None = None,
    ) -> AsyncIterator[str]:
        """Async-stream the answer token-by-token.

        Token usage is captured by asking LiteLLM to include it in
        the final chunk (``stream_options={"include_usage": True}``).
        The streaming loop honours :pyattr:`timeout_seconds` with
        :func:`asyncio.timeout` so a slow LLM does not block indefinitely.
        """
        messages = self.build_messages(
            system_prompt=system_prompt,
            conversation=conversation,
            context=context,
            question=question,
            image_paths=image_paths,
            session_history=session_history,
        )
        self.require_litellm()
        options = {
            "model": self.model_name,
            "messages": messages,
            "temperature": self.temperature,
            "stream": True,
            "stream_options": {"include_usage": True},
            "api_key": self.api_key,
            "api_base": self.api_base,
        }
        if self.timeout_seconds is not None:
            options["timeout"] = self.timeout_seconds
        with LLMValueErrorBoundary("LiteLLM streaming failed"):
            response = await litellm.acompletion(**options)

        prompt_tokens = 0
        completion_tokens = 0
        async with asyncio.timeout(self.timeout_seconds):
            async for chunk in response:
                usage = (
                    chunk.get("usage") if isinstance(chunk, dict) else getattr(chunk, "usage", None)
                )
                if usage:
                    if isinstance(usage, dict):
                        prompt_tokens = usage.get("prompt_tokens") or usage.get("input_tokens") or 0
                        completion_tokens = (
                            usage.get("completion_tokens") or usage.get("output_tokens") or 0
                        )
                    else:
                        prompt_tokens = getattr(usage, "prompt_tokens", 0) or 0
                        completion_tokens = getattr(usage, "completion_tokens", 0) or 0
                if not isinstance(chunk, dict):
                    continue
                choices = chunk.get("choices") or []
                if not choices:
                    continue
                delta = choices[0].get("delta") or {}
                content = delta.get("content")
                if content:
                    yield content

        if prompt_tokens or completion_tokens:
            self.last_usage = {
                "prompt": int(prompt_tokens),
                "completion": int(completion_tokens),
                "model": self.model_name,
            }


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
    return LiteLLMProvider(model=model_name, api_key=api_key)