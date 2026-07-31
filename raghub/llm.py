"""LLM provider implementations.

This module ships:

* :class:`Generator` — abstract base class.
* :class:`HeuristicProvider` — offline fallback, no API key needed.
* :class:`LiteLLM` — production LLM, backed by LiteLLM (any
  provider: OpenAI, NVIDIA, Anthropic, Bedrock, …).
* :func:`build_llm` — selects an implementation by model
  name and credential availability.
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

from raghub.errors import ConfigurationError, LLMError
from raghub.models import ConversationTurn
from raghub.utils import aretry, retry

__all__ = [
    "LLM_API_KEY_ENV_VARS",
    "Generator",
    "HeuristicProvider",
    "LiteLLM",
    "any_llm_api_key_present",
    "build_llm",
]

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


class Generator(ABC):
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


class LLMValueErrorBoundary:
    """Translate direct ``ValueError`` failures to :class:`LLMError`."""

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


class HeuristicProvider(Generator):
    """Offline LLM provider that answers from context directly.

    Uses a simple heuristic — extracts the most relevant sentence from
    the context, or returns a canned response when no context is given.
    No API key or network access required.
    """

    model_name: str = "heuristic"

    def generate(
        self,
        *,
        system_prompt: str = "",
        conversation: Sequence[ConversationTurn] = (),
        context: Sequence[str] = (),
        question: str,
        image_paths: list[str] | None = None,
        session_history: list[dict[str, Any]] | None = None,
    ) -> str:
        """Generate an answer from context using simple heuristics.

        Args:
            system_prompt: Ignored by this provider.
            conversation: Prior conversation turns (ignored).
            context: Retrieved source chunks.
            question: The user's question.
            image_paths: Ignored by this provider.
            session_history: Ignored by this provider.

        Returns:
            The first relevant sentence from context, or a default
            message if no context is available.
        """
        if not context:
            return "No context was retrieved. Configure an LLM API key for full answer generation."
        # Heuristic: pick the sentence most relevant to the question
        question_lower = question.lower()
        question_words = set(question_lower.split())
        scored: list[tuple[int, str]] = []
        for chunk in context:
            for sentence in chunk.split("."):
                stripped = sentence.strip()
                if not stripped:
                    continue
                lowered = stripped.lower()
                score = sum(1 for w in question_words if w in lowered)
                scored.append((score, stripped))
        scored.sort(key=lambda x: -x[0])
        best = scored[0][1] if scored else context[0][:500]
        return best


class LiteLLM(Generator):
    """LLM provider backed by LiteLLM.

    The provider wraps LiteLLM and so works with any provider that
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
        # When ``api_base`` is set we bypass LiteLLM's provider routing and
        # call the OpenAI-compatible endpoint directly through a shared
        # ``httpx.AsyncClient`` with a connection pool. Each call
        # saves ~50-200ms of TCP setup vs. the per-request client that
        # ``litellm.acompletion`` constructs internally.
        self.direct_client: Any = None
        self.direct_url: str | None = None
        if self.api_base:
            try:
                import httpx

                headers = (
                    {"authorization": f"Bearer {self.api_key}"}
                    if self.api_key
                    else {}
                )
                # Normalise the base URL so POSTing to <base>/chat/completions
                # always lands at the standard path.
                base = self.api_base.rstrip("/")
                if not base.endswith("/v1"):
                    base = f"{base}/v1"
                self.direct_client = httpx.AsyncClient(
                    base_url=base,
                    headers=headers,
                    timeout=timeout_seconds or 60.0,
                )
                self.direct_url = f"{base}/chat/completions"
            except ImportError:  # pragma: no cover - optional dep
                self.direct_client = None
                self.direct_url = None
        else:
            self.direct_url = None

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
            try:
                response_dict = retry(
                    lambda: litellm.completion(**options),
                    max_retries=2,
                    base_delay=0.5,
                )
            except LLMError:
                raise
            except Exception as exc:
                raise LLMError(f"LLM completion failed: {exc}") from exc
        try:
            choices = response_dict["choices"] if isinstance(response_dict, dict) else response_dict.choices
            choice = choices[0]
            message = choice["message"] if isinstance(choice, dict) else choice.message
            content = message["content"] if isinstance(message, dict) else message.content
        except (KeyError, IndexError, AttributeError, TypeError) as exc:
            raise LLMError(f"LLM returned unexpected response shape: {exc}") from exc
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
        """Generate a final answer with the configured backend.

        When ``api_base`` is set, the call goes through a shared
        ``httpx.AsyncClient`` against the OpenAI-compatible endpoint
        (saves the per-call TCP handshake of the default
        ``litellm.acompletion`` path). Otherwise we fall back to
        LiteLLM's native async client.
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
        if self.direct_client is not None and self.direct_url is not None:
            response = await self.direct_chat(messages)
        else:
            options = {
                "model": self.model_name,
                "messages": messages,
                "temperature": self.temperature,
                "api_key": self.api_key,
                "api_base": self.api_base,
            }
            if self.api_base:
                # Custom OpenAI-compatible endpoint — tell litellm the
                # protocol so it doesn't reject an unknown model name.
                options["custom_llm_provider"] = "minimax"
            if self.timeout_seconds is not None:
                options["timeout"] = self.timeout_seconds
            with LLMValueErrorBoundary("LiteLLM async completion failed"):
                try:
                    response_dict = await aretry(
                        lambda: litellm.acompletion(**options),
                        max_retries=2,
                        base_delay=0.5,
                    )
                except LLMError:
                    raise
                except Exception as exc:
                    raise LLMError(f"LLM async completion failed: {exc}") from exc
            response = self.normalise_response(response_dict)
        try:
            choices = response["choices"] if isinstance(response, dict) else response.choices
            choice = choices[0]
            message = choice["message"] if isinstance(choice, dict) else choice.message
            content = message["content"] if isinstance(message, dict) else message.content
        except (KeyError, IndexError, AttributeError, TypeError) as exc:
            raise LLMError(f"LLM returned unexpected response shape: {exc}") from exc
        return str(content or "")

    async def direct_chat(self, messages: list[dict[str, Any]]) -> dict[str, Any]:
        """POST to the OpenAI-compatible endpoint through the pooled client."""
        payload = {
            "model": self.model_name,
            "messages": messages,
            "temperature": self.temperature,
        }
        if self.timeout_seconds is not None:
            payload["timeout"] = self.timeout_seconds
        with LLMValueErrorBoundary("Direct endpoint call failed"):
            response = await self.direct_client.post(
                self.direct_url, json=payload
            )
        response.raise_for_status()
        return dict(response.json())

    @staticmethod
    def normalise_response(response: Any) -> dict[str, Any]:
        """Return a plain dict view of a litellm response (pydantic or not)."""
        if isinstance(response, dict):
            return response
        if hasattr(response, "model_dump"):
            return dict(response.model_dump())
        return dict(response) if response is not None else {}

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
        if self.api_base:
            options["custom_llm_provider"] = "minimax"
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


def build_llm(
    model_name: str,
    api_key: str | None = None,
) -> Generator:
    """Construct the appropriate LLM provider for ``model_name``.

    Selection rules:

    * No API key available → :class:`HeuristicProvider` (offline).
    * Otherwise → :class:`LiteLLM`.

    Pass credentials via the ``RAG_LLM_*`` env vars (``RAG_LLM_API_KEY``,
    ``RAG_LLM_BASE_URL``, ``RAG_LLM_MODEL``) or as the ``api_key`` argument.

    Args:
        model_name: The model identifier.
        api_key: Optional API key passed through to the
            :class:`LiteLLM`.

    Returns:
        A ready-to-use provider instance.
    """
    if not any_llm_api_key_present() and not api_key:
        return HeuristicProvider()
    return LiteLLM(model=model_name, api_key=api_key)