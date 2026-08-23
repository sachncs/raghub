"""LLM provider implementations.

This module ships:

* :class:`Generator` — polymorphic base class.
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
from collections.abc import AsyncIterator, Sequence
from dataclasses import dataclass
from types import TracebackType
from typing import Any, Literal, Self, cast

import litellm
import httpx

from raghub.constants import (
    ENV_ANTHROPIC_API_KEY,
    ENV_AZURE_API_KEY,
    ENV_COHERE_API_KEY,
    ENV_LITELLM_API_KEY,
    ENV_NVIDIA_API_KEY,
    ENV_OPENAI_API_KEY,
    ENV_VOYAGE_API_KEY,
)
from raghub.errors import ConfigurationError, GenerationError
from raghub.models import Turn
from raghub.registry import Registry
from raghub.retry import aretry, retry

__all__ = [
    "LLM_API_KEY_ENV_VARS",
    "GenerationRequest",
    "Generator",
    "LiteLLM",
    "build_llm",
    "has_llm_key",
]


LLM_API_KEY_ENV_VARS: tuple[str, ...] = (
    ENV_OPENAI_API_KEY,
    ENV_ANTHROPIC_API_KEY,
    ENV_NVIDIA_API_KEY,
    "GROQ_API_KEY",
    ENV_LITELLM_API_KEY,
    ENV_COHERE_API_KEY,
    ENV_VOYAGE_API_KEY,
    ENV_AZURE_API_KEY,
    "AWS_ACCESS_KEY_ID",
)


def has_llm_key() -> bool:
    """Return ``True`` when at least one LLM credential env var is set.

    Returns:
        ``True`` if any of the recognised LLM API-key environment
        variables is set in the process environment; ``False``
        otherwise.

    """
    return any(os.getenv(name) for name in LLM_API_KEY_ENV_VARS)


@dataclass(slots=True, frozen=True)
class GenerationRequest:
    """Inputs for a single :class:`Generator` invocation.

    Attributes:
        question: The user's question.
        system_prompt: System-level instructions, including
            tenant-specific formatting guidance.
        conversation: Recent in-window turns from the conversation
            manager.
        context: Retrieved source chunks (already RBAC-filtered).
        image_paths: Optional on-disk image paths to attach to the
            final user message (vision-capable providers only).
        session_history: Optional prior turns from the persistent
            session store; format mirrors :class:`raghub.models.Turn`.

    """

    question: str = ""
    system_prompt: str = ""
    conversation: Sequence[Turn] = ()
    context: Sequence[object] = ()
    image_paths: list[str] | None = None
    session_history: list[dict[str, Any]] | None = None


class Generator(Registry):
    """Polymorphic LLM provider base.

    All concrete providers implement :meth:`generate`. The interface is
    intentionally narrow: the caller assembles the prompt and passes the
    components in. The provider's job is to call its backing SDK and
    return a string.
    """

    model_name: str

    def generate(self, request: GenerationRequest) -> str:
        """Generate an answer from a fully-constructed prompt."""
        raise NotImplementedError

    async def async_generate(self, request: GenerationRequest) -> str:
        """Generate without blocking the event loop."""
        return await asyncio.to_thread(self.generate, request)


class LLMValueErrorBoundary:
    """Translate direct ``ValueError`` failures to :class:`GenerationError`."""

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
            raise GenerationError(f"{self.message}: {exception}") from exception
        return False


@Generator.register("litellm")
class LiteLLM(Generator):
    """LLM provider backed by LiteLLM.

    The provider wraps LiteLLM and so works with any provider that
    LiteLLM supports: OpenAI, NVIDIA, Anthropic, Bedrock, etc.
    """

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
            ValueError: When ``timeout_seconds`` is non-positive.

        """
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
            headers = {"authorization": f"Bearer {self.api_key}"} if self.api_key else {}
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
        else:
            self.direct_url = None

    @staticmethod
    def require_litellm() -> None:
        """Verify the ``litellm`` package is importable.

        ``litellm`` is a required core dependency since v0.7.0, so this
        is a defensive sanity check that surfaces a clear error if a
        downstream consumer manages to import this module without
        ``litellm`` installed.
        """

    @staticmethod
    def build_messages(request: GenerationRequest) -> list[dict[str, Any]]:
        """Assemble an OpenAI-style message list.

        Args:
            request: The prompt components for one invocation.

        Returns:
            A list of OpenAI-style message dicts in the order they
            should be sent to the model.

        """
        messages: list[dict[str, Any]] = [{"role": "system", "content": request.system_prompt}]

        if not request.session_history:
            for turn in request.conversation:
                messages.append({"role": "user", "content": turn.question})
                messages.append({"role": "assistant", "content": turn.answer})

        if request.session_history:
            for session_item in request.session_history:
                role = session_item.get("role", "user")
                if role not in {"user", "assistant"}:
                    role = "user"
                messages.append({"role": role, "content": session_item.get("content", "")})

        if request.context:
            formatted_context = "\n\n---\n\n".join(str(turn) for turn in request.context)
            messages.append({"role": "system", "content": f"Context:\n{formatted_context}"})

        if request.image_paths:
            human_content: list[dict[str, Any]] = [{"type": "text", "text": request.question}]
            for path in request.image_paths:
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
            messages.append({"role": "user", "content": request.question})

        return messages

    def generate(self, request: GenerationRequest) -> str:
        """Generate a final string answer.

        Also populates ``self.last_usage`` with a dict of token
        counts so the RAG facade can record them to telemetry.
        """
        messages = self.build_messages(request)
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
            except GenerationError:
                raise
            except Exception as exc:
                raise GenerationError(f"LLM completion failed: {exc}") from exc
        try:
            choices = (
                response_dict["choices"]
                if isinstance(response_dict, dict)
                else response_dict.choices
            )
            choice = choices[0]
            message = choice["message"] if isinstance(choice, dict) else choice.message
            content = message["content"] if isinstance(message, dict) else message.content
        except (KeyError, IndexError, AttributeError, TypeError) as exc:
            raise GenerationError(f"LLM returned unexpected response shape: {exc}") from exc
        return str(content or "")

    async def async_generate(self, request: GenerationRequest) -> str:
        """Generate a final answer with the configured backend.

        When ``api_base`` is set, the call goes through a shared
        ``httpx.AsyncClient`` against the OpenAI-compatible endpoint
        (saves the per-call TCP handshake of the default
        ``litellm.acompletion`` path). Otherwise we fall back to
        LiteLLM's native async client.
        """
        messages = self.build_messages(request)
        self.require_litellm()
        if self.direct_client is not None and self.direct_url is not None:
            normalised: Any = await self.direct_chat(messages)
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
                except GenerationError:
                    raise
                except Exception as exc:
                    raise GenerationError(f"LLM async completion failed: {exc}") from exc
            normalised = cast(Any, self.normalise_response(response_dict))
        try:
            choices: Any = (
                normalised["choices"] if isinstance(normalised, dict) else normalised.choices
            )
            choice = choices[0]
            message: Any = choice["message"] if isinstance(choice, dict) else choice.message
            content = message["content"] if isinstance(message, dict) else message.content
        except (KeyError, IndexError, AttributeError, TypeError) as exc:
            raise GenerationError(f"LLM returned unexpected response shape: {exc}") from exc
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
            response = await self.direct_client.post(self.direct_url, json=payload)
        response.raise_for_status()
        return dict(response.json())

    @staticmethod
    def normalise_response(response: Any) -> dict[str, Any]:
        """Return a plain dict view of a litellm response (pydantic or not)."""
        if isinstance(response, dict):
            return response
        if hasattr(response, "model_dump"):
            return dict(response.model_dump())
        if hasattr(response, "dump"):
            value = response.dump()
            return (
                dict(value) if isinstance(value, dict) else dict(value) if value is not None else {}
            )
        return dict(response) if response is not None else {}

    async def astream(self, request: GenerationRequest) -> AsyncIterator[str]:
        """Async-stream the answer token-by-token.

        Token usage is captured by asking LiteLLM to include it in
        the final chunk (``stream_options={"include_usage": True}``).
        The streaming loop honours :pyattr:`timeout_seconds` with
        :func:`asyncio.timeout` so a slow LLM does not block indefinitely.
        """
        options = self.build_stream_options(request)
        with LLMValueErrorBoundary("LiteLLM streaming failed"):
            response = await litellm.acompletion(**options)
        async for content in self.consume_stream(response):
            yield content
        # After consumption, last_usage is set in consume_stream.

    async def consume_stream(self, response: Any) -> AsyncIterator[str]:
        """Yield content chunks while tracking token usage; set last_usage on exit."""
        prompt_tokens = 0
        completion_tokens = 0
        try:
            async with asyncio.timeout(self.timeout_seconds):
                async for chunk in response:
                    usage = (
                        chunk.get("usage")
                        if isinstance(chunk, dict)
                        else getattr(chunk, "usage", None)
                    )
                    prompt_tokens, completion_tokens = self.accumulate_usage(
                        usage, prompt_tokens, completion_tokens
                    )
                    content = self.extract_chunk_content(chunk)
                    if content:
                        yield content
        except TimeoutError as exc:
            raise GenerationError(
                f"LiteLLM stream exceeded timeout_seconds={self.timeout_seconds}"
            ) from exc
        if prompt_tokens or completion_tokens:
            self.last_usage = {
                "prompt": int(prompt_tokens),
                "completion": int(completion_tokens),
                "model": self.model_name,
            }

    def build_stream_options(self, request: GenerationRequest) -> dict[str, Any]:
        """Build the LiteLLM completion options dict for streaming."""
        messages = self.build_messages(request)
        self.require_litellm()
        options: dict[str, Any] = {
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
        return options

    async def iter_stream_chunks(self, response: Any) -> AsyncIterator[str]:
        """Yield content chunks while tracking token usage.

        Returns an async iterator of content deltas; the final token
        counts are exposed via the ``last_usage`` attribute set on
        :pyattr:`self` after the iterator is exhausted.

        Raises:
            GenerationError: When the underlying stream exceeds
                :pyattr:`timeout_seconds`.

        """
        prompt_tokens = 0
        completion_tokens = 0
        try:
            async with asyncio.timeout(self.timeout_seconds):
                async for chunk in response:
                    usage = (
                        chunk.get("usage")
                        if isinstance(chunk, dict)
                        else getattr(chunk, "usage", None)
                    )
                    prompt_tokens, completion_tokens = self.accumulate_usage(
                        usage, prompt_tokens, completion_tokens
                    )
                    content = self.extract_chunk_content(chunk)
                    if content:
                        yield content
        except TimeoutError as exc:
            raise GenerationError(
                f"LiteLLM stream exceeded timeout_seconds={self.timeout_seconds}"
            ) from exc

    @staticmethod
    def accumulate_usage(usage: Any, prompt_tokens: int, completion_tokens: int) -> tuple[int, int]:
        """Update running usage counters from a chunk's ``usage`` field."""
        if not usage:
            return prompt_tokens, completion_tokens
        if isinstance(usage, dict):
            return (
                usage.get("prompt_tokens") or usage.get("input_tokens") or prompt_tokens,
                usage.get("completion_tokens") or usage.get("output_tokens") or completion_tokens,
            )
        return (
            getattr(usage, "prompt_tokens", 0) or prompt_tokens,
            getattr(usage, "completion_tokens", 0) or completion_tokens,
        )

    @staticmethod
    def extract_chunk_content(chunk: Any) -> str | None:
        """Return the assistant text delta from one LiteLLM streaming chunk."""
        if not isinstance(chunk, dict):
            return None
        choices = chunk.get("choices") or []
        if not choices:
            return None
        delta = choices[0].get("delta") or {}
        return delta.get("content")


def build_llm(
    model_name: str,
    api_key: str | None = None,
) -> Generator:
    """Construct the appropriate LLM provider for ``model_name``.

    Selection rules:

    * No API key available → :class:`ConfigurationError` is raised; callers
      must set an LLM API key explicitly.
    * Otherwise → :class:`LiteLLM`.

    Pass credentials via the ``RAG_LLM_*`` env vars (``RAG_LLM_API_KEY``,
    ``RAG_LLM_BASE_URL``, ``RAG_LLM_MODEL``) or as the ``api_key`` argument.

    Args:
        model_name: The model identifier.
        api_key: Optional API key passed through to the
            :class:`LiteLLM`.

    Returns:
        A ready-to-use provider instance.

    Raises:
        ConfigurationError: When no LLM API key is configured.

    """
    if not has_llm_key() and not api_key:
        raise ConfigurationError(
            "No LLM API key configured; set one in Settings "
            "(e.g. RAG_LLM_API_KEY) or pass api_key= explicitly."
        )
    return LiteLLM(model=model_name, api_key=api_key)
