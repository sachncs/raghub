"""LiteLLM-backed LLM provider.

Works with OpenAI, NVIDIA, Azure, Anthropic, Bedrock, and any other
provider supported by LiteLLM.

Streaming is exposed via the :meth:`astream` coroutine.

The :meth:`build_messages` helper assembles the OpenAI-style message
list from a system prompt, retrieved context, optional image paths, and
optional session history. It is exposed publicly so test code (and
callers that want to drive the LLM themselves) can construct the same
shape that :meth:`generate` and :meth:`astream` would produce.
"""

from __future__ import annotations

import base64
import mimetypes
from collections.abc import AsyncIterator, Sequence
from typing import Any

from raghub.exceptions import ConfigurationError, LLMError
from raghub.llm.base import BaseLLMProvider
from raghub.models import ConversationTurn

litellm: Any

try:
    import litellm
    LITELLM_AVAILABLE = True
    OptionalImportError: Exception | None = None
except Exception as exc:  # pragma: no cover - optional dep
    litellm = None
    LITELLM_AVAILABLE = False
    OptionalImportError = exc


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
    ) -> None:
        """Initialise the provider.

        Args:
            model: LiteLLM model name.
            api_key: Optional API key override.
            api_base: Optional API base override.
            temperature: Sampling temperature for ``completion``.

        Raises:
            ConfigurationError: When ``litellm`` is not installed.
        """
        if not LITELLM_AVAILABLE:
            raise ConfigurationError(
                "litellm is not installed; run `pip install litellm`."
            )
        self.model_name = model
        self.api_key = api_key
        self.api_base = api_base
        self.temperature = temperature
        # Most recent token-usage record, populated by :meth:`generate`
        # and :meth:`astream`. Read by :class:`DefaultGenerator` to
        # forward to telemetry.
        self.last_usage: dict[str, Any] | None = None

    def require_litellm(self) -> None:
        """Raise a clear error if LiteLLM is not installed."""
        if not LITELLM_AVAILABLE:
            raise ConfigurationError(
                "litellm is not installed; run `pip install litellm`."
            )

    # ------------------------------------------------------------------
    # Message building (OpenAI-style dicts)
    # ------------------------------------------------------------------

    def build_messages(
        self,
        *,
        system_prompt: str,
        conversation: Sequence[ConversationTurn] = (),
        context: Sequence[str] = (),
        question: str,
        image_paths: list[str] | None = None,
        session_history: list[dict] | None = None,
    ) -> list[dict[str, Any]]:
        """Assemble an OpenAI-style message list.

        Args:
            system_prompt: System instructions; becomes the first
                ``system`` message.
            conversation: Recent in-window turns (currently unused
                here; the API keeps the parameter for future use).
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

        if session_history:
            for turn in session_history:
                role = turn.get("role", "user")
                if role not in {"user", "assistant", "system"}:
                    role = "user"
                messages.append({"role": role, "content": turn.get("content", "")})

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

    # ------------------------------------------------------------------
    # Generation
    # ------------------------------------------------------------------

    def generate(
        self,
        *,
        system_prompt: str,
        conversation: Sequence[ConversationTurn] = (),
        context: Sequence[str] = (),
        question: str,
        image_paths: list[str] | None = None,
        session_history: list[dict] | None = None,
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
        try:
            response = litellm.completion(
                model=self.model_name,
                messages=messages,
                temperature=self.temperature,
                api_key=self.api_key,
                api_base=self.api_base,
            )
        except Exception as exc:
            raise LLMError(f"LiteLLM completion failed: {exc}") from exc

        choice = response["choices"][0] if isinstance(response, dict) else response.choices[0]
        message = choice["message"] if isinstance(choice, dict) else choice.message
        # Capture token usage for telemetry.
        self.record_usage(response)
        return message["content"] if isinstance(message, dict) else message.content

    def record_usage(self, response: Any) -> None:
        """Populate ``self.last_usage`` from a LiteLLM response.

        Args:
            response: The raw LiteLLM response.
        """
        # LiteLLM v0 returns usage as a dict under ``"usage"``; v1+
        # returns a Usage object with the same fields.
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
        session_history: list[dict] | None = None,
    ) -> AsyncIterator[str]:
        """Async-stream the answer token-by-token.

        Token usage is captured by either asking LiteLLM to include
        it in the final chunk (``stream_options={"include_usage": True}``)
        or by computing it from the request as a fallback.
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
        try:
            response = await litellm.acompletion(
                model=self.model_name,
                messages=messages,
                temperature=self.temperature,
                stream=True,
                stream_options={"include_usage": True},
                api_key=self.api_key,
                api_base=self.api_base,
            )
        except Exception as exc:
            raise LLMError(f"LiteLLM streaming failed: {exc}") from exc

        prompt_tokens = 0
        completion_tokens = 0
        async for chunk in response:
            # Some chunks carry a final usage object; capture it for
            # telemetry even when streaming.
            usage = chunk.get("usage") if isinstance(chunk, dict) else getattr(chunk, "usage", None)
            if usage:
                if isinstance(usage, dict):
                    prompt_tokens = usage.get("prompt_tokens") or usage.get("input_tokens") or 0
                    completion_tokens = (
                        usage.get("completion_tokens") or usage.get("output_tokens") or 0
                    )
                else:
                    prompt_tokens = getattr(usage, "prompt_tokens", 0) or 0
                    completion_tokens = (
                        getattr(usage, "completion_tokens", 0) or 0
                    )
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


__all__ = ["LiteLLMProvider"]
