"""NVIDIA LLM provider backed by :mod:`langchain_nvidia_ai_endpoints`.

This is the production LLM provider. It uses the ``ChatNVIDIA`` SDK
to call NVIDIA-hosted models (defaults to ``nvidia/nemotron-4-340b-instruct``)
and wraps the call in the package's standard exponential-backoff retry
helper so transient 429/5xx errors are absorbed before bubbling up.
"""

from __future__ import annotations

import base64
import mimetypes
from collections.abc import Sequence
from typing import Any, cast

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_nvidia_ai_endpoints import ChatNVIDIA

from raghub.llm.base import BaseLLMProvider
from raghub.models import ConversationTurn
from raghub.utils.retry import retry as _retry


class NvidiaLLMProvider(BaseLLMProvider):
    """LLM provider backed by ``ChatNVIDIA``.

    Attributes:
        model_name: The fully-qualified NVIDIA model id, e.g.
            ``"nvidia/nemotron-4-340b-instruct"``.
    """

    def __init__(
        self,
        model: str = "nvidia/nemotron-4-340b-instruct",
        api_key: str | None = None,
        timeout: int = 120,
    ) -> None:
        """Initialise the provider.

        Args:
            model: NVIDIA model id. Default is the Nemotron 340B
                instruction-tuned model.
            api_key: NVIDIA API key. When ``None``, the SDK falls back
                to the ``NVIDIA_API_KEY`` environment variable.
            timeout: Per-call timeout in seconds.
        """
        self.model_name = model
        self.client = ChatNVIDIA(model=model, timeout=timeout, api_key=api_key)

    def generate(
        self,
        *,
        system_prompt: str,
        conversation: Sequence[ConversationTurn],
        context: Sequence[str],
        question: str,
        image_paths: list[str] | None = None,
        session_history: list[dict] | None = None,
    ) -> str:
        """Generate an answer via the NVIDIA chat endpoint.

        Builds a LangChain message list, invokes the model, and
        unwraps the result into a string. The invocation is wrapped
        in :func:`raghub.utils.retry.retry` so transient upstream
        errors are retried automatically.

        Args:
            system_prompt: System instructions to seed the model.
            conversation: Recent in-window turns.
            context: Retrieved chunks; joined into a single
                ``"Context:"`` system message.
            question: The latest user question.
            image_paths: Optional list of on-disk image paths.
                Each path is base64-encoded and embedded as a data
                URL in the user message.
            session_history: Optional prior turns rendered as
                ``HumanMessage``/``AIMessage``/``SystemMessage``
                entries.

        Returns:
            The model's reply as a string.
        """
        messages = self.build_messages(
            system_prompt=system_prompt,
            conversation=conversation,
            context=context,
            question=question,
            image_paths=image_paths,
            session_history=session_history,
        )
        return _retry(lambda: cast(str, self.client.invoke(messages).content))

    def build_messages(
        self,
        *,
        system_prompt: str,
        conversation: Sequence[ConversationTurn],
        context: Sequence[str],
        question: str,
        image_paths: list[str] | None = None,
        session_history: list[dict] | None = None,
    ) -> list[BaseMessage]:
        """Assemble a LangChain message list ready for invocation.

        Args:
            system_prompt: System instructions; becomes the first
                ``SystemMessage``.
            conversation: Recent in-window turns (currently unused
                here; the API keeps the parameter for future use).
            context: Retrieved chunks; joined with ``"\\n\\n---\\n\\n"``
                into a single ``SystemMessage`` labelled ``"Context:"``.
            question: The latest user question. When ``image_paths``
                is empty, the question is rendered as a plain string;
                otherwise it is rendered as a content array with text
                plus one ``image_url`` entry per file.
            image_paths: Optional list of on-disk image paths.
            session_history: Optional prior turns; ``role``
                determines the message class (``user`` → ``HumanMessage``,
                ``assistant`` → ``AIMessage``, anything else →
                ``SystemMessage``).

        Returns:
            A list of :class:`langchain_core.messages.BaseMessage`
            instances in the order they should be sent to the model.
        """
        messages: list[BaseMessage] = [
            SystemMessage(content=system_prompt),
        ]

        if session_history:
            for turn in session_history:
                role = turn.get("role", "user")
                content = turn.get("content", "")
                if role == "user":
                    messages.append(HumanMessage(content=content))
                elif role == "assistant":
                    messages.append(AIMessage(content=content))
                else:
                    messages.append(SystemMessage(content=content))

        if context:
            formatted_context = "\n\n---\n\n".join(context)
            messages.append(SystemMessage(content=f"Context:\n{formatted_context}"))

        if image_paths:
            human_content: list[str | dict[str, Any]] = [{"type": "text", "text": question}]
            for path in image_paths:
                # Vision-capable models accept base64 data URLs; we
                # base64-encode the file bytes and embed them inline.
                with open(path, "rb") as f:
                    encoded = base64.b64encode(f.read()).decode("utf-8")
                mime_type, _ = mimetypes.guess_type(path)
                if mime_type is None:
                    # Default to PNG when the OS cannot infer the MIME;
                    # this is the common case for ``.png`` files on
                    # platforms that strip extension associations.
                    mime_type = "image/png"
                human_content.append(
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:{mime_type};base64,{encoded}"},
                    }
                )
            messages.append(HumanMessage(content=human_content))
        else:
            messages.append(HumanMessage(content=question))

        return messages