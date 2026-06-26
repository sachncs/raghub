"""MiniMax M3 multimodal LLM via ChatNVIDIA."""

from __future__ import annotations

import base64
import mimetypes

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_nvidia_ai_endpoints import ChatNVIDIA


class NvidiaLLMProvider:
    """MiniMax M3 multimodal LLM via ChatNVIDIA."""

    def __init__(
        self,
        model: str = "minimaxai/minimax-m3",
        api_key: str | None = None,
    ) -> None:
        self.model_name = model
        kwargs = {"model": model}
        if api_key is not None:
            kwargs["api_key"] = api_key
        self._client = ChatNVIDIA(**kwargs)

    def generate(
        self,
        question: str,
        context: list[dict] | None = None,
        image_paths: list[str] | None = None,
        session_history: list[dict] | None = None,
    ) -> str:
        messages = self._build_messages(question, context, image_paths, session_history)
        return self._client.invoke(messages).content

    async def generate_async(
        self,
        question: str,
        context: list[dict] | None = None,
        image_paths: list[str] | None = None,
        session_history: list[dict] | None = None,
    ) -> str:
        messages = self._build_messages(question, context, image_paths, session_history)
        result = await self._client.ainvoke(messages)
        return result.content

    def _build_messages(
        self,
        question: str,
        context: list[dict] | None = None,
        image_paths: list[str] | None = None,
        session_history: list[dict] | None = None,
    ) -> list[BaseMessage]:
        messages: list[BaseMessage] = [
            SystemMessage(content="You are a helpful RAG assistant. Answer based on the provided context.")
        ]

        if session_history:
            for turn in session_history:
                role = turn.get("role", "user")
                content = turn.get("content", "")
                if role == "user":
                    messages.append(HumanMessage(content=content))
                else:
                    messages.append(SystemMessage(content=content))

        if context:
            formatted_context = "\n\n".join(
                c.get("text", str(c)) for c in context
            )
            messages.append(SystemMessage(content=f"Context:\n{formatted_context}"))

        if image_paths:
            human_content: list[dict] = [{"type": "text", "text": question}]
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
            messages.append(HumanMessage(content=human_content))
        else:
            messages.append(HumanMessage(content=question))

        return messages
