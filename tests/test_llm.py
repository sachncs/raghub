"""Tests for LLM message building.

Exercises :meth:`NvidiaLLMProvider.build_messages` against the
standard set of inputs (system prompt, retrieved context,
session history, images). The provider is instantiated with a
placeholder API key; no actual inference occurs in these tests.
"""

from __future__ import annotations

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from raghub.llm.nvidia import NvidiaLLMProvider


def test_build_messages_simple() -> None:
    provider = NvidiaLLMProvider(api_key="test")
    messages = provider.build_messages(
        system_prompt="You are a helpful assistant.",
        conversation=[],
        context=["Context 1", "Context 2"],
        question="What is this?",
    )
    assert len(messages) == 3
    assert isinstance(messages[0], SystemMessage)
    assert "Context" in messages[1].content
    assert isinstance(messages[2], HumanMessage)


def test_build_messages_with_session_history() -> None:
    provider = NvidiaLLMProvider(api_key="test")
    messages = provider.build_messages(
        system_prompt="Be helpful.",
        conversation=[],
        context=[],
        question="What is X?",
        session_history=[
            {"role": "user", "content": "What is RAG?"},
            {"role": "assistant", "content": "RAG is Retrieval Augmented Generation."},
            {"role": "user", "content": "What are its benefits?"},
            {"role": "assistant", "content": "It improves accuracy and reduces hallucinations."},
        ],
    )
    assert len(messages) == 6
    assert isinstance(messages[0], SystemMessage)
    assert isinstance(messages[1], HumanMessage)
    assert messages[1].content == "What is RAG?"
    assert isinstance(messages[2], AIMessage)
    assert messages[2].content == "RAG is Retrieval Augmented Generation."
    assert isinstance(messages[3], HumanMessage)
    assert messages[3].content == "What are its benefits?"
    assert isinstance(messages[4], AIMessage)
    assert isinstance(messages[5], HumanMessage)
    assert messages[5].content == "What is X?"


def test_build_messages_with_images() -> None:
    provider = NvidiaLLMProvider(api_key="test")
    messages = provider.build_messages(
        system_prompt="Be helpful.",
        conversation=[],
        context=[],
        question="What is in this image?",
        image_paths=[],
    )
    assert len(messages) == 2
    assert isinstance(messages[0], SystemMessage)
    assert isinstance(messages[1], HumanMessage)
