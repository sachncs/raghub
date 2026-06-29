"""Tests for the LiteLLM-backed LLM provider.

Exercises :meth:`LiteLLMProvider.build_messages` against the
standard set of inputs (system prompt, retrieved context,
session history, images). The provider is instantiated with a
placeholder API key; no actual inference occurs in these tests.
"""

from __future__ import annotations

import pytest

from raghub.llm.litellm import LiteLLMProvider

litellm = pytest.importorskip("litellm")


def test_build_messages_simple() -> None:
    provider = LiteLLMProvider(api_key="test")
    messages = provider.build_messages(
        system_prompt="You are a helpful assistant.",
        conversation=[],
        context=["Context 1", "Context 2"],
        question="What is this?",
    )
    assert len(messages) == 3
    assert messages[0] == {"role": "system", "content": "You are a helpful assistant."}
    assert messages[1]["role"] == "system"
    assert "Context" in messages[1]["content"]
    assert messages[2] == {"role": "user", "content": "What is this?"}


def test_build_messages_with_session_history() -> None:
    provider = LiteLLMProvider(api_key="test")
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
    assert messages[0] == {"role": "system", "content": "Be helpful."}
    assert messages[1] == {"role": "user", "content": "What is RAG?"}
    assert messages[2] == {
        "role": "assistant",
        "content": "RAG is Retrieval Augmented Generation.",
    }
    assert messages[3] == {"role": "user", "content": "What are its benefits?"}
    assert messages[4] == {
        "role": "assistant",
        "content": "It improves accuracy and reduces hallucinations.",
    }
    assert messages[5] == {"role": "user", "content": "What is X?"}


def test_build_messages_with_images() -> None:
    provider = LiteLLMProvider(api_key="test")
    messages = provider.build_messages(
        system_prompt="Be helpful.",
        conversation=[],
        context=[],
        question="What is in this image?",
        image_paths=[],
    )
    assert len(messages) == 2
    assert messages[0] == {"role": "system", "content": "Be helpful."}
    assert messages[1] == {"role": "user", "content": "What is in this image?"}


def test_build_messages_with_image_paths_renders_content_array() -> None:
    """A user message with images becomes a content array."""
    import os
    import tempfile

    provider = LiteLLMProvider(api_key="test")
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
        tmp.write(b"\x89PNG\r\n\x1a\nfakebytes")
        tmp_path = tmp.name
    try:
        messages = provider.build_messages(
            system_prompt="Be helpful.",
            conversation=[],
            context=[],
            question="What is in this image?",
            image_paths=[tmp_path],
        )
    finally:
        os.unlink(tmp_path)

    assert len(messages) == 2
    user = messages[1]
    assert user["role"] == "user"
    content = user["content"]
    assert isinstance(content, list)
    assert content[0] == {"type": "text", "text": "What is in this image?"}
    assert content[1]["type"] == "image_url"
    assert content[1]["image_url"]["url"].startswith("data:image/png;base64,")

