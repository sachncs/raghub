"""Prompt-builder and token-counter tests.

Covers :class:`raghub.prompts.builder.PromptBuilder`'s token-budget
allocator and the tiktoken-based :class:`TokenCounter` (plus its
whitespace fallback when tiktoken is unavailable).
"""

from __future__ import annotations

from raghub.models import ConversationTurn
from raghub.prompts.builder import PromptBuilder, PromptConfig, TokenCounter


class TestTokenCounter:
    def test_count_tokens_tiktoken(self):
        counter = TokenCounter()
        count = counter.count("Hello, world!")
        assert count > 0

    def test_truncate_short_text(self):
        counter = TokenCounter()
        text = "Hello world"
        truncated = counter.truncate(text, 100)
        assert truncated == text

    def test_truncate_long_text(self):
        counter = TokenCounter()
        text = "word " * 100
        truncated = counter.truncate(text, 10)
        assert len(truncated.split()) <= 15

    def test_fallback_without_tiktoken(self):
        counter = TokenCounter()
        counter.enc = None
        text = "one two three four five"
        assert counter.count(text) == 5


class TestPromptBuilder:
    def test_build_minimal(self):
        builder = PromptBuilder()
        result = builder.build_messages(question="What is RAG?")
        assert result["question"] == "What is RAG?"
        assert result["system"] == PromptConfig().system_prompt
        assert len(result["history"]) == 0
        assert len(result["context"]) == 0

    def test_build_with_context(self):
        builder = PromptBuilder()
        result = builder.build_messages(
            question="Summarize",
            context=[{"text": "Doc1 content"}, {"text": "Doc2 content"}],
        )
        assert len(result["context"]) == 2
        assert "Doc1 content" in result["context"]

    def test_build_with_history(self):
        builder = PromptBuilder()
        history = [
            ConversationTurn(question="Hi", answer="Hello!"),
            ConversationTurn(question="How are you?", answer="I'm fine."),
        ]
        result = builder.build_messages(
            question="What was my first question?",
            session_history=history,
        )
        assert len(result["history"]) > 0
        assert result["history"][0] == {"role": "user", "content": "Hi"}

    def test_build_with_image_paths(self):
        builder = PromptBuilder()
        result = builder.build_messages(
            question="Describe this image",
            image_paths=["/path/to/img.png"],
        )
        assert "/path/to/img.png" in result["image_paths"]

    def test_truncation_removes_old_context(self):
        builder = PromptBuilder(PromptConfig(max_tokens=100, reserved_output_tokens=20))
        long_context = [{"text": "word " * 200} for _ in range(10)]
        result = builder.build_messages(
            question="Hi",
            context=long_context,
        )
        assert len(result["context"]) < 10


class TestSystemPromptTemplate:
    def test_template_exported(self):
        from raghub.prompts import SYSTEM_PROMPT_TEMPLATE

        assert isinstance(SYSTEM_PROMPT_TEMPLATE, str)
        assert "retrieval-augmented" in SYSTEM_PROMPT_TEMPLATE
        assert "documents" in SYSTEM_PROMPT_TEMPLATE
