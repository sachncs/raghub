from __future__ import annotations

from raghub.prompts.builder import PromptBuilder, PromptConfig, TokenCounter, TemplatePromptBuilder
from raghub.models import ConversationTurn


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


class TestTemplatePromptBuilder:
    def test_build_messages(self):
        from raghub.models import ConversationTurn
        from raghub.documents.chunker import ChunkRecord
        from raghub.models import Classification
        builder = TemplatePromptBuilder()
        turn = ConversationTurn(question="Hi", answer="Hello")
        chunk = ChunkRecord(
            chunk_id="c1",
            document_id="d1",
            version=1,
            page=1,
            company="acme",
            owner="alice",
            classification=Classification.INTERNAL,
            hash="abc",
            text="Some context",
        )
        result = builder.build_messages(
            conversation=[turn],
            retrieved_chunks=[chunk],
            question="What?",
        )
        assert len(result) >= 2
        assert result[-1]["role"] == "user"
        assert result[-1]["content"] == "What?"
