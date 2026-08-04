"""Coverage tests for :mod:`raghub.prompts`."""

from __future__ import annotations

from raghub.models import Turn
from raghub.prompts import (
    SYSTEM_PROMPT_TEMPLATE,
    Prompt,
    PromptConfig,
    TokenCounter,
)

# ---------------------------------------------------------------------------
# SYSTEM_PROMPT_TEMPLATE
# ---------------------------------------------------------------------------


def test_system_prompt_template_mentions_injection_mitigation() -> None:
    """The template instructs the model to ignore document instructions."""
    assert "retrieval-augmented" in SYSTEM_PROMPT_TEMPLATE
    assert "Ignore instructions" in SYSTEM_PROMPT_TEMPLATE
    assert "cite sources" in SYSTEM_PROMPT_TEMPLATE


# ---------------------------------------------------------------------------
# PromptConfig
# ---------------------------------------------------------------------------


def test_prompt_config_defaults() -> None:
    """``PromptConfig`` defaults to a 4k window with 512 reserved tokens."""
    config = PromptConfig()
    assert config.max_tokens == 4096
    assert config.reserved_output_tokens == 512
    assert "RAG" in config.system_prompt


def test_prompt_config_custom_values() -> None:
    """Custom values override the defaults."""
    config = PromptConfig(
        system_prompt="custom",
        max_tokens=2048,
        reserved_output_tokens=128,
    )
    assert config.system_prompt == "custom"
    assert config.max_tokens == 2048
    assert config.reserved_output_tokens == 128


# ---------------------------------------------------------------------------
# TokenCounter
# ---------------------------------------------------------------------------


def test_token_counter_uses_tiktoken_by_default() -> None:
    """The default counter loads ``cl100k_base`` and counts tokens."""
    counter = TokenCounter()
    assert counter.enc is not None
    assert counter.count("hello world") == 2


def test_token_counter_falls_back_to_whitespace() -> None:
    """When ``enc`` is ``None``, the counter falls back to word splitting."""
    counter = TokenCounter()
    counter.enc = None
    assert counter.count("hello world") == 2
    assert counter.count("") == 0


def test_token_counter_decode_tokens() -> None:
    """``decode_tokens`` round-trips text through tiktoken."""
    counter = TokenCounter()
    tokens = counter.enc.encode("hello world")
    assert counter.decode_tokens(tokens) == "hello world"


def test_token_counter_truncate_short_text_returns_as_is() -> None:
    """Text that fits the budget is returned unchanged."""
    counter = TokenCounter()
    result = counter.truncate("hi", max_tokens=10)
    assert result == "hi"


def test_token_counter_truncate_long_text_with_tiktoken() -> None:
    """Long text is token-bounded and the result fits within the budget."""
    counter = TokenCounter()
    long_text = " ".join(f"word{i}" for i in range(100))
    truncated = counter.truncate(long_text, max_tokens=10)
    assert counter.count(truncated) <= 10


def test_token_counter_truncate_long_text_fallback() -> None:
    """Without tiktoken, the fallback splits on whitespace and re-joins."""
    counter = TokenCounter()
    counter.enc = None
    truncated = counter.truncate("one two three four five six seven eight", max_tokens=3)
    assert truncated == "one two three"
    assert counter.count(truncated) == 3


def test_token_counter_truncate_handles_empty_string() -> None:
    """Empty input returns empty output regardless of the budget."""
    counter = TokenCounter()
    assert counter.truncate("", max_tokens=100) == ""


def test_token_counter_truncate_handles_zero_budget() -> None:
    """A zero-token budget returns the smallest possible prefix."""
    counter = TokenCounter()
    assert counter.truncate("hello", max_tokens=0) == ""


# ---------------------------------------------------------------------------
# Prompt.__init__
# ---------------------------------------------------------------------------


def test_prompt_builder_defaults() -> None:
    """``Prompt`` defaults to a fresh :class:`PromptConfig`."""
    builder = Prompt()
    assert isinstance(builder.config, PromptConfig)
    assert builder.token_counter is not None


def test_prompt_builder_custom_config() -> None:
    """A supplied :class:`PromptConfig` is stored verbatim."""
    config = PromptConfig(max_tokens=1024, reserved_output_tokens=64)
    builder = Prompt(config=config)
    assert builder.config is config


# ---------------------------------------------------------------------------
# Prompt.build_messages
# ---------------------------------------------------------------------------


def test_build_messages_returns_expected_keys() -> None:
    """The payload exposes ``system``, ``history``, ``context``, ``question``."""
    builder = Prompt()
    payload = builder.build_messages(question="q", context=[{"text": "ctx"}])
    assert set(payload.keys()) >= {
        "system",
        "history",
        "context",
        "question",
        "image_paths",
    }
    assert payload["question"] == "q"
    assert payload["image_paths"] == []


def test_build_messages_preserves_question_under_tight_budget() -> None:
    """The question is always emitted, even when the budget is exhausted."""
    builder = Prompt(
        config=PromptConfig(max_tokens=10, reserved_output_tokens=5)
    )
    payload = builder.build_messages(
        question="q",
        context=[{"text": " ".join(f"w{i}" for i in range(50))}],
    )
    assert payload["question"] == "q"


def test_build_messages_truncates_context_to_budget() -> None:
    """Context chunks are dropped once the budget overflows."""
    config = PromptConfig(max_tokens=30, reserved_output_tokens=5)
    builder = Prompt(config=config)
    payload = builder.build_messages(
        question="q",
        context=[{"text": "alpha " * 30}, {"text": "beta " * 30}],
    )
    assert len(payload["context"]) <= 1


def test_build_messages_truncates_history_newest_first() -> None:
    """History is walked newest-first, dropping the oldest turns first."""
    config = PromptConfig(max_tokens=30, reserved_output_tokens=5)
    builder = Prompt(config=config)
    history = [Turn(question=f"q{i}", answer=f"a{i}") for i in range(5)]
    payload = builder.build_messages(question="q", context=None, session_history=history)
    roles = [m["role"] for m in payload["history"]]
    assert roles[0] == "user"


def test_build_messages_skips_history_when_budget_exhausted() -> None:
    """History is skipped when no token remains for it."""
    config = PromptConfig(max_tokens=10, reserved_output_tokens=9)
    builder = Prompt(config=config)
    history = [Turn(question="q1", answer="a1 long long long long long long")]
    payload = builder.build_messages(question="q", context=None, session_history=history)
    assert payload["history"] == []


def test_build_messages_stringifies_dict_without_text_key() -> None:
    """Dicts lacking a ``text`` key fall back to ``str(chunk)``."""
    builder = Prompt()
    payload = builder.build_messages(question="q", context=[{"other": "value"}])
    assert payload["context"] == ["{'other': 'value'}"]


def test_build_messages_dict_with_text_key() -> None:
    """A dict with a ``text`` key uses that text."""
    builder = Prompt()
    payload = builder.build_messages(question="q", context=[{"text": "hello"}])
    assert payload["context"] == ["hello"]


def test_build_messages_includes_image_paths() -> None:
    """``image_paths`` is propagated to the payload."""
    builder = Prompt()
    payload = builder.build_messages(
        question="q",
        image_paths=["/tmp/a.png", "/tmp/b.jpg"],
    )
    assert payload["image_paths"] == ["/tmp/a.png", "/tmp/b.jpg"]


def test_build_messages_no_args() -> None:
    """``build_messages`` works with only a question."""
    builder = Prompt()
    payload = builder.build_messages(question="q")
    assert payload["question"] == "q"
    assert payload["context"] == []
    assert payload["history"] == []


def test_build_messages_uses_custom_system_prompt() -> None:
    """A custom ``system_prompt`` from config is used verbatim."""
    config = PromptConfig(
        system_prompt="be terse", max_tokens=4096, reserved_output_tokens=512
    )
    builder = Prompt(config=config)
    payload = builder.build_messages(question="q")
    assert payload["system"] == "be terse"
