"""Comprehensive tests for the LLM modules.

Covers:

* :func:`any_llm_api_key_present` — env-var detection.
* :func:`build_llm` — factory selection by model name + credentials.
* :class:`HeuristicProvider` — deterministic offline contract.
* :class:`LiteLLM` — ``build_messages`` edge cases and error wrapping.
"""

from __future__ import annotations

import os
import tempfile
import types
from unittest.mock import MagicMock

import pytest

from raghub.errors import GenerationError
from raghub.llm import (
    LLM_API_KEY_ENV_VARS,
    HeuristicProvider,
    LiteLLM,
    any_llm_api_key_present,
    build_llm,
)

# ---------------------------------------------------------------------------
# any_llm_api_key_present
# ---------------------------------------------------------------------------


class TestAnyLlmApiKeyPresent:
    """any_llm_api_key_present() detects configured provider credentials."""

    def test_returns_true_when_env_var_is_set(self, monkeypatch: pytest.MonkeyPatch) -> None:
        for var in LLM_API_KEY_ENV_VARS:
            monkeypatch.delenv(var, raising=False)
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
        assert any_llm_api_key_present() is True

    def test_returns_false_when_no_env_vars_are_set(self, monkeypatch: pytest.MonkeyPatch) -> None:
        for var in LLM_API_KEY_ENV_VARS:
            monkeypatch.delenv(var, raising=False)
        assert any_llm_api_key_present() is False

    def test_respects_aws_access_key_id(self, monkeypatch: pytest.MonkeyPatch) -> None:
        for var in LLM_API_KEY_ENV_VARS:
            monkeypatch.delenv(var, raising=False)
        monkeypatch.setenv("AWS_ACCESS_KEY_ID", "AKIA-test")
        assert any_llm_api_key_present() is True


# ---------------------------------------------------------------------------
# build_llm
# ---------------------------------------------------------------------------


class TestBuildLlm:
    """build_llm() selects the right provider based on name and env."""

    def test_empty_string_returns_heuristic(self, monkeypatch: pytest.MonkeyPatch) -> None:
        for var in LLM_API_KEY_ENV_VARS:
            monkeypatch.delenv(var, raising=False)
        provider = build_llm("")
        assert isinstance(provider, HeuristicProvider)
        assert provider.model_name == "heuristic"

    def test_no_api_key_falls_back_to_heuristic(self, monkeypatch: pytest.MonkeyPatch) -> None:
        for var in LLM_API_KEY_ENV_VARS:
            monkeypatch.delenv(var, raising=False)
        provider = build_llm("gpt-4o")
        assert isinstance(provider, HeuristicProvider)

    def test_api_key_arg_returns_litellm(self, monkeypatch: pytest.MonkeyPatch) -> None:
        for var in LLM_API_KEY_ENV_VARS:
            monkeypatch.delenv(var, raising=False)
        provider = build_llm("gpt-4o", api_key="sk-test")
        assert isinstance(provider, LiteLLM)
        assert provider.model_name == "gpt-4o"
        assert provider.api_key == "sk-test"

    def test_env_key_returns_litellm(self, monkeypatch: pytest.MonkeyPatch) -> None:
        for var in LLM_API_KEY_ENV_VARS:
            monkeypatch.delenv(var, raising=False)
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
        provider = build_llm("gpt-4o")
        assert isinstance(provider, LiteLLM)


# ---------------------------------------------------------------------------
# HeuristicProvider — determinism and contract
# ---------------------------------------------------------------------------


class TestHeuristicProviderDeterminism:
    def test_same_inputs_same_output(self) -> None:
        """The heuristic provider is documented as deterministic — it
        picks a sentence based on the question words and ignores any
        randomness. A regression that introduced non-determinism
        would break the production test suite."""
        p = HeuristicProvider()
        kwargs = dict(
            system_prompt="sys",
            conversation=[],
            context=["alpha beta gamma"],
            question="q",
        )
        assert p.generate(**kwargs) == p.generate(**kwargs)

    def test_empty_context_returns_sentinel(self) -> None:
        """An empty context returns the documented 'no chunks' string
        so callers can detect the no-information case."""
        p = HeuristicProvider()
        out = p.generate(system_prompt="s", conversation=[], context=[], question="q")
        assert "no" in out.lower(), (
            "The empty-context sentinel is the contract callers depend "
            "on to detect 'I have no information'."
        )

    def test_picks_relevant_sentence(self) -> None:
        """When the question contains a word that appears in a context
        sentence, that sentence wins."""
        p = HeuristicProvider()
        out = p.generate(
            system_prompt="s",
            conversation=[],
            context=["apple pie is delicious", "banana split is cold"],
            question="apple",
        )
        assert "apple" in out.lower()

    def test_ignores_question_and_history(self) -> None:
        """The heuristic does not consult the conversation history.
        Different system prompts/questions with the same context
        yield the same answer when no question words match."""
        p = HeuristicProvider()
        ctx = ["alpha beta"]
        a = p.generate(system_prompt="s1", conversation=[], context=ctx, question="q1")
        b = p.generate(system_prompt="s2", conversation=[], context=ctx, question="q2")
        # Neither contains words from ctx → both fall through to first chunk.
        assert a == b


# ---------------------------------------------------------------------------
# LiteLLM — require_litellm + error wrapping
# ---------------------------------------------------------------------------


class TestRequireLitellm:
    def test_raises_when_litellm_not_available(self) -> None:
        import raghub.llm as litellm_mod

        saved = litellm_mod.LITELLM_AVAILABLE
        try:
            litellm_mod.LITELLM_AVAILABLE = False
            provider = litellm_mod.LiteLLM.__new__(litellm_mod.LiteLLM)
            provider.model_name = "m"
            with pytest.raises(Exception, match="litellm is not installed"):
                provider.require_litellm()
        finally:
            litellm_mod.LITELLM_AVAILABLE = saved

    def test_passes_when_litellm_is_available(self) -> None:
        provider = LiteLLM(api_key="test")
        provider.require_litellm()


class TestGenerateErrorHandling:
    """generate() wraps provider exceptions as GenerationError."""

    def test_litellm_error_is_wrapped(self) -> None:
        import raghub.llm as litellm_mod

        saved = litellm_mod.litellm
        try:
            litellm_mod.litellm = types.ModuleType("litellm")
            litellm_mod.litellm.completion = MagicMock(side_effect=ValueError("API down"))
            litellm_mod.LITELLM_AVAILABLE = True

            provider = litellm_mod.LiteLLM(model="m")
            with pytest.raises(GenerationError, match="completion failed"):
                provider.generate(
                    system_prompt="s",
                    conversation=[],
                    context=[],
                    question="q",
                )
        finally:
            litellm_mod.litellm = saved


# ---------------------------------------------------------------------------
# build_messages — edge cases
# ---------------------------------------------------------------------------


class TestBuildMessagesContext:
    def test_context_is_formatted_and_appended(self) -> None:
        provider = LiteLLM(api_key="test")
        messages = provider.build_messages(
            system_prompt="sys",
            question="q",
            context=["chunk A", "chunk B", "chunk C"],
        )
        assert messages[0] == {"role": "system", "content": "sys"}
        assert messages[-2] == {
            "role": "system",
            "content": "Context:\nchunk A\n\n---\n\nchunk B\n\n---\n\nchunk C",
        }

    def test_context_empty_omits_context_message(self) -> None:
        provider = LiteLLM(api_key="test")
        messages = provider.build_messages(
            system_prompt="sys",
            question="q",
            context=[],
        )
        assert not any(
            m["role"] == "system" and str(m["content"]).startswith("Context:")
            for m in messages
        )

    def test_single_context_item(self) -> None:
        provider = LiteLLM(api_key="test")
        messages = provider.build_messages(
            system_prompt="sys",
            question="q",
            context=["only one"],
        )
        assert any("Context:\nonly one" in str(m["content"]) for m in messages)

    def test_session_history_invalid_role_falls_back_to_user(self) -> None:
        provider = LiteLLM(api_key="test")
        messages = provider.build_messages(
            system_prompt="sys",
            question="q",
            session_history=[{"role": "moderator", "content": "please behave"}],
        )
        # 1 system + 1 history (user) + 1 final user = 3
        assert len(messages) == 3
        assert messages[1] == {"role": "user", "content": "please behave"}

    def test_many_session_history_messages_all_included(self) -> None:
        provider = LiteLLM(api_key="test")
        history = [{"role": "user", "content": f"q{i}"} for i in range(20)]
        messages = provider.build_messages(
            system_prompt="sys",
            question="q_final",
            session_history=history,
        )
        assert len(messages) == 22
        assert messages[-1]["content"] == "q_final"


class TestBuildMessagesImages:
    def test_image_unknown_extension_defaults_to_png(self) -> None:
        provider = LiteLLM(api_key="test")
        with tempfile.NamedTemporaryFile(suffix=".unknownxyz", delete=False) as tmp:
            tmp.write(b"fake image bytes")
            path = tmp.name
        try:
            messages = provider.build_messages(
                system_prompt="sys",
                question="what is this?",
                image_paths=[path],
            )
        finally:
            os.unlink(path)

        content = messages[-1]["content"]
        assert isinstance(content, list)
        image_entries = [c for c in content if c.get("type") == "image_url"]
        assert image_entries, "expected at least one image_url entry"
        assert image_entries[0]["image_url"]["url"].startswith("data:image/png;base64,")

    def test_image_with_no_mime_type_still_works(self) -> None:
        provider = LiteLLM(api_key="test")
        with tempfile.NamedTemporaryFile(suffix="", delete=False) as tmp:
            tmp.write(b"test")
            path = tmp.name
        try:
            messages = provider.build_messages(
                system_prompt="sys",
                question="what?",
                image_paths=[path],
            )
        finally:
            os.unlink(path)
        assert messages[-1]["role"] == "user"


class TestBuildMessagesEdgeCases:
    def test_empty_question(self) -> None:
        provider = LiteLLM(api_key="test")
        messages = provider.build_messages(system_prompt="sys", question="")
        assert messages[-1] == {"role": "user", "content": ""}

    def test_context_with_special_characters(self) -> None:
        provider = LiteLLM(api_key="test")
        messages = provider.build_messages(
            system_prompt="sys",
            question="q",
            context=["multi\nline\ntext", 'with "quotes"', "日本語 unicode"],
        )
        assert any("multi" in str(m["content"]) for m in messages)
        assert any("日本語" in str(m["content"]) for m in messages)


# ---------------------------------------------------------------------------
# LiteLLM construction
# ---------------------------------------------------------------------------


class TestLiteLLMConstruction:
    def test_default_model_name(self) -> None:
        provider = LiteLLM(api_key="test")
        assert provider.model_name == "gpt-4o-mini"
        assert provider.temperature == 0.2
        assert provider.api_key == "test"
        assert provider.api_base is None

    def test_model_name_passed_through(self) -> None:
        provider = LiteLLM(model="claude-3-5-sonnet")
        assert provider.model_name == "claude-3-5-sonnet"

    def test_raises_configuration_error_when_litellm_not_installed(self) -> None:
        import raghub.llm as litellm_mod
        from raghub.errors import ConfigurationError

        saved_available = litellm_mod.LITELLM_AVAILABLE
        saved_litellm = litellm_mod.litellm
        try:
            litellm_mod.litellm = None
            litellm_mod.LITELLM_AVAILABLE = False
            with pytest.raises(ConfigurationError, match="litellm is not installed"):
                litellm_mod.LiteLLM(model="m")
        finally:
            litellm_mod.litellm = saved_litellm
            litellm_mod.LITELLM_AVAILABLE = saved_available
