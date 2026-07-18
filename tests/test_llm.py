"""Comprehensive tests for the LLM modules.

Covers uncovered lines in:
- ``raghub/llm/__init__.py``  (lines 55, 60--64, 92--99)
- ``raghub/llm/litellm.py``   (lines 82, 127, 141, 190--191,
  215--216, 261--262, 269--289, 292)
"""

from __future__ import annotations

import os
import tempfile
import types
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from raghub.llm import (
    LLM_API_KEY_ENV_VARS,
    HeuristicLLMProvider,
    any_llm_api_key_present,
    build_llm_provider,
)

# ---------------------------------------------------------------------------
# raghub.llm.__init__  tests
# ---------------------------------------------------------------------------


class TestAnyLlmApiKeyPresent:
    """Cover line 55 of __init__.py — any_llm_api_key_present()."""

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


class TestGetAttr:
    """Cover lines 60–64 of __init__.py — module __getattr__."""

    def test_lazy_loads_litellm_provider(self) -> None:
        from raghub.llm import LiteLLMProvider as LazyLiteLLM
        from raghub.llm.litellm import LiteLLMProvider as DirectLiteLLM

        assert LazyLiteLLM is DirectLiteLLM

    def test_raises_attribute_error_for_unknown_name(self) -> None:
        import raghub.llm as llm_mod

        with pytest.raises(AttributeError, match="has no attribute 'Nonsense'"):
            llm_mod.__getattr__("Nonsense")


class TestBuildLlmProvider:
    """Cover lines 92–99 of __init__.py — build_llm_provider()."""

    def test_empty_string_returns_heuristic(self) -> None:
        provider = build_llm_provider("")
        assert isinstance(provider, HeuristicLLMProvider)
        assert provider.model_name == "heuristic-llm"

    def test_heuristic_name_returns_heuristic(self) -> None:
        provider = build_llm_provider("heuristic")
        assert isinstance(provider, HeuristicLLMProvider)
        assert provider.model_name == "heuristic"

    def test_heuristic_dash_llm_returns_heuristic(self) -> None:
        provider = build_llm_provider("heuristic-llm")
        assert isinstance(provider, HeuristicLLMProvider)
        assert provider.model_name == "heuristic-llm"

    def test_no_api_key_falls_back_to_heuristic(self, monkeypatch: pytest.MonkeyPatch) -> None:
        for var in LLM_API_KEY_ENV_VARS:
            monkeypatch.delenv(var, raising=False)
        provider = build_llm_provider("gpt-4o")
        assert isinstance(provider, HeuristicLLMProvider)
        assert provider.model_name == "gpt-4o"

    def test_api_key_arg_returns_litellm(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Passing api_key explicitly bypasses env-var check."""
        import raghub.llm.litellm as litellm_mod

        for var in LLM_API_KEY_ENV_VARS:
            monkeypatch.delenv(var, raising=False)

        with patch.object(litellm_mod, "LiteLLMProvider") as mock_cls:
            build_llm_provider("gpt-4o", api_key="sk-test")
        mock_cls.assert_called_once_with(model="gpt-4o", api_key="sk-test")

    def test_env_key_returns_litellm(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Having an API key in the env is sufficient for LiteLLMProvider."""
        import raghub.llm.litellm as litellm_mod

        for var in LLM_API_KEY_ENV_VARS:
            monkeypatch.delenv(var, raising=False)
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")

        with patch.object(litellm_mod, "LiteLLMProvider") as mock_cls:
            build_llm_provider("gpt-4o")
        mock_cls.assert_called_once_with(model="gpt-4o", api_key=None)

    def test_model_name_is_stripped_and_lowered(self, monkeypatch: pytest.MonkeyPatch) -> None:
        for var in LLM_API_KEY_ENV_VARS:
            monkeypatch.delenv(var, raising=False)
        provider = build_llm_provider("  HEURISTIC  ")
        assert isinstance(provider, HeuristicLLMProvider)

    def test_litellm_is_imported_lazily(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """The ``from .litellm import LiteLLMProvider`` inside
        ``build_llm_provider`` is tested by ensuring it is reachable."""
        import raghub.llm.litellm as litellm_mod

        for var in LLM_API_KEY_ENV_VARS:
            monkeypatch.delenv(var, raising=False)
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")

        with patch.object(litellm_mod, "LiteLLMProvider") as mock_cls:
            build_llm_provider("claude-3")
        mock_cls.assert_called_once_with(model="claude-3", api_key=None)


# ---------------------------------------------------------------------------
# raghub.llm.litellm  tests
# ---------------------------------------------------------------------------


class TestRequireLitellm:
    """Cover line 82 of litellm.py — require_litellm()."""

    def test_raises_when_litellm_not_available(self) -> None:
        import raghub.llm.litellm as litellm_mod

        saved = litellm_mod.LITELLM_AVAILABLE
        try:
            litellm_mod.LITELLM_AVAILABLE = False
            provider = litellm_mod.LiteLLMProvider.__new__(litellm_mod.LiteLLMProvider)
            provider.model_name = "m"
            with pytest.raises(Exception, match="litellm is not installed"):
                provider.require_litellm()
        finally:
            litellm_mod.LITELLM_AVAILABLE = saved

    def test_passes_when_litellm_is_available(self) -> None:
        from raghub.llm.litellm import LiteLLMProvider

        provider = LiteLLMProvider(api_key="test")
        provider.require_litellm()


class TestBuildMessages:
    """Cover lines 127 and 141 of litellm.py — edge cases in build_messages."""

    def test_session_history_invalid_role_falls_back_to_user(self) -> None:
        """Cover line 127: role not in {user, assistant, system} → 'user'."""
        from raghub.llm.litellm import LiteLLMProvider

        provider = LiteLLMProvider(api_key="test")
        messages = provider.build_messages(
            system_prompt="sys",
            question="q",
            session_history=[
                {"role": "moderator", "content": "please behave"},
            ],
        )
        assert len(messages) == 3
        assert messages[1] == {"role": "user", "content": "please behave"}
        assert messages[2] == {"role": "user", "content": "q"}

    def test_image_unknown_extension_defaults_to_png(self) -> None:
        """Cover line 141: mimetypes.guess_type returns None → 'image/png'."""
        from raghub.llm.litellm import LiteLLMProvider

        provider = LiteLLMProvider(api_key="test")
        # Use a temp file with no recognizable extension
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

        assert len(messages) == 2
        content = messages[1]["content"]
        assert isinstance(content, list)
        assert content[1]["type"] == "image_url"
        assert content[1]["image_url"]["url"].startswith("data:image/png;base64,")

    def test_image_with_no_mime_type_still_works(self) -> None:
        """Cover line 141 when mime_type is None."""
        from raghub.llm.litellm import LiteLLMProvider

        provider = LiteLLMProvider(api_key="test")
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

        assert len(messages) == 2


class TestGenerateErrorHandling:
    """Cover lines 190–191 of litellm.py — generate() error wrapping."""

    def test_litellm_error_is_wrapped(self) -> None:
        import raghub.llm.litellm as litellm_mod

        saved = litellm_mod.litellm
        try:
            litellm_mod.litellm = types.ModuleType("litellm")
            litellm_mod.litellm.completion = MagicMock(side_effect=ValueError("API down"))
            litellm_mod.LITELLM_AVAILABLE = True

            provider = litellm_mod.LiteLLMProvider(model="m")
            with pytest.raises(Exception, match="LiteLLM completion failed: API down") as exc_info:
                provider.generate(system_prompt="s", question="q")
            from raghub.exceptions import LLMError

            assert isinstance(exc_info.value, LLMError)
        finally:
            litellm_mod.litellm = saved


class TestRecordUsage:
    """Cover lines 215–216 of litellm.py — record_usage edge cases."""

    def test_dict_with_input_output_tokens(self) -> None:
        """When 'input_tokens' and 'output_tokens' are used instead of
        'prompt_tokens' / 'completion_tokens'."""
        import raghub.llm.litellm as litellm_mod

        provider = litellm_mod.LiteLLMProvider.__new__(litellm_mod.LiteLLMProvider)
        provider.model_name = "gpt-4"
        provider.last_usage = None

        response = {"usage": {"input_tokens": 9, "output_tokens": 99}}
        provider.record_usage(response)
        assert provider.last_usage == {"prompt": 9, "completion": 99, "model": "gpt-4"}

    def test_dict_with_completion_tokens_fallback(self) -> None:
        """'completion_tokens' takes priority over 'output_tokens'."""
        import raghub.llm.litellm as litellm_mod

        provider = litellm_mod.LiteLLMProvider.__new__(litellm_mod.LiteLLMProvider)
        provider.model_name = "gpt-4"
        provider.last_usage = None

        response = {"usage": {"prompt_tokens": 5, "completion_tokens": 10, "output_tokens": 999}}
        provider.record_usage(response)
        assert provider.last_usage == {"prompt": 5, "completion": 10, "model": "gpt-4"}

    def test_record_usage_none_usage_returns_early(self) -> None:
        """When the response has no 'usage' field, last_usage stays None."""
        import raghub.llm.litellm as litellm_mod

        provider = litellm_mod.LiteLLMProvider.__new__(litellm_mod.LiteLLMProvider)
        provider.model_name = "m"
        provider.last_usage = None

        provider.record_usage({"choices": []})
        assert provider.last_usage is None

    def test_record_usage_object_style(self) -> None:
        """Object-style usage via getattr."""
        import raghub.llm.litellm as litellm_mod

        provider = litellm_mod.LiteLLMProvider.__new__(litellm_mod.LiteLLMProvider)
        provider.model_name = "m"
        provider.last_usage = None

        class FakeUsage:
            prompt_tokens = 3
            completion_tokens = 7

        class FakeResponse:
            usage = FakeUsage()

        provider.record_usage(FakeResponse())
        assert provider.last_usage == {"prompt": 3, "completion": 7, "model": "m"}

    def test_record_usage_object_missing_tokens(self) -> None:
        """When getattr returns 0 for prompt_tokens/completion_tokens."""
        import raghub.llm.litellm as litellm_mod

        provider = litellm_mod.LiteLLMProvider.__new__(litellm_mod.LiteLLMProvider)
        provider.model_name = "m"
        provider.last_usage = None

        class FakeUsage:
            pass

        class FakeResponse:
            usage = FakeUsage()

        provider.record_usage(FakeResponse())
        assert provider.last_usage == {"prompt": 0, "completion": 0, "model": "m"}


class TestAstreamErrorHandling:
    """Cover lines 261–262 of litellm.py — astream() error wrapping."""

    @pytest.mark.asyncio
    async def test_litellm_error_is_wrapped(self) -> None:
        import raghub.llm.litellm as litellm_mod

        saved = litellm_mod.litellm
        try:
            litellm_mod.litellm = types.ModuleType("litellm")
            litellm_mod.litellm.acompletion = AsyncMock(side_effect=ValueError("stream failed"))
            litellm_mod.LITELLM_AVAILABLE = True

            provider = litellm_mod.LiteLLMProvider(model="m")
            gen = provider.astream(system_prompt="s", question="q")
            with pytest.raises(
                Exception, match="LiteLLM streaming failed: stream failed"
            ) as exc_info:
                async for _ in gen:
                    pass
            from raghub.exceptions import LLMError

            assert isinstance(exc_info.value, LLMError)
        finally:
            litellm_mod.litellm = saved


class TestAstreamChunks:
    """Cover lines 269–289 and 292 of litellm.py — astream chunk processing."""

    @staticmethod
    def _make_async_iter(items):
        """Return an async-iterable from a list."""

        class _AsyncIter:
            def __init__(self, items):
                self._items = list(items)

            def __aiter__(self):
                return self

            async def __anext__(self):
                if not self._items:
                    raise StopAsyncIteration
                return self._items.pop(0)

        return _AsyncIter(items)

    @pytest.mark.asyncio
    async def test_dict_chunks_with_content_and_usage(self) -> None:
        """Cover dict-path chunk processing (lines 269–275, 281–289, 292)."""
        import raghub.llm.litellm as litellm_mod

        saved = litellm_mod.litellm
        try:
            litellm_mod.litellm = types.ModuleType("litellm")
            chunks = [
                {"choices": [{"delta": {"content": "Hello"}}]},
                {"choices": [{"delta": {"content": " "}}]},
                {"choices": [{"delta": {"content": "world"}}]},
                {"usage": {"prompt_tokens": 10, "completion_tokens": 20}},
                {"choices": [{"delta": {"content": ""}}]},
            ]
            litellm_mod.litellm.acompletion = AsyncMock(return_value=self._make_async_iter(chunks))
            litellm_mod.LITELLM_AVAILABLE = True

            provider = litellm_mod.LiteLLMProvider(model="m")
            tokens = []
            async for token in provider.astream(system_prompt="s", question="q"):
                tokens.append(token)

            assert "".join(tokens) == "Hello world"
            assert provider.last_usage == {"prompt": 10, "completion": 20, "model": "m"}
        finally:
            litellm_mod.litellm = saved

    @pytest.mark.asyncio
    async def test_dict_chunks_with_input_output_tokens(self) -> None:
        """Cover dict-path usage with 'input_tokens' / 'output_tokens' keys."""
        import raghub.llm.litellm as litellm_mod

        saved = litellm_mod.litellm
        try:
            litellm_mod.litellm = types.ModuleType("litellm")
            chunks = [
                {"usage": {"input_tokens": 5, "output_tokens": 15}},
            ]
            litellm_mod.litellm.acompletion = AsyncMock(return_value=self._make_async_iter(chunks))
            litellm_mod.LITELLM_AVAILABLE = True

            provider = litellm_mod.LiteLLMProvider(model="gpt-4")
            async for _ in provider.astream(system_prompt="s", question="q"):
                pass

            assert provider.last_usage == {"prompt": 5, "completion": 15, "model": "gpt-4"}
        finally:
            litellm_mod.litellm = saved

    @pytest.mark.asyncio
    async def test_dict_chunk_no_choices_skipped(self) -> None:
        """Dict chunk lacking 'choices' is skipped (line 283–284)."""
        import raghub.llm.litellm as litellm_mod

        saved = litellm_mod.litellm
        try:
            litellm_mod.litellm = types.ModuleType("litellm")
            chunks = [
                {"foo": "bar"},  # no choices → skip
                {"choices": [{"delta": {"content": "hi"}}]},
            ]
            litellm_mod.litellm.acompletion = AsyncMock(return_value=self._make_async_iter(chunks))
            litellm_mod.LITELLM_AVAILABLE = True

            provider = litellm_mod.LiteLLMProvider(model="m")
            tokens = []
            async for token in provider.astream(system_prompt="s", question="q"):
                tokens.append(token)

            assert "".join(tokens) == "hi"
        finally:
            litellm_mod.litellm = saved

    @pytest.mark.asyncio
    async def test_object_chunks_skipped(self) -> None:
        """Cover line 281: non-dict chunks are skipped."""
        import raghub.llm.litellm as litellm_mod

        saved = litellm_mod.litellm
        try:
            litellm_mod.litellm = types.ModuleType("litellm")

            class FakeObj:
                pass

            obj_chunk = FakeObj()
            chunks = [
                obj_chunk,  # not a dict → continue
                {"choices": [{"delta": {"content": "yes"}}]},
            ]
            litellm_mod.litellm.acompletion = AsyncMock(return_value=self._make_async_iter(chunks))
            litellm_mod.LITELLM_AVAILABLE = True

            provider = litellm_mod.LiteLLMProvider(model="m")
            tokens = []
            async for token in provider.astream(system_prompt="s", question="q"):
                tokens.append(token)

            assert "".join(tokens) == "yes"
        finally:
            litellm_mod.litellm = saved

    @pytest.mark.asyncio
    async def test_object_chunk_with_usage(self) -> None:
        """Cover object-path usage in astream (lines 276–280).

        Non-dict chunk carries usage data (object path for ``getattr``);
        dict chunk carries content.
        """
        import raghub.llm.litellm as litellm_mod

        saved = litellm_mod.litellm
        try:
            litellm_mod.litellm = types.ModuleType("litellm")

            class FakeUsage:
                prompt_tokens = 3
                completion_tokens = 7

            # Non-dict chunk with usage → hits object-path (lines 269, 276-280)
            class FakeUsageChunk:
                usage = FakeUsage()

            chunks = [
                FakeUsageChunk(),
                {"choices": [{"delta": {"content": "obj"}}]},
            ]
            litellm_mod.litellm.acompletion = AsyncMock(return_value=self._make_async_iter(chunks))
            litellm_mod.LITELLM_AVAILABLE = True

            provider = litellm_mod.LiteLLMProvider(model="m")
            tokens = []
            async for token in provider.astream(system_prompt="s", question="q"):
                tokens.append(token)

            assert "".join(tokens) == "obj"
            assert provider.last_usage == {"prompt": 3, "completion": 7, "model": "m"}
        finally:
            litellm_mod.litellm = saved

    @pytest.mark.asyncio
    async def test_astream_no_usage_no_last_usage(self) -> None:
        """When no usage chunk appears and no tokens accumulated, last_usage stays None."""
        import raghub.llm.litellm as litellm_mod

        saved = litellm_mod.litellm
        try:
            litellm_mod.litellm = types.ModuleType("litellm")
            chunks = [
                {"choices": [{"delta": {"content": "hello"}}]},
            ]
            litellm_mod.litellm.acompletion = AsyncMock(return_value=self._make_async_iter(chunks))
            litellm_mod.LITELLM_AVAILABLE = True

            provider = litellm_mod.LiteLLMProvider(model="m")
            async for _ in provider.astream(system_prompt="s", question="q"):
                pass

            assert provider.last_usage is None
        finally:
            litellm_mod.litellm = saved

    @pytest.mark.asyncio
    async def test_astream_usage_captured_at_end(self) -> None:
        """Cover line 292: usage snapshot written after loop."""
        import raghub.llm.litellm as litellm_mod

        saved = litellm_mod.litellm
        try:
            litellm_mod.litellm = types.ModuleType("litellm")
            chunks = [
                {"choices": [{"delta": {"content": "t"}}]},
            ]
            litellm_mod.litellm.acompletion = AsyncMock(return_value=self._make_async_iter(chunks))
            litellm_mod.LITELLM_AVAILABLE = True

            provider = litellm_mod.LiteLLMProvider(model="m")
            async for _ in provider.astream(system_prompt="s", question="q"):
                pass

            # No usage info in chunks → no last_usage
            assert provider.last_usage is None
        finally:
            litellm_mod.litellm = saved


class TestGenerateRecordUsageIntegration:
    """Verify generate + record_usage together with the dict path."""

    def test_generate_dict_usage_captures_input_output_tokens(self) -> None:
        """Cover both generate and record_usage dict paths in one flow."""
        import raghub.llm.litellm as litellm_mod

        saved = litellm_mod.litellm
        try:
            litellm_mod.litellm = types.ModuleType("litellm")
            litellm_mod.litellm.completion = MagicMock(
                return_value={
                    "choices": [{"message": {"content": "answer"}}],
                    "usage": {"input_tokens": 7, "output_tokens": 14},
                }
            )
            litellm_mod.LITELLM_AVAILABLE = True

            provider = litellm_mod.LiteLLMProvider(model="claude")
            result = provider.generate(system_prompt="s", question="q")
            assert result == "answer"
            assert provider.last_usage == {"prompt": 7, "completion": 14, "model": "claude"}
        finally:
            litellm_mod.litellm = saved


class TestLiteLLMProviderInit:
    """Additional edge cases for __init__."""

    def test_default_model_name(self) -> None:
        from raghub.llm.litellm import LiteLLMProvider

        provider = LiteLLMProvider(api_key="test")
        assert provider.model_name == "gpt-4o-mini"
        assert provider.temperature == 0.2
        assert provider.api_key == "test"
        assert provider.api_base is None

    def test_raises_configuration_error_when_litellm_not_installed(self) -> None:
        """Cover line 67: __init__ raises ConfigurationError."""
        import raghub.llm.litellm as litellm_mod

        saved_available = litellm_mod.LITELLM_AVAILABLE
        saved_litellm = litellm_mod.litellm
        try:
            litellm_mod.litellm = None
            litellm_mod.LITELLM_AVAILABLE = False
            with pytest.raises(Exception, match="litellm is not installed"):
                litellm_mod.LiteLLMProvider(model="m")
        finally:
            litellm_mod.litellm = saved_litellm
            litellm_mod.LITELLM_AVAILABLE = saved_available


class TestBuildMessagesContext:
    """Cover lines 131–132 of litellm.py — context formatting."""

    def test_context_is_formatted_and_appended(self) -> None:
        from raghub.llm.litellm import LiteLLMProvider

        provider = LiteLLMProvider(api_key="test")
        messages = provider.build_messages(
            system_prompt="sys",
            question="q",
            context=["chunk A", "chunk B", "chunk C"],
        )
        assert len(messages) == 3
        assert messages[0] == {"role": "system", "content": "sys"}
        assert messages[1] == {
            "role": "system",
            "content": "Context:\nchunk A\n\n---\n\nchunk B\n\n---\n\nchunk C",
        }
        assert messages[2] == {"role": "user", "content": "q"}

    def test_context_empty_omits_context_message(self) -> None:
        from raghub.llm.litellm import LiteLLMProvider

        provider = LiteLLMProvider(api_key="test")
        messages = provider.build_messages(
            system_prompt="sys",
            question="q",
            context=[],
        )
        assert len(messages) == 2
        assert messages[1] == {"role": "user", "content": "q"}

    def test_single_context_item(self) -> None:
        from raghub.llm.litellm import LiteLLMProvider

        provider = LiteLLMProvider(api_key="test")
        messages = provider.build_messages(
            system_prompt="sys",
            question="q",
            context=["only one"],
        )
        assert len(messages) == 3
        assert messages[1]["content"] == "Context:\nonly one"
