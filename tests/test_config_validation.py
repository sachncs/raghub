"""Tests for configuration validation and helper functions."""

from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import patch

import pytest

from raghub.config import Settings, csv_to_transforms, env_bool


class TestEnvBool:
    def test_returns_default_when_unset(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            assert env_bool("NONEXISTENT_VAR_FOR_TEST", default=True) is True
            assert env_bool("NONEXISTENT_VAR_FOR_TEST", default=False) is False

    @pytest.mark.parametrize(
        "value,expected",
        [
            ("1", True),
            ("true", True),
            ("True", True),
            ("TRUE", True),
            ("yes", True),
            ("on", True),
            (" ON ", True),
            ("0", False),
            ("false", False),
            ("no", False),
            ("off", False),
            ("random_string", False),
            ("", False),
        ],
    )
    def test_truthy_and_falsy_values(self, value: str, expected: bool) -> None:
        with patch.dict(os.environ, {"TEST_BOOL_VAR": value}):
            assert env_bool("TEST_BOOL_VAR", default=False) is expected


class TestCsvToTransforms:
    def test_empty_string_returns_default(self) -> None:
        result = csv_to_transforms("", ["hyde", "multi_query"])
        assert result == ["hyde", "multi_query"]

    def test_valid_transforms_parsed(self) -> None:
        result = csv_to_transforms("hyde,step_back", [])
        assert result == ["hyde", "step_back"]

    def test_unknown_transforms_dropped(self) -> None:
        result = csv_to_transforms("hyde,unknown_foo,multi_query", [])
        assert result == ["hyde", "multi_query"]

    def test_duplicates_deduplicated(self) -> None:
        result = csv_to_transforms("hyde,hyde,multi_query", [])
        assert result == ["hyde", "multi_query"]

    def test_case_insensitive(self) -> None:
        result = csv_to_transforms("HYDE,Multi_Query", [])
        assert result == ["hyde", "multi_query"]

    def test_whitespace_trimmed(self) -> None:
        result = csv_to_transforms("  hyde , multi_query  ", [])
        assert result == ["hyde", "multi_query"]

    def test_all_valid_transforms(self) -> None:
        result = csv_to_transforms("hyde,multi_query,step_back,decompose", [])
        assert result == ["hyde", "multi_query", "step_back", "decompose"]


class TestAppSettingsOverride:
    def test_override_returns_new_instance(self) -> None:
        original = Settings()
        overridden = original.override(chunk_size_words=999)
        assert original.chunk_size_words != 999
        assert overridden.chunk_size_words == 999

    def test_override_preserves_other_fields(self) -> None:
        original = Settings(llm_model="test-model")
        overridden = original.override(chunk_size_words=999)
        assert overridden.llm_model == "test-model"

    def test_override_unknown_keys_go_to_extra(self) -> None:
        original = Settings()
        overridden = original.override(custom_key="custom_value")
        assert overridden.extra.get("custom_key") == "custom_value"

    def test_override_multiple_fields(self) -> None:
        original = Settings()
        overridden = original.override(chunk_size_words=100, top_k=5)
        assert overridden.chunk_size_words == 100
        assert overridden.top_k == 5


class TestAppSettingsValidation:
    def test_default_values_are_sane(self) -> None:
        settings = Settings()
        assert settings.chunk_size_words > 0
        assert settings.top_k > 0
        assert settings.max_upload_bytes > 0
        assert settings.embedding_dim > 0

    def test_environment_default_is_development(self) -> None:
        settings = Settings()
        assert settings.environment == "development"


class TestProductionValidation:
    def test_production_requires_jwt_secret(self, tmp_path: Path) -> None:
        (tmp_path / "production.yaml").write_text(
            "environment: production\nallow_passwordless_login: false\n",
            encoding="utf-8",
        )
        with patch.dict(
            os.environ, {"RAG_CONFIG_DIR": str(tmp_path), "JWT_SECRET": ""}, clear=False
        ):
            with pytest.raises(RuntimeError, match="JWT_SECRET"):
                Settings.load("production")

    def test_production_rejects_short_jwt_secret(self, tmp_path: Path) -> None:
        (tmp_path / "production.yaml").write_text(
            "environment: production\nallow_passwordless_login: false\n",
            encoding="utf-8",
        )
        with patch.dict(
            os.environ,
            {"RAG_CONFIG_DIR": str(tmp_path), "JWT_SECRET": "short"},
            clear=False,
        ):
            with pytest.raises(RuntimeError, match="32 bytes"):
                Settings.load("production")

    def test_production_rejects_passwordless_login(self, tmp_path: Path) -> None:
        (tmp_path / "production.yaml").write_text(
            "environment: production\nallow_passwordless_login: true\n",
            encoding="utf-8",
        )
        env = {"RAG_CONFIG_DIR": str(tmp_path), "JWT_SECRET": "a" * 32, "RAG_ALLOW_PASSWORDLESS": "1"}
        with patch.dict(os.environ, env, clear=False):
            with pytest.raises(RuntimeError, match="Passwordless login"):
                Settings.load("production")
