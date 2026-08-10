"""Tests for configuration validation and helper functions."""

from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import patch

import pytest

from raghub.config import (
    ArchiveConfig,
    FeedbackConfig,
    QueueConfig,
    RateLimitConfig,
    Settings,
    TenantsConfig,
    csv_to_transforms,
    env_bool,
)


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
        with (
            patch.dict(
                os.environ, {"RAG_CONFIG_DIR": str(tmp_path), "JWT_SECRET": ""}, clear=False
            ),
            pytest.raises(RuntimeError, match="JWT_SECRET"),
        ):
            Settings.load("production")

    def test_production_rejects_short_jwt_secret(self, tmp_path: Path) -> None:
        (tmp_path / "production.yaml").write_text(
            "environment: production\nallow_passwordless_login: false\n",
            encoding="utf-8",
        )
        with (
            patch.dict(
                os.environ,
                {"RAG_CONFIG_DIR": str(tmp_path), "JWT_SECRET": "short"},
                clear=False,
            ),
            pytest.raises(RuntimeError, match="32 bytes"),
        ):
            Settings.load("production")

    def test_production_rejects_passwordless_login(self, tmp_path: Path) -> None:
        (tmp_path / "production.yaml").write_text(
            "environment: production\nallow_passwordless_login: true\n",
            encoding="utf-8",
        )
        env = {
            "RAG_CONFIG_DIR": str(tmp_path),
            "JWT_SECRET": "a" * 32,
            "RAG_ALLOW_PASSWORDLESS": "1",
        }
        with (
            patch.dict(os.environ, env, clear=False),
            pytest.raises(RuntimeError, match="Passwordless login"),
        ):
            Settings.load("production")


# ---------------------------------------------------------------------------
# v0.9.0 Tier 1 — Item 1: Settings.queue block
# ---------------------------------------------------------------------------


class TestQueueConfig:
    def test_queue_config_defaults(self) -> None:
        """Default QueueConfig has memory backend, max_inflight 256."""
        settings = Settings()
        assert isinstance(settings.queue, QueueConfig)
        assert settings.queue.backend == "memory"
        assert settings.queue.db_path is None
        assert settings.queue.max_inflight == 256

    def test_queue_config_env_override(self) -> None:
        """Env vars RAG_QUEUE_BACKEND and RAG_QUEUE_MAX_INFLIGHT parse."""
        env = {
            "RAG_QUEUE_BACKEND": "sqlite",
            "RAG_QUEUE_MAX_INFLIGHT": "512",
        }
        with patch.dict(os.environ, env, clear=False):
            settings = Settings.load()
        assert settings.queue.backend == "sqlite"
        assert settings.queue.max_inflight == 512

    def test_queue_config_constructor_override(self) -> None:
        """Constructor injection wins over defaults."""
        settings = Settings(queue=QueueConfig(backend="sqlite", max_inflight=1024))
        assert settings.queue.backend == "sqlite"
        assert settings.queue.max_inflight == 1024


# ---------------------------------------------------------------------------
# v0.9.0 Tier 1 — Item 2: Settings.feedback block
# ---------------------------------------------------------------------------


class TestFeedbackConfig:
    def test_feedback_config_defaults(self) -> None:
        """Default FeedbackConfig has none backend."""
        settings = Settings()
        assert isinstance(settings.feedback, FeedbackConfig)
        assert settings.feedback.backend == "none"
        assert settings.feedback.db_path is None
        assert settings.feedback.dsn is None

    def test_feedback_config_env_override(self) -> None:
        """Env vars RAG_FEEDBACK_BACKEND and RAG_FEEDBACK_DSN parse."""
        env = {
            "RAG_FEEDBACK_BACKEND": "postgres",
            "RAG_FEEDBACK_DSN": "postgres://localhost/rag",
        }
        with patch.dict(os.environ, env, clear=False):
            settings = Settings.load()
        assert settings.feedback.backend == "postgres"
        assert settings.feedback.dsn == "postgres://localhost/rag"


# ---------------------------------------------------------------------------
# v0.9.0 Tier 1 — Item 3: Settings.rate_limit block
# ---------------------------------------------------------------------------


class TestRateLimitConfig:
    def test_rate_limit_config_defaults(self) -> None:
        """Default RateLimitConfig has memory backend, 10 rps / 20 burst."""
        from raghub.constants import RATE_LIMIT_BURST, RATE_LIMIT_RPS

        settings = Settings()
        assert isinstance(settings.rate_limit, RateLimitConfig)
        assert settings.rate_limit.backend == "memory"
        assert settings.rate_limit.per_tenant_rps == RATE_LIMIT_RPS == 10.0
        assert settings.rate_limit.per_tenant_burst == RATE_LIMIT_BURST == 20

    def test_rate_limit_config_env_override(self) -> None:
        """Env vars parse correctly; exempt_tenants is CSV-parsed."""
        env = {
            "RAG_RATE_LIMIT_EXEMPT_TENANTS": "acme, beta ,gamma",
        }
        with patch.dict(os.environ, env, clear=False):
            settings = Settings.load()
        assert settings.rate_limit.exempt_tenants == ["acme", "beta", "gamma"]


# ---------------------------------------------------------------------------
# v0.9.0 Tier 1 — Item 4: Settings.archive block
# ---------------------------------------------------------------------------


class TestArchiveConfig:
    def test_archive_config_defaults(self) -> None:
        """Default ArchiveConfig has none backend, local_dir ./data/archives."""
        settings = Settings()
        assert isinstance(settings.archive, ArchiveConfig)
        assert settings.archive.backend == "none"
        assert Path(settings.archive.local_dir) == Path("./data/archives")

    def test_archive_config_env_override(self) -> None:
        """RAG_ARCHIVE_DIR parses to Path."""
        env = {"RAG_ARCHIVE_DIR": "/tmp/archives"}
        with patch.dict(os.environ, env, clear=False):
            settings = Settings.load()
        assert Path(settings.archive.local_dir) == Path("/tmp/archives")


# ---------------------------------------------------------------------------
# v0.9.0 Tier 1 — Item 5: Settings.tenants block
# ---------------------------------------------------------------------------


class TestTenantsConfig:
    def test_tenants_config_defaults(self) -> None:
        """Default TenantsConfig has none resolver, row_level isolation."""
        settings = Settings()
        assert isinstance(settings.tenants, TenantsConfig)
        assert settings.tenants.resolver == "none"
        assert settings.tenants.isolation == "row_level"

    def test_tenants_config_env_override(self) -> None:
        """RAG_TENANTS_ISOLATION parses."""
        env = {"RAG_TENANTS_ISOLATION": "schema_per_tenant"}
        with patch.dict(os.environ, env, clear=False):
            settings = Settings.load()
        assert settings.tenants.isolation == "schema_per_tenant"
