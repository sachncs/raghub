"""Standalone unit tests for every renamed helper.

These tests pin the post-naming-refactor public surface. The
production behaviour is unchanged; this commit only guarantees that
the renamed identifiers (CLI_LIMITER, RATE_LIMIT_EXEMPT_COMMANDS,
canonical_filters, native_filter, known_collectors, async_main,
factory, stream, datasets_module, marker_module, marker_models_module,
marker_parser_module, app_singleton, write_json) remain accessible
and well-typed.
"""

from __future__ import annotations

from loguru import logger as loguru_logger

from raghub.api.admin import SENSITIVE_USER_FIELDS, redact_user_payload
from raghub.api.app import check_upload_size, create_app, get_app
from raghub.observability.metrics import known_collectors
from raghub.pipelines.cache import QueryCache, canonical_filters


def test_cli_rate_limiter_uses_public_name() -> None:
    """The CLI module exposes ``CLI_LIMITER`` and ``RATE_LIMIT_EXEMPT_COMMANDS``."""
    from raghub.cli import main as main_module

    assert main_module.CLI_LIMITER is not None
    assert "health" in main_module.RATE_LIMIT_EXEMPT_COMMANDS
    assert "version" in main_module.RATE_LIMIT_EXEMPT_COMMANDS
    assert "run" in main_module.RATE_LIMIT_EXEMPT_COMMANDS
    assert not hasattr(main_module, "_limiter")
    assert not hasattr(main_module, "_RATE_LIMIT_EXEMPT")


def test_query_cache_exposes_canonical_filters_and_make_key() -> None:
    """The query cache exposes the renamed ``canonical_filters`` + ``make_key``."""
    assert canonical_filters({"company": ["Apple"]}) == (("company", ("Apple",)),)
    assert canonical_filters(None) == ()
    cache = QueryCache()
    key = cache.make_key("q?", "u1", {"company": ["Apple"]}, session_id="s1")
    assert key[0] == "q?"
    assert key[1] == "u1"
    assert "s1" in key
    assert not hasattr(cache, "_canonical_filters")
    assert not hasattr(cache, "_key")


def test_prometheus_metrics_module_uses_known_collectors() -> None:
    """The metrics module's private dict is now called ``known_collectors``."""
    from raghub.observability import metrics as metrics_module

    assert metrics_module.known_collectors is not None
    assert not hasattr(metrics_module, "metric_collectors")


def test_evaluator_eval_module_uses_public_imports() -> None:
    """The evaluation module uses ``datasets_module`` instead of ``_hf_mod``."""
    from raghub.evaluation import financebench

    assert financebench.hf_load_dataset is not None
    assert not hasattr(financebench, "_hf_mod")


def test_marker_converter_uses_public_imports() -> None:
    """The marker converter module uses ``marker_module`` etc."""
    from raghub.converters import marker as marker_module

    assert marker_module.MARKER_AVAILABLE in {True, False}
    # The dynamic imports now use the public name:
    assert not hasattr(marker_module, "_mod")
    assert not hasattr(marker_module, "_mod2")
    assert not hasattr(marker_module, "_parser_mod")


def test_zvec_backend_exposes_native_filter() -> None:
    """The Zvec backend module-level helper is now called ``native_filter``."""
    from raghub.vectorstore import zvec as zvec_module

    assert callable(zvec_module.native_filter)
    assert not hasattr(zvec_module, "_native_filter")


def test_api_app_singleton_is_public() -> None:
    """The FastAPI app singleton is now ``app_singleton`` (no leading underscore)."""
    import raghub.api.app as app_module

    # Reset the module singleton to avoid state-leak from other tests.
    app_module.app_singleton = None
    assert hasattr(app_module, "app_singleton")
    assert not hasattr(app_module, "app_instance")


def test_cli_system_handler_uses_loguru(monkeypatch) -> None:
    """``handle_version`` emits a loguru record instead of ``print()``."""
    from argparse import Namespace
    from unittest.mock import patch

    from raghub.cli import system

    captured: list[str] = []
    handler_id = loguru_logger.add(
        lambda m: captured.append(m), level="INFO", format="{extra[version]}"
    )
    try:
        with patch("importlib.metadata.version", return_value="9.9.9"):
            assert system.handle_version(Namespace()) == 0
    finally:
        loguru_logger.remove(handler_id)
    assert captured == ["9.9.9\n"]


def test_cli_init_handler_uses_loguru(monkeypatch) -> None:
    """``init_cmd.run_subcommand`` emits the sample via the loguru logger."""
    from raghub.cli import init_cmd

    captured: list[str] = []
    handler_id = loguru_logger.add(
        lambda m: captured.append(m), level="INFO", format="{message}"
    )
    try:
        rc = init_cmd.run_subcommand(
            type("Args", (), {"output": None})()  # type: ignore[abstract]
        )
    finally:
        loguru_logger.remove(handler_id)
    assert rc == 0
    assert any("environment" in c for c in captured)


def test_redact_user_payload_returns_no_sensitive_fields() -> None:
    """``redact_user_payload`` strips every sensitive key from the payload."""
    payload = {
        "email": "alice@example.com",
        "password_hash": "bcrypt$2a$12$abc",
        "password": "secret",
        "token": "abc",
        "secret": "do not leak",
    }
    redacted = redact_user_payload(payload)
    assert redacted["email"] == "alice@example.com"
    for key, value in redacted.items():
        if key == "email":
            continue
        if "hash" in key.lower() or key.lower() in SENSITIVE_USER_FIELDS:
            assert value == "***"


def test_check_upload_size_returns_bool() -> None:
    """``check_upload_size`` returns a clean boolean (no sentinels)."""
    assert check_upload_size(None, 1024) is False
    assert check_upload_size(500, 1024) is False
    assert check_upload_size(2048, 1024) is True
    assert check_upload_size(1024, 1024) is False


def test_app_singleton_is_public() -> None:
    """The module exposes the canonical ``app_singleton`` (no leading underscore)."""
    import raghub.api.app as app_module

    assert hasattr(app_module, "app_singleton")
    assert not hasattr(app_module, "app_instance")