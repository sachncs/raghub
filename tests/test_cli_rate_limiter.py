"""Tests for the CLI rate limiter."""

from __future__ import annotations

import io
import subprocess
import sys
from contextlib import redirect_stderr
from unittest.mock import patch

import pytest

from raghub.cli import main as main_module
from raghub.cli.rate_limiter import CLIRateLimiter, RateLimitExceeded


def test_rate_limiter_disabled_allows_every_call() -> None:
    """A disabled limiter never raises regardless of the call volume."""
    limiter = CLIRateLimiter(rate=0.001, burst=1, enabled=False)
    for _ in range(100):
        assert limiter.allow("ingest") is True


def test_rate_limiter_allows_within_burst() -> None:
    """Calls within the initial burst capacity are admitted."""
    limiter = CLIRateLimiter(rate=1.0, burst=5, enabled=True)
    for _ in range(5):
        assert limiter.allow("ingest") is True


def test_rate_limiter_denies_when_burst_exhausted() -> None:
    """After exhausting the burst, further calls are denied."""
    limiter = CLIRateLimiter(rate=0.0, burst=1, enabled=True)
    assert limiter.allow("ingest") is True
    assert limiter.allow("ingest") is False


def test_check_raises_on_exceeded() -> None:
    """``check`` raises :class:`RateLimitExceeded` when the bucket is empty."""
    limiter = CLIRateLimiter(rate=0.0, burst=1, enabled=True)
    limiter.check("ingest")
    with pytest.raises(RateLimitExceeded, match="Rate limit exceeded"):
        limiter.check("ingest")


def test_check_passes_when_disabled() -> None:
    """``check`` is a no-op when the limiter is disabled."""
    limiter = CLIRateLimiter(enabled=False)
    for _ in range(100):
        limiter.check("ingest")


def test_main_prints_rate_limit_error_and_returns_one() -> None:
    """main() prints the rate-limit error and exits with code 1."""
    real_limiter = main_module._limiter
    exhaust_limiter = CLIRateLimiter(rate=0.0, burst=0, enabled=True)
    main_module._limiter = exhaust_limiter
    try:
        with patch(
            "argparse.ArgumentParser.parse_args",
            return_value=_ns(command="ingest", handler=lambda ns: 0),
        ):
            buf = io.StringIO()
            with redirect_stderr(buf):
                rc = main_module.main()
        assert rc == 1
        assert "Rate limit exceeded" in buf.getvalue()
    finally:
        main_module._limiter = real_limiter


def test_python_m_raghub_cli_runs() -> None:
    """``python -m raghub.cli`` works end-to-end and exits with 0."""
    result = subprocess.run(
        [sys.executable, "-m", "raghub.cli", "version"],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr


def _ns(**attrs: object) -> object:
    import argparse

    return argparse.Namespace(**attrs)