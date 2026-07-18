"""Pytest fixtures and shared test configuration.

This module sets up a stable test environment by:

1. Defaulting :envvar:`JWT_SECRET` to a non-production value so the
   auth layer can mint tokens without an explicit secret.
2. Disabling passwordless login by default.
3. Pinning :envvar:`CORS_ORIGINS` to a non-wildcard value so the
   production guard refuses wildcard+credentials at startup; tests
   that explicitly need wildcard CORS must clear the env var.
4. Defaulting :envvar:`RAGHUB_RUN_PLATFORM_TESTS` to ``"1"`` so the
   end-to-end / dynamic-application test modules are part of the
   normal test run. Set the env var to ``"0"`` to opt out locally.
5. Adding the repository root to :data:`sys.path` so absolute imports
   like ``from raghub.…`` resolve from the working tree.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("JWT_SECRET", "test-secret-must-be-32-bytes-or-longer-for-sha256")
os.environ.setdefault("RAG_ALLOW_PASSWORDLESS", "0")
os.environ.setdefault("CORS_ORIGINS", "http://testserver")
os.environ.setdefault("RAGHUB_RUN_PLATFORM_TESTS", "1")
os.environ.setdefault("RAG_ZVEC_DIR", str(Path("/tmp").resolve() / "raghub_test_zvec"))

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
