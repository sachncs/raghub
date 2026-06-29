"""Pytest fixtures and shared test configuration.

This module sets up a stable test environment by:

1. Defaulting :envvar:`JWT_SECRET` to a non-production value so the
   auth layer can mint tokens without an explicit secret.
2. Disabling passwordless login by default.
3. Adding the repository root to :data:`sys.path` so absolute imports
   like ``from raghub.…`` resolve from the working tree.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("JWT_SECRET", "test-secret-must-be-32-bytes-or-longer-for-sha256")
os.environ.setdefault("RAG_ALLOW_PASSWORDLESS", "0")

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))