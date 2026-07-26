"""Configuration package.

This package loads runtime configuration from a YAML profile plus
environment variables via :func:`load_settings`. The parsed
:class:`AppSettings` is then injected into the service container at
startup.

The actual implementation lives in :mod:`.settings`; this ``__init__``
re-exports the two public names so callers can write
``from raghub.config import AppSettings, load_settings``.
"""

from .settings import AppSettings, load_settings

__all__ = ["AppSettings", "load_settings"]