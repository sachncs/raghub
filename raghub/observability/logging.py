"""Structured-logging helpers backed by :mod:`structlog`.

The logger is configured once per process by :func:`build_logger`
with the standard ISO-timestamp + console-renderer pipeline. The
:class:`StructuredLogger` adapter wraps the resulting structlog
logger and exposes a small, hand-rolled ``info``/``warning``/``error``
surface so callers don't have to import structlog directly.
"""

from __future__ import annotations

import logging
from typing import Any

import structlog


def build_logger(level: str = "INFO") -> Any:
    """Configure structlog and return the process-wide logger.

    The standard-library root logger is also configured to emit at
    the requested level so messages from non-structlog code paths
    show up. The structlog pipeline emits ISO-timestamped
    ``key=value`` records to stderr.

    Args:
        level: Minimum log level. ``"INFO"`` by default; unknown
            values fall back to ``logging.INFO``.

    Returns:
        A structlog bound logger ready for ``.info()``/``.warning()``/
        ``.error()`` calls.
    """
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(message)s",
    )
    structlog.configure(
        processors=[
            structlog.stdlib.filter_by_level,
            structlog.stdlib.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.dev.ConsoleRenderer(),
        ],
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )
    return structlog.get_logger()


class StructuredLogger:
    """Thin wrapper around a structlog logger.

    Keeps the project's call sites free of structlog imports and
    makes it easy to swap implementations in tests.
    """

    def __init__(self, logger: Any) -> None:
        """Store the wrapped structlog logger.

        Args:
            logger: A structlog-bound logger instance.
        """
        self.logger = logger

    def info(self, message: str, **kwargs: Any) -> None:
        """Emit an ``info``-level record.

        Args:
            message: The log message.
            **kwargs: Additional structlog key/value pairs.
        """
        self.logger.info(message, **kwargs)

    def warning(self, message: str, **kwargs: Any) -> None:
        """Emit a ``warning``-level record.

        Args:
            message: The log message.
            **kwargs: Additional structlog key/value pairs.
        """
        self.logger.warning(message, **kwargs)

    def error(self, message: str, **kwargs: Any) -> None:
        """Emit an ``error``-level record.

        Args:
            message: The log message.
            **kwargs: Additional structlog key/value pairs.
        """
        self.logger.error(message, **kwargs)