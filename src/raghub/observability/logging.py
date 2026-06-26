"""Structured logging helpers."""

from __future__ import annotations

import logging
from typing import Any


def build_logger(level: str = "INFO") -> logging.Logger:
    """Configure a package logger."""

    logger = logging.getLogger("raghub")
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s")
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    logger.setLevel(level.upper())
    return logger


class StructuredLogger:
    """Minimal structured logger adapter."""

    def __init__(self, logger: logging.Logger) -> None:
        self.logger = logger

    def info(self, message: str, **kwargs: Any) -> None:
        self.logger.info("%s %s", message, kwargs)

    def warning(self, message: str, **kwargs: Any) -> None:
        self.logger.warning("%s %s", message, kwargs)

    def error(self, message: str, **kwargs: Any) -> None:
        self.logger.error("%s %s", message, kwargs)

