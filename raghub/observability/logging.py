from __future__ import annotations

from typing import Any

import structlog


def build_logger(level: str = "INFO") -> Any:
    import logging

    logging.basicConfig(level=getattr(logging, level.upper(), logging.INFO), format="%(message)s")
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
    def __init__(self, logger: Any) -> None:
        self.logger = logger

    def info(self, message: str, **kwargs: Any) -> None:
        self.logger.info(message, **kwargs)

    def warning(self, message: str, **kwargs: Any) -> None:
        self.logger.warning(message, **kwargs)

    def error(self, message: str, **kwargs: Any) -> None:
        self.logger.error(message, **kwargs)
