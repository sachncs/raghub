"""Exception-aware callable execution helpers."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any


def capture(call: Callable[..., Any], *args: Any, **kwargs: Any) -> tuple[Any, Exception | None]:
    """Return a callable result and any raised exception."""
    try:
        return call(*args, **kwargs), None
    except Exception as error:
        return None, error


__all__ = ["capture"]
