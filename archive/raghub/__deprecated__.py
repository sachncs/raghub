"""Deprecation shim for the raghub 0.9.x Python release.

Importing this module prints a one-shot warning directing users to the
TypeScript packages that have replaced the Python tree.

Usage::

    import raghub.__deprecated__  # noqa: F401

This shim does not import or execute anything from the rest of the
package. It exists so callers can opt into a single, loud warning at
process startup without changing any other import behaviour.
"""

from __future__ import annotations

import warnings

_WARNING_EMITTED = False
_MESSAGE = (
    "raghub 0.9.x (Python) is deprecated and unsupported. "
    "The active codebase is the TypeScript monorepo at "
    "https://github.com/sachncs/raghub. Install with "
    "`npm install @raghub/core @raghub/orchestrator @raghub/api`."
)


def emit() -> None:
    """Print the deprecation warning exactly once per process."""
    global _WARNING_EMITTED
    if _WARNING_EMITTED:
        return
    _WARNING_EMITTED = True
    warnings.warn(_MESSAGE, DeprecationWarning, stacklevel=2)


emit()