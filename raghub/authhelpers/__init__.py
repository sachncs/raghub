"""Deprecated alias for :mod:`raghub.auth_support`.

This module is preserved for one deprecation cycle so existing callers
of ``from raghub.authhelpers import App`` keep working. New code should
import directly from :mod:`raghub.auth_support`.

Emits :class:`DeprecationWarning` on import; the shim will be removed in
the release after the next minor.
"""

from __future__ import annotations

import warnings

from raghub.auth_support import App, Auth, Bearer

warnings.warn(
    "raghub.authhelpers has been renamed to raghub.auth_support; "
    "import from raghub.auth_support instead. "
    "This compatibility shim will be removed in the next minor release.",
    DeprecationWarning,
    stacklevel=2,
)


__all__ = ["App", "Auth", "Bearer"]