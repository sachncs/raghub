"""Health checks and monitoring.

This package bundles runtime health-probing utilities. Currently the
only implementation is :class:`HealthService`, which fans out a
``health()`` call across a set of named components.
"""

from .health import HealthService

__all__ = ["HealthService"]