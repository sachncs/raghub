"""Dependency injection container and factory helpers.

This module re-exports the application builder and the container type
so callers can do ``from raghub.core.container import build_application``
without depending on the heavier :mod:`raghub.services.application`
module directly. The re-exports keep the import graph shallow for
embedders that only need the factory entry point.
"""

from __future__ import annotations

from raghub.config.settings import AppSettings, load_settings
from raghub.services.application import DynamicRagApplication, DynamicRagContainer, build_container


async def build_application(profile: str | None = None) -> DynamicRagApplication:
    """Build a fully wired application from configuration.

    Args:
        profile: Optional settings profile name. Passed to
            :func:`load_settings` to allow environment-specific overrides
            (e.g. ``"dev"``, ``"prod"``).

    Returns:
        A ready-to-use :class:`DynamicRagApplication`.

    Raises:
        RuntimeError: If ``JWT_SECRET`` is missing from settings or any
            required collaborator fails to initialise.
    """
    settings = load_settings(profile)
    container = await build_container(settings)
    return DynamicRagApplication(container)


__all__ = [
    "AppSettings",
    "DynamicRagApplication",
    "DynamicRagContainer",
    "build_application",
    "build_container",
]
