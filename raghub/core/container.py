"""Legacy dependency-injection container and factory helpers.

The :func:`build_application` helper wires the legacy
:class:`raghub.services.application.DynamicRagApplication`. New code
should prefer the public :class:`raghub.RAG` facade; this module is
retained for backwards compatibility and for the FastAPI admin
routes that depend on the auth-aware service container.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from raghub.config.settings import AppSettings, load_settings

if TYPE_CHECKING:
    from raghub.services.application import (
        DynamicRagApplication,
        DynamicRagContainer,
    )


def __getattr__(name: str) -> Any:
    """Lazily expose the legacy builders.

    Args:
        name: One of ``DynamicRagApplication``, ``DynamicRagContainer``,
            or ``build_container``.

    Returns:
        The corresponding object from
        :mod:`raghub.services.application`.

    Raises:
        AttributeError: When ``name`` is not a known lazy attribute.
    """
    if name in {"DynamicRagApplication", "DynamicRagContainer", "build_container"}:
        from raghub.services.application import (
            DynamicRagApplication as app_import,
        )
        from raghub.services.application import (
            DynamicRagContainer as ctr_import,
        )
        from raghub.services.application import (
            build_container as bc_import,
        )

        return {
            "DynamicRagApplication": app_import,
            "DynamicRagContainer": ctr_import,
            "build_container": bc_import,
        }[name]
    raise AttributeError(f"module 'raghub.core.container' has no attribute {name!r}")


async def build_application(profile: str | None = None) -> Any:
    """Build a fully wired :class:`DynamicRagApplication` from configuration.

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
    from raghub.services.application import (
        DynamicRagApplication,
    )
    from raghub.services.application import (
        build_container as build_container_import,
    )

    settings = load_settings(profile)
    container = await build_container_import(settings)
    return DynamicRagApplication(container)


__all__ = [
    "AppSettings",
    "DynamicRagApplication",
    "DynamicRagContainer",
    "build_application",
]
