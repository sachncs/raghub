"""Legacy dependency-injection container and factory helpers.

The :class:`ContainerBuilder` class wires the legacy
:class:`raghub.services.application.DynamicRagApplication`. New code
should prefer the public :class:`raghub.RAG` facade; this module is
retained for backwards compatibility and for the FastAPI admin
routes that depend on the auth-aware service container.

The :func:`build_application` async coroutine is kept as a thin
facade over :class:`ContainerBuilder.build` so existing call sites
keep working.

Module-level ``__getattr__`` lazily exposes ``DynamicRagApplication``,
``DynamicRagContainer``, and ``build_container`` for callers that
still import them from this module — the legacy surface is small
enough that explicit ``__getattr__`` is clearer than an eager import
that would couple :mod:`raghub.core` to :mod:`raghub.services` at
load time.
"""

from __future__ import annotations

from typing import Any

from raghub.config.settings import AppSettings, load_settings

_LAZY_EXPORTS: dict[str, str] = {
    "DynamicRagApplication": "DynamicRagApplication",
    "DynamicRagContainer": "DynamicRagContainer",
    "build_container": "build_container",
}


def __getattr__(name: str) -> Any:
    """Lazily expose legacy builders and container classes.

    Args:
        name: One of ``DynamicRagApplication``, ``DynamicRagContainer``,
            or ``build_container``.

    Returns:
        The corresponding object from
        :mod:`raghub.services.application`.

    Raises:
        AttributeError: When ``name`` is not a known lazy attribute.
    """
    if name not in _LAZY_EXPORTS:
        raise AttributeError(f"module 'raghub.core.container' has no attribute {name!r}")
    import importlib

    app_module = importlib.import_module("raghub.services.application")
    return getattr(app_module, _LAZY_EXPORTS[name])


class ContainerBuilder:
    """Asynchronously build a wired :class:`DynamicRagApplication`.

    The builder is intentionally small — its only job is to load
    settings (when none are supplied) and delegate to
    :func:`raghub.services.application.build_container`. Splitting it
    into a class makes it easy to inject mocks during tests and
    keeps the legacy :func:`build_application` coroutine trivial.

    Attributes:
        settings: The settings to build with. When ``None``, the
            builder calls :func:`load_settings` with no profile.
        profile: Optional profile name passed to :func:`load_settings`
            when ``settings`` is ``None``.
    """

    def __init__(
        self,
        settings: AppSettings | None = None,
        *,
        profile: str | None = None,
    ) -> None:
        """Store the settings and optional profile name."""
        self.settings = settings
        self.profile = profile

    async def build(self) -> Any:
        """Build a fully-wired application.

        Returns:
            A ready-to-use :class:`DynamicRagApplication`.
        """
        import importlib

        app_module = importlib.import_module("raghub.services.application")
        settings = self.settings or load_settings(self.profile)
        container = await app_module.build_container(settings)
        return app_module.DynamicRagApplication(container)


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
    return await ContainerBuilder(profile=profile).build()


__all__ = [
    "AppSettings",
    "ContainerBuilder",
    "DynamicRagApplication",
    "DynamicRagContainer",
    "build_application",
]