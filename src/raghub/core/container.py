"""Dependency injection container and factory helpers."""

from __future__ import annotations

from raghub.config.settings import AppSettings, load_settings
from raghub.services.application import DynamicRagApplication, DynamicRagContainer, build_container


def build_application(profile: str | None = None) -> DynamicRagApplication:
    """Build a fully wired application from configuration."""

    settings = load_settings(profile)
    return DynamicRagApplication(build_container(settings))


__all__ = [
    "AppSettings",
    "DynamicRagApplication",
    "DynamicRagContainer",
    "build_application",
    "build_container",
]

