from __future__ import annotations

import pytest

from raghub.core import build_application


class TestContainerCore:
    def test_known_names_importable(self) -> None:
        from raghub.services.application import (
            DynamicRagApplication,
            DynamicRagContainer,
            build_container,
        )

        assert callable(DynamicRagApplication)
        assert callable(DynamicRagContainer)
        assert callable(build_container)


class TestBuildApplication:
    async def test_build_application_requires_settings(self, monkeypatch) -> None:
        from raghub.config import Settings

        settings = Settings(
            environment="test",
            jwt_secret="test-secret",
            data_dir="/tmp/raghub_test",
        )
        builder_called = False

        async def mock_build_container(s: Settings) -> object:
            nonlocal builder_called
            builder_called = True
            assert s is settings
            from raghub.services.application import DynamicRagContainer

            return DynamicRagContainer(
                settings=s,
                logger=object(),
                metrics=object(),
                authorization=object(),
                registry=object(),
                conversation=object(),
                embeddings=object(),
                llm=object(),
                vector_store=object(),
                prompt_builder=object(),
                ingestion=object(),
                retrieval=object(),
                image_store=object(),
                user_store=object(),
                parser_registry=object(),
                store=object(),
                uow=object(),
            )

        monkeypatch.setattr("raghub.services.application.build_container", mock_build_container)
        monkeypatch.setattr("raghub.config.Settings.load", classmethod(lambda cls, *a, **kw: settings))

        await build_application()
        assert builder_called
