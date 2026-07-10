from __future__ import annotations

import pytest

from raghub.core.container import build_application


class TestGetAttr:
    def test_getattr_known_names(self) -> None:
        import raghub.core.container as ctr

        app_cls = ctr.DynamicRagApplication
        ctr_cls = ctr.DynamicRagContainer
        builder = ctr.build_container
        assert callable(app_cls)
        assert callable(ctr_cls)
        assert callable(builder)

    def test_getattr_unknown_name_raises(self) -> None:
        import raghub.core.container as ctr

        with pytest.raises(AttributeError, match="no attribute 'Nonsense'"):
            _ = ctr.Nonsense


class TestBuildApplication:
    async def test_build_application_requires_settings(self, monkeypatch) -> None:
        from raghub.config.settings import AppSettings

        settings = AppSettings(
            environment="test",
            jwt_secret="test-secret",
            data_dir="/tmp/raghub_test",
        )
        builder_called = False

        async def mock_build_container(s: AppSettings) -> object:
            nonlocal builder_called
            builder_called = True
            assert s is settings
            from raghub.services.application import DynamicRagContainer

            return DynamicRagContainer(
                settings=s,
                logger=object(),
                metrics=object(),
                authenticator=object(),
                authorization=object(),
                sessions=object(),
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

        monkeypatch.setattr(
            "raghub.services.application.build_container", mock_build_container
        )
        monkeypatch.setattr(
            "raghub.core.container.load_settings", lambda *a, **kw: settings
        )

        await build_application()
        assert builder_called
