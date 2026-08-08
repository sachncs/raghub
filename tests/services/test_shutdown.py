"""Tests for ``raghub.services.shutdown`` (Shutdown coordinator)."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from raghub.services.shutdown import Shutdown


def test_shutdown_targets_includes_expected_collaborators() -> None:
    """``SHUTDOWN_TARGETS`` lists the standard collaborator attributes."""

    assert "background_ingestion" in Shutdown.SHUTDOWN_TARGETS
    assert "ingestion" in Shutdown.SHUTDOWN_TARGETS
    assert "image_store" in Shutdown.SHUTDOWN_TARGETS
    assert "vector_store" in Shutdown.SHUTDOWN_TARGETS
    assert "store" in Shutdown.SHUTDOWN_TARGETS
    assert "uow" in Shutdown.SHUTDOWN_TARGETS


@pytest.mark.asyncio
async def test_release_invokes_close_on_each_collaborator() -> None:
    """``Shutdown.release`` calls ``close()`` on every held collaborator."""

    closed: list[str] = []

    class _MockCollab:
        def __init__(self, name: str) -> None:
            self.name = name

        def close(self) -> None:
            closed.append(self.name)

    container = SimpleNamespace(
        background_ingestion=_MockCollab("bgi"),
        ingestion=_MockCollab("ing"),
        image_store=_MockCollab("img"),
        vector_store=_MockCollab("vec"),
        store=_MockCollab("store"),
        uow=_MockCollab("uow"),
    )
    coordinator = Shutdown(container)
    await coordinator.release()
    assert closed == ["bgi", "ing", "img", "vec", "store", "uow"]


@pytest.mark.asyncio
async def test_release_falls_back_to_shutdown_method() -> None:
    """When ``close`` is absent, ``shutdown`` is invoked."""

    shutdown_calls: list[str] = []

    class _MockCollab:
        def __init__(self, name: str) -> None:
            self.name = name

        def shutdown(self) -> None:
            shutdown_calls.append(self.name)

    container = SimpleNamespace(
        background_ingestion=None,
        ingestion=None,
        image_store=None,
        vector_store=None,
        store=None,
        uow=_MockCollab("uow"),
    )
    coordinator = Shutdown(container)
    await coordinator.release()
    assert shutdown_calls == ["uow"]


@pytest.mark.asyncio
async def test_release_skips_missing_collaborators() -> None:
    """``Shutdown.release`` is a no-op when every collaborator is missing."""

    container = SimpleNamespace()
    coordinator = Shutdown(container)
    # Must not raise even though every SHUTDOWN_TARGETS attr is absent.
    await coordinator.release()


@pytest.mark.asyncio
async def test_release_skips_collaborators_without_lifecycle() -> None:
    """Collaborators lacking ``close`` and ``shutdown`` are silently skipped."""

    container = SimpleNamespace(
        background_ingestion=None,
        ingestion=SimpleNamespace(),  # no close / shutdown
        image_store=None,
        vector_store=None,
        store=None,
        uow=None,
    )
    coordinator = Shutdown(container)
    await coordinator.release()  # no exception


@pytest.mark.asyncio
async def test_release_awaits_async_close() -> None:
    """``Shutdown.release`` awaits a coroutine returned by ``close``."""

    closed = False

    class _AsyncCollab:
        async def close(self) -> None:
            nonlocal closed
            closed = True

    container = SimpleNamespace(
        background_ingestion=None,
        ingestion=None,
        image_store=None,
        vector_store=None,
        store=None,
        uow=_AsyncCollab(),
    )
    coordinator = Shutdown(container)
    await coordinator.release()
    assert closed is True


@pytest.mark.asyncio
async def test_release_propagates_first_failure() -> None:
    """A failure on one collaborator propagates; later collaborators are not closed."""

    closed: list[str] = []

    class _FailClose:
        name = "fail"

        def close(self) -> None:
            raise RuntimeError("intentional")

    class _OkClose:
        def __init__(self, name: str) -> None:
            self.name = name

        def close(self) -> None:
            closed.append(self.name)

    container = SimpleNamespace(
        background_ingestion=_OkClose("bgi"),
        ingestion=_FailClose(),
        image_store=None,
        vector_store=None,
        store=None,
        uow=None,
    )
    coordinator = Shutdown(container)
    with pytest.raises(RuntimeError, match="intentional"):
        await coordinator.release()
    assert closed == ["bgi"]