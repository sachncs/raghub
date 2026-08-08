"""Coverage tests for :mod:`raghub.core`."""

from __future__ import annotations

from typing import Any

import pytest

from raghub.core import (
    DocumentState,
    DocumentStateMachine,
    RbacGuard,
    allowed_company_filter,
    can_access_company,
)
from raghub.models import DocumentLifecycleStatus as Status

# ---------------------------------------------------------------------------
# DocumentState
# ---------------------------------------------------------------------------


def test_document_state_transition_returns_new_state() -> None:
    """``DocumentState.transition`` builds a new state with the target status."""
    state = DocumentState.transition(Status.Processing)
    assert state.status == Status.Processing


# ---------------------------------------------------------------------------
# DocumentStateMachine
# ---------------------------------------------------------------------------


@pytest.fixture
def machine() -> DocumentStateMachine:
    """Return a fresh :class:`DocumentStateMachine`."""
    return DocumentStateMachine()


def test_state_machine_can_transition_legal(machine: DocumentStateMachine) -> None:
    """Legal transitions return ``True``."""
    assert machine.can_transition(Status.New, Status.Validating) is True
    assert machine.can_transition(Status.Indexing, Status.Ready) is True


def test_state_machine_cannot_transition_illegal(machine: DocumentStateMachine) -> None:
    """Illegal transitions return ``False``."""
    assert machine.can_transition(Status.New, Status.Ready) is False
    assert machine.can_transition(Status.Archived, Status.New) is False


def test_state_machine_archived_is_terminal(machine: DocumentStateMachine) -> None:
    """``ARCHIVED`` allows no transitions."""
    for target in Status:
        assert machine.can_transition(Status.Archived, target) is False


def test_state_machine_failed_is_terminal(machine: DocumentStateMachine) -> None:
    """``FAILED`` allows no transitions."""
    for target in Status:
        assert machine.can_transition(Status.Failed, target) is False


def test_state_machine_unknown_current_returns_false(machine: DocumentStateMachine) -> None:
    """An unknown current status returns ``False`` rather than raising."""
    assert machine.can_transition("not-a-status", Status.Ready) is False  # type: ignore[arg-type]


def test_state_machine_allowed_table_covers_all_statuses(machine: DocumentStateMachine) -> None:
    """Every defined status appears in the allowed-transition table."""
    for status in Status:
        assert status in machine.allowed


# ---------------------------------------------------------------------------
# RbacGuard
# ---------------------------------------------------------------------------


def _user(**overrides: Any) -> Any:
    """Build a user-like object with sensible defaults."""
    defaults: dict[str, Any] = {
        "is_admin": False,
        "allowed_companies": ["acme"],
        "allowed_groups": [],
    }
    defaults.update(overrides)
    return type("_U", (), defaults)()


def test_rbac_admin_company_filter_is_empty() -> None:
    """An admin user gets an empty company filter (``{}``)."""
    user = _user(is_admin=True)
    assert RbacGuard(user).company_filter() == {}


def test_rbac_non_admin_company_filter_is_list() -> None:
    """A non-admin user gets a ``{"company": [...]}`` filter."""
    user = _user(allowed_companies=["acme", "beta"])
    assert RbacGuard(user).company_filter() == {"company": ["acme", "beta"]}


def test_rbac_admin_can_access_any_company() -> None:
    """An admin user can access any company."""
    user = _user(is_admin=True, allowed_companies=[])
    guard = RbacGuard(user)
    assert guard.can_access("anything") is True


def test_rbac_user_can_access_allowed_company() -> None:
    """A non-admin can access companies in their allow-list."""
    user = _user(allowed_companies=["acme"])
    assert RbacGuard(user).can_access("acme") is True


def test_rbac_user_cannot_access_other_company() -> None:
    """A non-admin cannot access companies outside their allow-list."""
    user = _user(allowed_companies=["acme"])
    assert RbacGuard(user).can_access("other") is False


def test_rbac_empty_allow_list_denies_all() -> None:
    """A non-admin with an empty allow-list is denied everything."""
    user = _user(allowed_companies=[])
    guard = RbacGuard(user)
    assert guard.can_access("acme") is False


def test_allowed_company_filter_delegates_to_guard() -> None:
    """The module-level helper delegates to :class:`RbacGuard`."""
    user = _user(allowed_companies=["x"])
    assert allowed_company_filter(user) == {"company": ["x"]}


def test_can_access_company_delegates_to_guard() -> None:
    """The module-level helper delegates to :class:`RbacGuard`."""
    user = _user(allowed_companies=["acme"], is_admin=False)
    assert can_access_company(user, "acme") is True
    assert can_access_company(user, "other") is False
