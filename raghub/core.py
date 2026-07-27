"""Core domain services and policies.

* :class:`ContainerBuilder` / :func:`build_application` — legacy
  dependency-injection helpers that wire the
  :class:`raghub.services.application.RagApplication`. New
  code should prefer the public :class:`raghub.RAG` facade.
* :class:`DocumentStateMachine` / :class:`DocumentState` — the
  document lifecycle state machine.
* :class:`RbacGuard` / :func:`allowed_company_filter` /
  :func:`can_access_company` — per-user RBAC checks and
  compatibility shims.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from raghub.config import Settings
from raghub.models import DocumentLifecycleStatus


class ContainerBuilder:
    """Asynchronously build a wired :class:`RagApplication`.

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
        settings: Settings | None = None,
        *,
        profile: str | None = None,
    ) -> None:
        """Store the settings and optional profile name."""
        self.settings = settings
        self.profile = profile

    async def build(self) -> Any:
        """Build a fully-wired application.

        Returns:
            A ready-to-use :class:`RagApplication`.
        """
        from raghub.services.application import (
            RagApplication,
            build_container,
        )

        settings = self.settings or Settings.load(self.profile)
        container = await build_container(settings)
        return RagApplication(container)


async def build_application(profile: str | None = None) -> Any:
    """Build a fully wired :class:`RagApplication` from configuration.

    Args:
        profile: Optional settings profile name. Passed to
            :func:`load_settings` to allow environment-specific overrides
            (e.g. ``"dev"``, ``"prod"``).

    Returns:
        A ready-to-use :class:`RagApplication`.

    Raises:
        RuntimeError: If ``JWT_SECRET`` is missing from settings or any
            required collaborator fails to initialise.
    """
    return await ContainerBuilder(profile=profile).build()


@dataclass(frozen=True)
class DocumentState:
    """A document's current lifecycle status.

    This is the value object persisted alongside the document record. It is
    intentionally minimal — the heavy lifting (transitions, validation)
    lives in :class:`DocumentStateMachine` so the state itself can be a
    trivial frozen dataclass.

    Attributes:
        status: The current lifecycle status.
    """

    status: DocumentLifecycleStatus

    def transition(self, target: DocumentLifecycleStatus) -> DocumentState:
        """Return a new :class:`DocumentState` with ``target`` as the status.

        This is a low-level constructor; **callers are responsible for
        verifying the transition is legal** via
        :meth:`DocumentStateMachine.can_transition` first.

        Args:
            target: The status to transition to.

        Returns:
            A new :class:`DocumentState` carrying ``target``.
        """
        return DocumentState(target)


class DocumentStateMachine:
    """Validates transitions between document lifecycle states.

    The class exposes the allowed-transition table as a class attribute
    (``:allowed:``) so callers can introspect the full graph or override it
    in tests. The instance method :meth:`can_transition` simply checks
    membership in the appropriate allow-set.

    Attributes:
        allowed: Mapping from current status to the set of statuses that
            are legal next steps. ``ARCHIVED`` and ``FAILED`` map to empty
            sets, marking them terminal.
    """

    def __init__(self) -> None:
        """Initialise the immutable allowed-transition table."""
        self.allowed: dict[DocumentLifecycleStatus, set[DocumentLifecycleStatus]] = {
            DocumentLifecycleStatus.NEW: {
                DocumentLifecycleStatus.VALIDATING,
                DocumentLifecycleStatus.FAILED,
            },
            DocumentLifecycleStatus.VALIDATING: {
                DocumentLifecycleStatus.PROCESSING,
                DocumentLifecycleStatus.FAILED,
            },
            DocumentLifecycleStatus.PROCESSING: {
                DocumentLifecycleStatus.CHUNKING,
                DocumentLifecycleStatus.FAILED,
            },
            DocumentLifecycleStatus.CHUNKING: {
                DocumentLifecycleStatus.EMBEDDING,
                DocumentLifecycleStatus.FAILED,
            },
            DocumentLifecycleStatus.EMBEDDING: {
                DocumentLifecycleStatus.INDEXING,
                DocumentLifecycleStatus.FAILED,
            },
            DocumentLifecycleStatus.INDEXING: {
                DocumentLifecycleStatus.READY,
                DocumentLifecycleStatus.UPDATING,
                DocumentLifecycleStatus.FAILED,
            },
            DocumentLifecycleStatus.READY: {
                DocumentLifecycleStatus.UPDATING,
                DocumentLifecycleStatus.DELETING,
                DocumentLifecycleStatus.ARCHIVED,
            },
            DocumentLifecycleStatus.UPDATING: {
                DocumentLifecycleStatus.INDEXING,
                DocumentLifecycleStatus.FAILED,
            },
            DocumentLifecycleStatus.DELETING: {
                DocumentLifecycleStatus.ARCHIVED,
                DocumentLifecycleStatus.FAILED,
            },
            DocumentLifecycleStatus.ARCHIVED: set(),
            DocumentLifecycleStatus.FAILED: set(),
        }

    def can_transition(
        self, current: DocumentLifecycleStatus, target: DocumentLifecycleStatus
    ) -> bool:
        """Return whether a transition is valid.

        Args:
            current: The document's current status.
            target: The status the caller wants to transition to.

        Returns:
            ``True`` if the transition appears in :pyattr:`allowed` for
            ``current``, ``False`` otherwise. Unknown ``current`` values
            fall through ``dict.get`` with a default empty set, so the
            answer is ``False`` rather than an exception.
        """
        return target in self.allowed.get(current, set())


class RbacGuard:
    """Per-user RBAC checks.

    The guard is cheap to construct and stateless after initialisation,
    so callers can hold one alongside a request scope without
    coordinating teardown.

    Attributes:
        user: The :class:`UserPrincipal` (or any duck-typed object
            with ``is_admin`` and ``allowed_companies``).
    """

    def __init__(self, user: Any) -> None:
        """Store the user reference.

        Args:
            user: The :class:`UserPrincipal` (or duck-typed object)
                whose RBAC attributes drive the guard.
        """
        self.user = user

    def company_filter(self) -> dict[str, list[str]]:
        """Return the canonical company metadata filter for this user.

        Admin users receive ``{}``, which means no company
        restriction. Every non-admin receives a company filter,
        including ``{"company": []}`` for an empty allow-list,
        which matches no records.

        Returns:
            The metadata filter dict understood by the in-memory
            vector store.
        """
        user = self.user
        if getattr(user, "is_admin", False):
            return {}
        return {"company": list(getattr(user, "allowed_companies", []) or [])}

    def can_access(self, company: str) -> bool:
        """Return whether this user may access ``company``.

        Args:
            company: The tenant (company) string to test.

        Returns:
            ``True`` when the user is an admin or when ``company``
            appears in ``user.allowed_companies``.
        """
        user = self.user
        if getattr(user, "is_admin", False):
            return True
        return company in (getattr(user, "allowed_companies", []) or [])


def allowed_company_filter(user: Any) -> dict[str, list[str]]:
    """Return the canonical company metadata filter for ``user``.

    Args:
        user: The :class:`UserPrincipal`.

    Returns:
        The filter dict (see :meth:`RbacGuard.company_filter`).
    """
    return RbacGuard(user).company_filter()


def can_access_company(user: Any, company: str) -> bool:
    """Return whether ``user`` may access documents scoped to ``company``.

    Args:
        user: The :class:`UserPrincipal`.
        company: The tenant string.

    Returns:
        ``True`` when the user is an admin or ``company`` is in
        ``user.allowed_companies``.
    """
    return RbacGuard(user).can_access(company)