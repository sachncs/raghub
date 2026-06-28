"""Role-based access control (RBAC) helpers for query authorisation.

The framework models multi-tenant document isolation by tagging every
chunk with a ``company`` string and every user with a set of
``allowed_companies``. Retrieval is gated by translating a user's
allow-list into a metadata filter expression that the vector store pushes
into its native query language. An admin user (or a user with an empty
allow-list) is treated as "sees everything" and emits an empty filter.

Security note: the produced filter string is concatenated directly into
the store's filter grammar. The store implementations
(:class:`raghub.vectorstore.memory.InMemoryVectorStore`,
:class:`raghub.vectorstore.zvec.ZvecVectorStore`) parse the string with
simple, non-quote-aware extractors, so we single-quote the company names
when assembling the ``IN`` clause. **Callers must not pass company names
containing embedded single quotes**, or filter injection becomes possible.
In practice company names are validated at user-provisioning time; this is
the trust boundary.
"""

from __future__ import annotations

from raghub.models import UserPrincipal


def allowed_company_filter(user: UserPrincipal) -> str:
    """Return a metadata filter expression that restricts results to ``user``.

    The result is a fragment of the vector store's filter DSL, not a
    boolean: an empty string means "no filter" (admin or empty allow-list
    case). A non-empty result looks like ``"company IN ('acme', 'globex')"``.

    Args:
        user: The principal making the request.

    Returns:
        A filter string suitable for passing to
        :meth:`raghub.interfaces.vectorstore.VectorStore.search` as
        ``metadata_filter``. An empty string disables filtering.

    Note:
        The function trusts that company names do not contain single
        quotes. The user-provisioning flow is responsible for sanitising
        them; see the module docstring for the trust boundary.

    Example:
        >>> from raghub.models import UserPrincipal
        >>> u = UserPrincipal(user_id="u1", email="u@x", is_admin=False, allowed_companies=("acme",))
        >>> allowed_company_filter(u)
        "company IN ('acme')"
        >>> allowed_company_filter(UserPrincipal(user_id="a", email="a@x", is_admin=True, allowed_companies=()))
        ''
    """
    # Admin bypass: an admin can see every company. An empty allow-list
    # is also treated as unrestricted (legacy behaviour for unscoped users).
    if user.is_admin or not user.allowed_companies:
        return ""
    # Single-quote each name to keep the filter parser happy. This is safe
    # only if company names cannot contain single quotes — see the
    # module-level security note.
    quoted = ", ".join(f"'{company}'" for company in user.allowed_companies)
    return f"company IN ({quoted})"


def can_access_company(user: UserPrincipal, company: str) -> bool:
    """Return whether ``user`` may access documents scoped to ``company``.

    Args:
        user: The principal performing the access check.
        company: The company identifier of the resource.

    Returns:
        ``True`` if the user is an admin or if ``company`` is in their
        allow-list; ``False`` otherwise.

    Note:
        This helper is used for in-process decisions (e.g. upload-time
        authorisation). For query-time enforcement the vector store
        applies :func:`allowed_company_filter` directly.
    """
    return user.is_admin or company in user.allowed_companies

