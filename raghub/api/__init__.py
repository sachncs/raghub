"""FastAPI reference application.

This package exposes the HTTP layer of the framework: the
:func:`create_app` factory and the :class:`RouteGroup` that owns the
v1, admin, and preferences routers.

Helpers used by the route handlers — bearer parsing, application-facade
lookup, SSE framing, response construction, secret redaction, and the
token-bucket rate limiter — live in :mod:`raghub.api.helper` as a
single class per concern.

The factory pattern lets embedders build multiple applications per
process (e.g. one per tenant profile) without conflicting router
state.
"""
