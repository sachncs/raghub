"""FastAPI reference application.

This package bundles the HTTP layer of the framework: the
:func:`create_app` factory, the admin router, dependency providers,
rate-limiting middleware, and re-exports of the transport schemas.

The factory pattern lets embedders build multiple applications per
process (e.g. one per tenant profile) without conflicting router
state.
"""

from raghub.api.app import create_app
