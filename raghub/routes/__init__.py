"""HTTP routes and exception handlers for the RAGHub API.

Re-exports every public symbol from the :mod:`raghub.routes._routes`
submodule so existing imports (e.g. ``from raghub.routes import
RouteGroup``) keep working without modification.
"""

from raghub.routes._limits import (
    check_size,
    content_length,
    enforce_limit,
)
from raghub.routes._routes import (
    AdminRoute,
    AuthRoute,
    DocumentRoute,
    Exceptions,
    FeedbackAggregateResponse,
    FeedbackRoute,
    FeedbackSubmission,
    HealthRoute,
    PreferencesPatch,
    PreferencesResponse,
    QueryRoute,
    QueryRequest,
    PreferenceRoute,
    RouteGroup,
    has_flags,
    user_store_or_raise,
)

__all__ = [
    "AdminRoute",
    "AuthRoute",
    "DocumentRoute",
    "Exceptions",
    "FeedbackAggregateResponse",
    "FeedbackRoute",
    "FeedbackSubmission",
    "HealthRoute",
    "PreferencesPatch",
    "PreferencesResponse",
    "QueryRequest",
    "QueryRoute",
    "PreferenceRoute",
    "RouteGroup",
    "check_size",
    "content_length",
    "enforce_limit",
    "has_flags",
    "user_store_or_raise",
]
