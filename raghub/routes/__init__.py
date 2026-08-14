"""HTTP routes and exception handlers for the RAGHub API.

Re-exports every public symbol from :mod:`raghub.routes.limits` and
:mod:`raghub.routes.routes` so existing imports
(e.g. ``from raghub.routes import RouteGroup``) keep working.
"""

from raghub.routes.limits import (
    check_size,
    content_length,
    enforce_limit,
)
from raghub.routes.routes import (
    AdminRoute,
    AuthRoute,
    DocumentRoute,
    Exceptions,
    FeedbackAggregateResponse,
    FeedbackRoute,
    FeedbackSubmission,
    HealthRoute,
    PreferenceRoute,
    PreferencesPatch,
    PreferencesResponse,
    QueryRequest,
    QueryRoute,
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
    "PreferenceRoute",
    "PreferencesPatch",
    "PreferencesResponse",
    "QueryRequest",
    "QueryRoute",
    "RouteGroup",
    "check_size",
    "content_length",
    "enforce_limit",
    "has_flags",
    "user_store_or_raise",
]
