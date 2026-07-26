"""Resolve the effective tool/agent config for a single query.

Precedence (highest wins):

1. Request-level overrides (``QueryRequest.tools_enabled`` etc.)
2. Session-level overrides (Phase 1.12 — conversation metadata)
3. User preferences (``UserPrincipal.tool_settings``)
4. Global defaults from :class:`Settings`

The resolver is pure: it never reads a database or calls the LLM. All
callers feed it the three override layers they already loaded.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from raghub.config import Settings

ALLOWED_TOOLS = frozenset(
    {
        "vector_search",
        "keyword_search",
        "hybrid_search",
        "summary_search",
        "graph_search",
        "web_search",
        "date_today",
    }
)

ALLOWED_RERANKERS = frozenset({"none", "cohere", "bge", "llm", "cascade"})

ALLOWED_TRANSFORMS = frozenset({"hyde", "multi_query", "step_back", "decompose"})


@dataclass(frozen=True)
class ResolvedConfig:
    """Effective settings after precedence resolution.

    Attributes:
        agent_enabled: Whether the agent loop should run this query.
        tools_enabled: Set of tool names the agent may invoke.
        reranker: Resolved reranker provider (``"none"`` for identity).
        long_context_pass: Whether to invoke the long-context second pass.
        query_transforms: Ordered list of transforms to run before retrieval.
        max_steps: Planner step cap.
    """

    agent_enabled: bool
    tools_enabled: frozenset[str] = field(default_factory=frozenset)
    reranker: str = "none"
    long_context_pass: bool = False
    query_transforms: tuple[str, ...] = ()
    max_steps: int = 8

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-friendly dict (sets/tuples → lists)."""
        return {
            "agent_enabled": self.agent_enabled,
            "tools_enabled": sorted(self.tools_enabled),
            "reranker": self.reranker,
            "long_context_pass": self.long_context_pass,
            "query_transforms": list(self.query_transforms),
            "max_steps": self.max_steps,
        }


def coerce_tools(value: Any) -> set[str]:
    """Coerce a value to a validated set of tool names.

    Accepts ``None`` (returns empty set), ``list[str]``, or ``set[str]``.
    Unknown names are dropped — invalid input must not crash startup.

    Args:
        value: Raw value from one of the override layers.

    Returns:
        The validated set of tool names.
    """
    if value is None:
        return set()
    if isinstance(value, (list, tuple, set, frozenset)):
        return {str(v) for v in value if isinstance(v, str) and v in ALLOWED_TOOLS}
    return set()


def coerce_transforms(value: Any) -> tuple[str, ...]:
    """Coerce to a validated, de-duplicated, order-preserving list.

    Args:
        value: Raw value from one of the override layers.

    Returns:
        The validated tuple of transform names.
    """
    if not value:
        return ()
    seen: dict[str, None] = {}
    for v in value:
        if isinstance(v, str) and v in ALLOWED_TRANSFORMS and v not in seen:
            seen[v] = None
    return tuple(seen.keys())


def coerce_reranker(value: Any) -> str:
    """Return ``value`` if it's a known reranker; ``"none"`` otherwise.

    Args:
        value: Raw reranker name.

    Returns:
        The validated reranker name, or ``"none"`` when the input
        is missing or unknown.
    """
    if isinstance(value, str) and value in ALLOWED_RERANKERS:
        return value
    return "none"


def coerce_max_steps(value: Any, fallback: int) -> int:
    """Return ``value`` as an int clamped to ``[1, 64]``; fallback on bad input.

    Args:
        value: Raw max-step value (may be ``None`` or a non-int).
        fallback: Returned when ``value`` is missing or unparseable.

    Returns:
        A clamped integer in ``[1, 64]``.
    """
    try:
        n = int(value)
    except (TypeError, ValueError):
        return fallback
    return max(1, min(n, 64))


def pick_value(layers: tuple[dict[str, Any] | None, ...], key: str) -> Any:
    """Return the first non-None value across ``layers``.

    Order: first element wins (request > session > user). ``None``
    skips a layer so ``agent=False`` in the session doesn't get
    overridden by ``agent=True`` at the global default.

    Args:
        layers: Tuple of dicts ordered highest-precedence first.
        key: The preference key to look up.

    Returns:
        The first present, non-None value, or ``None`` when every
        layer leaves it unset.
    """
    for layer in layers:
        if layer is not None and key in layer and layer[key] is not None:
            return layer[key]
    return None


def resolve(
    *,
    request_overrides: dict[str, Any] | None,
    session_overrides: dict[str, Any] | None,
    user_prefs: dict[str, Any] | None,
    settings: Settings,
) -> ResolvedConfig:
    """Compute the effective config for one query.

    Args:
        request_overrides: Fields extracted from the inbound
            :class:`QueryRequest` (``None`` values mean "no override").
        session_overrides: Session-scoped overrides (Phase 1.12).
        user_prefs: User-scoped overrides (Phase 1.10/1.11).
        settings: The application settings carrying global defaults.

    Returns:
        A :class:`ResolvedConfig` ready to hand to the query pipeline.
    """
    request = request_overrides or {}
    session = session_overrides or {}
    user = user_prefs or {}
    layers = (request, session, user)

    # --- Agent enabled: request > session > user > global ----------
    req_agent = request.get("agent")
    if req_agent is not None:
        agent_enabled = bool(req_agent)
    elif "agent_enabled" in session:
        agent_enabled = bool(session["agent_enabled"])
    else:
        agent_enabled = bool(user.get("agent_enabled", settings.agent.enabled))

    # --- Tools: union of explicit names + shortcuts ----------------
    tools = coerce_tools(request.get("tools_enabled"))
    if not tools:
        tools = coerce_tools(session.get("tools_enabled"))
    if not tools:
        tools = coerce_tools(user.get("tools_enabled"))
    if request.get("web") is True:
        tools = tools | {"web_search"}
    if request.get("graph") is True:
        tools = tools | {"graph_search"}
    if request.get("summaries") is True:
        tools = tools | {"summary_search"}

    # --- Reranker --------------------------------------------------
    requested_reranker = pick_value(layers, "reranker")
    reranker = coerce_reranker(
        requested_reranker
        if requested_reranker is not None
        else settings.reranker.provider
    )

    # --- Long-context pass -----------------------------------------
    requested_lcp = pick_value(layers, "long_context_pass")
    long_context_pass = bool(
        requested_lcp
        if requested_lcp is not None
        else settings.long_context_pass.enabled
    )

    # --- Query transforms ------------------------------------------
    transforms = coerce_transforms(request.get("query_transforms"))
    if not transforms:
        transforms = coerce_transforms(session.get("query_transforms"))
    if not transforms:
        transforms = coerce_transforms(user.get("query_transforms"))
    if not transforms:
        transforms = tuple(settings.query_transforms.enabled)

    # --- Max steps -------------------------------------------------
    raw_steps = pick_value(layers, "max_steps")
    if raw_steps is None:
        raw_steps = settings.agent.max_steps
    max_steps = coerce_max_steps(raw_steps, settings.agent.max_steps)

    return ResolvedConfig(
        agent_enabled=agent_enabled,
        tools_enabled=frozenset(tools),
        reranker=reranker,
        long_context_pass=long_context_pass,
        query_transforms=transforms,
        max_steps=max_steps,
    )


__all__ = [
    "ALLOWED_RERANKERS",
    "ALLOWED_TOOLS",
    "ALLOWED_TRANSFORMS",
    "ResolvedConfig",
    "coerce_max_steps",
    "coerce_reranker",
    "coerce_tools",
    "coerce_transforms",
    "pick_value",
    "resolve",
]