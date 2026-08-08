"""Query pipeline helpers — small, pure functions extracted from :class:`QueryPipeline`.

This module hosts the ``QueryPipeline`` helpers that do not need
``self`` binding. Each function takes its collaborators (cache,
conversation store, generator, telemetry, …) explicitly so it can be
unit-tested in isolation and re-used without instantiating the
pipeline.

Why split?
- Keeps :mod:`raghub.pipeline.query` focused on orchestration.
- Helpers depend only on :mod:`raghub.models` and the standard
  library — no circular-import risk with the pipeline package.
- Each helper has one responsibility and a single, stable signature.

The functions exposed here cover:

* :func:`user_filter` — derive the RBAC metadata filter from a user.
* :func:`scope_triple` — build the cache scope tuple from a user.
* :func:`triggers_agent` — check whether ``resolved_config`` activates
  the agent loop.
* :func:`annotate_span` / :func:`annotate_stream` — stamp user /
  session attributes on the active telemetry span.
* :func:`filter_hits` — drop hits that fail the per-user metadata
  filter.
* :func:`build_citations` — convert :class:`Hit` objects into the
  facade's :class:`Citation` shape.
* :func:`record_turn` / :func:`record_streamed` — append a turn to the
  conversation store, with a warning log when the answer is empty.
* :func:`record_generate_tokens` — forward LLM token usage to
  telemetry (async path used by :meth:`QueryPipeline.generate_answer`).
* :func:`record_stream_tokens` — forward streaming token usage to
  telemetry (sync path used by :meth:`QueryPipeline.stream`).
* :func:`cache_lookup` / :func:`cache_persist` — read and write a
  cached :class:`Pipeline`.
"""

from __future__ import annotations
from loguru import logger

import inspect
from typing import Any

from raghub.models import Citation, Hit, Pipeline, Turn



def user_filter(user: Any) -> dict[str, Any] | str:
    """Derive a metadata filter for the vector store from a user.

    Security-sensitive: a non-admin user with an empty
    ``allowed_companies`` list must not be allowed to see every
    chunk. We return a filter that matches NOTHING so the
    downstream search returns an empty hit list instead of
    silently returning the full corpus. Admin users and the
    anonymous case keep the prior behaviour of ``""`` (no
    filter).
    """
    if user is None:
        return ""
    if getattr(user, "is_admin", False):
        return ""
    companies = list(getattr(user, "allowed_companies", []) or [])
    if not companies:
        return {"company": "__no_companies_allowed__"}
    return {"company": companies}


def scope_triple(user: Any) -> tuple[bool, tuple[str, ...], tuple[str, ...]]:
    """Build the cache scope tuple for ``user``."""
    return (
        bool(getattr(user, "is_admin", False)),
        tuple(sorted(str(value) for value in getattr(user, "allowed_companies", []) or [])),
        tuple(sorted(str(value) for value in getattr(user, "allowed_groups", []) or [])),
    )


def triggers_agent(inputs: dict[str, Any]) -> bool:
    """Return whether ``resolved_config`` activates the agent loop.

    Args:
        inputs: The resolved ``inputs`` mapping for the request.
            Looks for ``resolved_config`` and inspects it for
            ``agent_enabled`` / ``tools_enabled`` keys.

    Returns:
        ``True`` when either flag is set in ``resolved_config``;
        ``False`` otherwise (including when ``resolved_config``
        is absent or not a dict).

    """
    record_overrides = inputs.get("resolved_config")
    if not isinstance(record_overrides, dict):
        return False
    return bool(record_overrides.get("agent_enabled") or record_overrides.get("tools_enabled"))


def annotate_span(
    span: Any,
    user: Any | None,
    session_id: str | None,
) -> None:
    """Stamp user / session attributes on the active query span."""
    if user is not None:
        email = getattr(user, "email", None)
        if email:
            span.set_attribute("user_id", email)
    if session_id:
        span.set_attribute("session_id", session_id)


def annotate_stream(
    span: Any,
    user: Any | None,
    session_id: str | None,
) -> None:
    """Stamp user / session attributes on the active stream span."""
    if user is not None and getattr(user, "email", None):
        span.set_attribute("user_id", user.email)
    if session_id:
        span.set_attribute("session_id", session_id)


def filter_hits(
    hits: list[Hit],
    user_filter: dict[str, Any] | str,
) -> list[Hit]:
    """Drop hits that fail the per-user metadata filter."""
    if not (isinstance(user_filter, dict) and user_filter):
        return hits
    return [
        h for h in hits if all(getattr(h.chunk, k, None) == v for k, v in user_filter.items())
    ]


def build_citations(hits: list[Hit]) -> list[Citation]:
    """Convert ``Hit`` objects into the facade's ``Citation`` shape."""
    return [
        Citation(
            chunk=h.chunk,
            document_id=h.chunk.document_id,
            version=h.chunk.version,
            page=h.chunk.page,
            section=h.chunk.section,
            quote=h.chunk.text,
            score=h.score,
            source_uri=h.chunk.source_location or h.chunk.document_id,
        )
        for h in hits
    ]


def record_turn(
    conversation_store: Any,
    record: bool,
    session_id: str | None,
    question: str,
    answer: Any,
) -> None:
    """Append a turn to the conversation store when conditions allow.

    Empty answers (the LLM returned ``""``) are still dropped but
    now logged so the silent-data-loss case is at least observable.
    """
    if not (record and session_id and answer):
        if record and session_id and not answer:
            logger.warning(f"dropped turn: empty answer, session_id={session_id}")
        return
    conversation_store.append(
        session_id,
        Turn(question=question, answer=str(answer)),
    )


def record_streamed(
    conversation_store: Any,
    session_id: str | None,
    question: str,
    collected: list[str],
) -> None:
    """Append the streamed answer to the conversation store.

    Empty streamed answers (the LLM yielded nothing) are still
    dropped but now logged so the silent-data-loss case is at
    least observable.
    """
    if not (session_id and collected):
        if session_id and not collected:
            logger.warning(f"dropped turn: empty answer, session_id={session_id}")
        return
    conversation_store.append(
        session_id,
        Turn(
            question=question,
            answer="".join(collected),
        ),
    )


async def record_generate_tokens(generator: Any, telemetry: Any) -> None:
    """Forward LLM token usage to the telemetry provider when available."""
    record_tokens = getattr(generator, "record_tokens", None)
    if not callable(record_tokens):
        return
    tokens = record_tokens()
    if inspect.isawaitable(tokens):
        tokens = await tokens
    if not isinstance(tokens, dict) or not tokens:
        return
    telemetry.record_tokens(
        "query.generate",
        prompt_tokens=int(tokens.get("prompt", 0)),
        completion_tokens=int(tokens.get("completion", 0)),
        model=str(tokens.get("model", "")),
    )


def record_stream_tokens(generator: Any, telemetry: Any) -> None:
    """Forward streaming token usage to the telemetry provider."""
    record_tokens = getattr(generator, "record_tokens", None)
    if not callable(record_tokens):
        return
    tokens = record_tokens()
    if inspect.isawaitable(tokens):
        return
    if not isinstance(tokens, dict) or not tokens:
        return
    with telemetry.span("query.tokens") as tok_span:
        tok_span.set_attribute("prompt_tokens", int(tokens.get("prompt", 0)))
        tok_span.set_attribute("completion_tokens", int(tokens.get("completion", 0)))
    telemetry.record_tokens(
        "query.stream",
        prompt_tokens=int(tokens.get("prompt", 0)),
        completion_tokens=int(tokens.get("completion", 0)),
        model=str(tokens.get("model", "")),
    )


def cache_lookup(cache: Any, ctx: Any) -> Pipeline | None:
    """Return a cached ``Pipeline`` for the request, or ``None``."""
    if cache is None:
        return None
    cached = cache.get(
        ctx.question,
        ctx.user_id,
        ctx.user_filter,
        top_k=ctx.top_k,
        response_model=ctx.response_model,
        session_id=ctx.session_id,
        history=ctx.history,
        scope=ctx.scope,
    )
    return cached if isinstance(cached, Pipeline) else None


def cache_persist(cache: Any, result: Pipeline, ctx: Any) -> None:
    """Persist the pipeline result in the cache when configured."""
    if cache is None:
        return
    cache.set(
        ctx.question,
        ctx.user_id,
        ctx.user_filter,
        result,
        top_k=ctx.top_k,
        response_model=ctx.response_model,
        session_id=ctx.session_id,
        history=ctx.history,
        scope=ctx.scope,
    )


__all__ = [
    "annotate_span",
    "annotate_stream",
    "build_citations",
    "cache_lookup",
    "cache_persist",
    "filter_hits",
    "record_generate_tokens",
    "record_stream_tokens",
    "record_streamed",
    "record_turn",
    "scope_triple",
    "triggers_agent",
    "user_filter",
]