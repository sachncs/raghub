"""Sliding-window conversation history manager with token-aware trimming.

A long-running chat session can accumulate arbitrarily many turns; if we
forward them all to the LLM we eventually blow past the model's context
window. This module provides :class:`SlidingWindowManager`, a small utility
that keeps the most recent turns whose combined token count fits inside a
configured budget.

Algorithm:

* Iterate the history **from newest to oldest**.
* Accumulate each turn's question + answer token count plus a fixed per-turn
  overhead constant (currently ``10`` tokens, an empirical estimate of the
  role/separator markup).
* Stop as soon as the next turn would push the running total past
  ``max_tokens``; everything older is discarded.
* Reverse the surviving window back into chronological order.

The intent is a simple, predictable, and dependency-light trim that mirrors
the well-known "sliding window of context" pattern from the LLM literature.
For stricter accounting you can swap :meth:`count_tokens` with a model-specific
counter, or replace the entire class with a summarising compressor.
"""

from __future__ import annotations

from typing import Any

from raghub.models import ConversationTurn


class SlidingWindowManager:
    """Trim a :class:`ConversationTurn` history to fit within a token budget.

    The manager optionally uses ``tiktoken`` (``cl100k_base``) for accurate
    token counting. If ``tiktoken`` is unavailable at construction time
    (missing dependency or sandbox restriction) the manager falls back to a
    whitespace word-count approximation, which is significantly faster but
    less precise.

    Attributes:
        max_tokens: Total token budget for the returned window.
        enc: A ``tiktoken`` encoding handle, or ``None`` if unavailable.

    Thread safety:
        Instances are immutable after ``__init__`` and therefore safe to share
        across asyncio tasks in a single process. Do not mutate
        :pyattr:`enc` from concurrent code if you ever swap it for a
        non-thread-safe encoder.
    """

    def __init__(self, max_tokens: int = 2048) -> None:
        """Initialise the manager and try to load the ``cl100k_base`` encoder.

        Args:
            max_tokens: Maximum number of tokens the trimmed window may
                contain. Default 2048, well under the context window of most
                4k-class chat models but conservative enough to leave room
                for the system prompt and retrieved context.

        Note:
            If ``tiktoken`` cannot be imported (e.g. minimal CI images,
            offline installs) the encoder falls back to ``None`` and the
            manager silently switches to the whitespace approximation.
            Downstream callers can detect this via the public :pyattr:`enc`
            attribute if they need to log a warning.
        """
        self.max_tokens = max_tokens
        self.enc: Any = None
        # ``tiktoken`` is an optional dependency: importing it should not
        # crash the service if the wheel is unavailable. We deliberately
        # catch a broad ``Exception`` because tiktoken raises a variety of
        # subclasses on missing data files.
        try:
            import tiktoken

            self.enc = tiktoken.get_encoding("cl100k_base")
        except Exception:
            # Fall back to word-count approximation. The trim still works,
            # it just over-estimates token counts by ~1.3x on average.
            pass

    def counttokenize(self, text: str) -> int:
        """Count tokens in a single string.

        Args:
            text: The string to count.

        Returns:
            The exact token count when ``tiktoken`` is available, otherwise
            the number of whitespace-separated words.

        Note:
            Whitespace counting systematically over-counts for languages that
            don't use Latin word boundaries. For non-English deployments,
            install ``tiktoken`` to get accurate counts.
        """
        if self.enc:
            return len(self.enc.encode(text))
        # Whitespace word count is a cheap proxy: usually within 1.3x of the
        # real token count for English. Document this trade-off explicitly.
        return len(text.split())

    def trim(self, history: list[ConversationTurn]) -> list[ConversationTurn]:
        """Return the newest contiguous slice of ``history`` that fits the budget.

        The function iterates **in reverse** so we can stop the moment the
        budget is exhausted and never revisit already-considered older
        turns. Surviving turns are re-inserted at index ``0`` to preserve
        chronological order in the returned list.

        Args:
            history: The full conversation history, oldest turn first.

        Returns:
            A new list containing the newest turns whose summed tokens
            (question + answer + 10-token overhead) is at most
            :pyattr:`max_tokens`. Returns an empty list if even a single
            turn exceeds the budget. The input list is not mutated.
        """
        total = 0
        trimmed: list[ConversationTurn] = []
        # Reverse iteration: walk newest -> oldest. ``insert(0, ...)`` keeps
        # the chronological order in the output list, but at O(n) cost per
        # insert. For typical history depths (< 100 turns) this is fine;
        # pathological histories should pre-filter before calling trim.
        for turn in reversed(history):
            # ``+ 10`` accounts for the per-turn role markers, separators,
            # and chat-template overhead that surround each Q/A pair. The
            # exact number depends on the model; 10 is an empirical average
            # for the cl100k/o200k families.
            turn_tokens = self.counttokenize(turn.question) + self.counttokenize(turn.answer) + 10
            if total + turn_tokens > self.max_tokens:
                # Adding this turn would exceed the budget: stop and return
                # what we have. Older turns are discarded silently.
                break
            trimmed.insert(0, turn)
            total += turn_tokens
        return trimmed
