"""Pure helpers for :mod:`raghub.knowledge.graph`.

Keeps the :class:`GraphIndex` module under its file-budget by
offloading the stateless utilities.
"""

from __future__ import annotations

import asyncio
import json
import re
from typing import Any, TYPE_CHECKING

from raghub.runtime import capture

if TYPE_CHECKING:
    from raghub.knowledge.graph import GraphIndex
    from raghub.models import Chunk

MIN_TOKEN_LENGTH = 2


def extract_json_object(raw: str) -> dict[str, Any] | None:
    """Pull the first balanced JSON object out of a string."""
    if not raw:
        return None
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", raw, re.DOTALL)
    candidate = fenced.group(1) if fenced else raw
    start = candidate.find("{")
    if start == -1:
        return None
    depth = 0
    end = -1
    for idx in range(start, len(candidate)):
        ch = candidate[idx]
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                end = idx + 1
                break
    if end == -1:
        return None
    parsed, _ = capture(json.loads, candidate[start:end])
    return parsed if isinstance(parsed, dict) else None


def tokenise(text: str) -> set[str]:
    """Lower-case word tokens, dropping words of length ≤ 2."""
    return {token for token in re.findall(r"\w+", text.lower()) if len(token) > MIN_TOKEN_LENGTH}


def connected_components(graph_like: GraphIndex) -> list[set[str]]:
    """Networkx-free connected components over the graph field."""
    visited: set[str] = set()
    components: list[set[str]] = []
    for node in graph_like.graph:
        if node in visited:
            continue
        stack = [node]
        component: set[str] = set()
        while stack:
            current = stack.pop()
            if current in visited:
                continue
            visited.add(current)
            component.add(current)
            stack.extend(graph_like.graph[current] - visited)
        components.append(component)
    return components


def running_loop_present() -> bool:
    """Return ``True`` when an asyncio loop is running in this thread."""
    _, error = capture(asyncio.get_running_loop)
    return error is None


def run_in_thread(graph_like: GraphIndex, chunks: list[Chunk]) -> None:
    """Run :meth:`GraphIndex.drive_extraction` on a daemon thread and join."""
    import threading

    thread = threading.Thread(
        target=lambda: asyncio.run(graph_like.drive_extraction(chunks)),
        daemon=True,
    )
    thread.start()
    thread.join()


__all__ = [
    "MIN_TOKEN_LENGTH",
    "connected_components",
    "extract_json_object",
    "run_in_thread",
    "running_loop_present",
    "tokenise",
]
