"""ReAct planner prompts + JSON parsing (Phase 7.1).

The agent loop uses a JSON-formatted tool-call embedded in the LLM
response. This is the canonical ReAct pattern: every turn the model
returns a JSON object with one of two shapes:

    {"thought": "...", "final_answer": "..."}                  # done
    {"thought": "...", "action": {"name": "...", "args": {...}}} # call a tool

The :func:`parse_turn` function returns a :class:`PlannerAction`
or :class:`PlannerFinal`; unknown shapes fall back to a final
answer with the raw text so the planner always terminates.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any

SYSTEM_PROMPT = """You are a planner. You solve the user's question by either:
1. Calling a tool — reply with JSON
   {"thought": "...", "action": {"name": "<tool>", "args": {...}}}
2. Producing a final answer — reply with JSON
   {"thought": "...", "final_answer": "..."}

Rules:
- Reply with JSON only. No prose, no markdown fences, no preamble.
- One tool call per turn.
- Call a tool when you need more information than you currently have.
- When you have enough information, produce final_answer.
- Never invent tool names; use only those listed below.
- Never invent chunk ids or facts.

Available tools:
{tool_schemas}
"""

OBSERVATION_PROMPT = """Tool `{name}` returned:
{observation}

Decide your next turn. JSON only.
"""


@dataclass
class PlannerAction:
    """A tool call parsed from an LLM turn.

    Attributes:
        thought: Free-form reasoning the LLM emitted. Stored in the
            :class:`AgentTrace` for observability.
        name: Tool name.
        args: Argument dict — already schema-validated by the
            caller.
    """

    thought: str
    name: str
    args: dict[str, Any]


@dataclass
class PlannerFinal:
    """A final answer parsed from an LLM turn.

    Attributes:
        thought: Free-form reasoning.
        answer: The synthesised answer to surface to the user.
    """

    thought: str
    answer: str


@dataclass
class PlannerParseError:
    """A turn that could not be parsed as an action or final.

    Attributes:
        thought: Empty — no model reasoning available.
        raw: The raw LLM output, for observability.
    """

    thought: str = ""
    raw: str = ""


def extract_json_object(raw: str) -> dict[str, Any] | None:
    """Pull the first balanced JSON object out of a string.

    Args:
        raw: The LLM's raw output.

    Returns:
        The first balanced JSON object, or ``None`` when none can
        be found or parsed.
    """
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
    try:
        return json.loads(candidate[start:end])
    except ValueError:
        return None


def parse_turn(raw: str) -> PlannerAction | PlannerFinal | PlannerParseError:
    """Parse one ReAct turn.

    Args:
        raw: The LLM's raw output for this turn.

    Returns:
        * :class:`PlannerAction` when ``action`` is present.
        * :class:`PlannerFinal` when ``final_answer`` is present.
        * :class:`PlannerParseError` when neither can be parsed;
          the caller treats the error as a final answer whose text
          is the raw LLM output (best-effort fallback).
    """
    obj = extract_json_object(raw or "")
    if obj is None:
        return PlannerParseError(raw=raw or "")
    thought = str(obj.get("thought", "") or "")
    if isinstance(obj.get("final_answer"), str):
        return PlannerFinal(thought=thought, answer=obj["final_answer"])
    action = obj.get("action")
    if isinstance(action, dict):
        name = action.get("name")
        args = action.get("args") or {}
        if isinstance(name, str) and isinstance(args, dict):
            return PlannerAction(thought=thought, name=name, args=args)
    return PlannerParseError(raw=raw or "")


def render_system_prompt(tool_schemas: list[dict[str, Any]]) -> str:
    """Compose the ReAct system prompt for a given tool catalog.

    Args:
        tool_schemas: List of ``{"name": ..., "description": ..., "json_schema": ...}``
            dicts.

    Returns:
        The full system prompt with the catalog embedded.
    """
    if not tool_schemas:
        catalog = "(no tools available — produce final_answer only)"
    else:
        lines: list[str] = []
        for schema in tool_schemas:
            lines.append(f"- {schema['name']}: {schema['description']}")
            if schema.get("json_schema"):
                lines.append(
                    "  args: " + json.dumps(schema["json_schema"], separators=(",", ":"))
                )
        catalog = "\n".join(lines)
    return SYSTEM_PROMPT.replace("{tool_schemas}", catalog)


__all__ = [
    "OBSERVATION_PROMPT",
    "PlannerAction",
    "PlannerFinal",
    "PlannerParseError",
    "render_system_prompt",
]