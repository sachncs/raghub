"""Structured-output adapters.

The default provider uses Instructor to coerce LLM output into typed
Pydantic models. The provider is optional; when Instructor isn't
installed, :class:`InstructorStructuredOutputProvider` raises
:class:`ConfigurationError` on construction.

Public re-exports:

* :class:`InstructorStructuredOutputProvider` — the v1+ Instructor
  adapter (uses the documented ``from_provider("litellm/<model>")``
  entry point).
"""

from raghub.structured.instructor import InstructorStructuredOutputProvider

__all__ = ["InstructorStructuredOutputProvider"]
