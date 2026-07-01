"""Generation pipeline.

Orchestrates prompt construction, LLM invocation, and citation
attachment. The default is :class:`DefaultGenerator` which wraps any
:class:`raghub.llm.BaseLLMProvider`.
"""

from .generator import DefaultGenerator

__all__ = ["DefaultGenerator"]
