"""Document converters.

Adapters that turn source bytes into canonical :class:`KnowledgeBundle`
objects. The default is :class:`MarkerConverter` for PDF inputs and
:class:`PlainTextConverter` for plain-text / Markdown / unknown
inputs.
"""

from .directory import convert_path
from .markdown import normalise_markdown
from .marker import MarkerConverter, looks_like_pdf
from .plaintext import PlainTextConverter

__all__ = [
    "MarkerConverter",
    "PlainTextConverter",
    "convert_path",
    "looks_like_pdf",
    "normalise_markdown",
]
