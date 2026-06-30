"""Document converters.

Adapters that turn source bytes into canonical :class:`KnowledgeBundle`
objects. The default is :class:`MarkerConverter` for PDF inputs and
:class:`PlainTextConverter` for plain-text / Markdown / unknown
inputs.
"""
