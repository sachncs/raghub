"""Chunker implementations.

Adapters turn a :class:`KnowledgeBundle` (or raw text) into a list of
:class:`raghub.models.Chunk` records. The default is a built-in
word-window chunker; Chonkie provides a higher-quality default for
production.
"""
