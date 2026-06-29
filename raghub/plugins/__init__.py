"""Plugin system.

Plugins can register converters, chunkers, embedders, vector stores,
retrievers, rerankers, generators, telemetry providers, and
evaluators. The framework discovers plugins via entry points
(``group="raghub.plugins"``) and via explicit registration through
:class:`PluginRegistry`.
"""
