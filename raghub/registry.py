"""Polymorphic class registry.

The :class:`Registry` mixin turns a base class into a by-name
dispatch table. Concrete subclasses register themselves with the
``@Base.register("name")`` decorator, and callers instantiate them
through :meth:`Registry.get`:

.. code-block:: python

    class Rerank(Registry):
        name: str

        def rerank(self, *, question, hits): ...

    @Rerank.register("identity")
    class Identity(Rerank):
        name = "identity"

        def rerank(self, *, question, hits):
            return list(hits)

    # Construct by name:
    Identity = Rerank.lookup("identity")()

Why a class mixin rather than a free function?

* The base class documents the contract (:class:`Rerank` here is the
  polymorphic type).
* Subclasses automatically belong to their parent's registry; no
  external wiring.
* ``Rerank.names()`` returns the known set for error messages and
  CLI ``--help`` output.

The trade-off: each pluggable base class adds one tiny mixin
inheritance. That's strictly less abstract than the ABC + Protocol
+ factory function combo this replaces.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import ClassVar, TypeVar

T = TypeVar("T")


class Registry:
    """Mixin that turns a base class into a by-name dispatch table.

    Each subclass gets its own ``items`` dict through
    :meth:`__init_subclass__`, so multiple independent registries
    (e.g. ``Rerank`` and ``Transformer``) don't bleed into each
    other.

    Attributes:
        items: Mapping from registered name to concrete subclass. The
            ``@register`` decorator populates this; subclasses
            shouldn't touch it directly.

    """

    items: ClassVar[dict[str, type]] = {}

    def __init_subclass__(cls, **kwargs: object) -> None:
        """Give every subclass its own ``items`` dict."""
        super().__init_subclass__(**kwargs)
        cls.items = {}

    @classmethod
    def register(cls, name: str) -> Callable[[type], type]:
        """Register a concrete subclass under ``name``.

        Args:
            name: Stable identifier (e.g. ``"cohere"``, ``"hyde"``).

        Returns:
            A class decorator that records the subclass and returns it
            unchanged so it can still be used as a normal class.

        """

        def deco(subclass: type) -> type:
            """Record ``subclass`` under ``name`` and return it unchanged."""
            cls.items[name] = subclass
            return subclass

        return deco

    @classmethod
    def lookup(cls, name: str) -> type:
        """Look up a registered subclass by name.

        Args:
            name: The identifier passed to :meth:`register`.

        Returns:
            The concrete subclass (not an instance).

        Raises:
            ValueError: When ``name`` is not registered under ``cls``.

        """
        if name not in cls.items:
            raise ValueError(f"Unknown {cls.__name__}: {name!r}; known: {sorted(cls.items)}")
        return cls.items[name]

    @classmethod
    def names(cls) -> list[str]:
        """Return the registered names in sorted order."""
        return sorted(cls.items)


__all__ = ["Registry"]
