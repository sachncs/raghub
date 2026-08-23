"""Tests for ``raghub.types`` (JSONValue recursive alias)."""

from __future__ import annotations

from raghub.types import JSONValue


def test_jsonvalue_alias_accepts_scalar_types() -> None:
    """``JSONValue`` resolves to a type usable for scalars (str/int/float/bool/None)."""

    scalars: list[JSONValue] = ["s", 1, 1.5, True, None]
    for s in scalars:
        assert isinstance(s, str | int | float | bool | type(None))


def test_jsonvalue_alias_accepts_nested_lists_and_dicts() -> None:
    """A value deeply nested as lists/dicts is constructible under JSONValue."""

    nested: JSONValue = {
        "users": [
            {"name": "alice", "tags": ["admin", "finance"], "active": True},
            {"name": "bob", "tags": [], "active": False, "score": None},
        ],
        "meta": {"count": 2, "limit": 1.0},
    }
    assert nested["users"][0]["name"] == "alice"  # type: ignore[index]
    assert nested["users"][1]["score"] is None  # type: ignore[index]


def test_jsonvalue_alias_used_as_typed_dict_value() -> None:
    """Functions declared ``**kwargs: JSONValue`` accept the full JSON grammar."""

    def echo(**kwargs: JSONValue) -> JSONValue:
        return kwargs["payload"]

    result = echo(payload={"x": [1, 2, 3], "y": None})
    assert result == {"x": [1, 2, 3], "y": None}
