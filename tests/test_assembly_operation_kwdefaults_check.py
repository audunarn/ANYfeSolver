"""The assembly-operation authority rejects exactly what its original check rejected.

The per-guard kwdefaults identity check is evaluated with C-level iteration for
dictionaries with exact ``str`` keys and keeps the original generator for any
other key type.  These tests pin the accept/reject outcome (and the exception
type) for in-place edits of both captured kwdefaults dictionaries, including a
hash-colliding key whose ``__eq__`` answers differently per probe.
"""

from __future__ import annotations

from typing import Any, Callable, Iterator

import pytest

import anyfesolver.matrix_assembly as matrix_assembly_module

REQUIRE = matrix_assembly_module._require_exact_assembly_operation_metadata
PUBLISHED = matrix_assembly_module._assemble_element_matrix_under_lease
IMPLEMENTATION = matrix_assembly_module._assemble_element_matrix_under_lease_impl


def _restoring(target: dict[str, Any]) -> Callable[[], None]:
    snapshot = list(target.items())

    def restore() -> None:
        target.clear()
        target.update(snapshot)

    return restore


@pytest.fixture(params=("published", "implementation"))
def kwdefaults(request: pytest.FixtureRequest) -> Iterator[dict[str, Any]]:
    function = PUBLISHED if request.param == "published" else IMPLEMENTATION
    target = function.__kwdefaults__
    assert type(target) is dict and target
    restore = _restoring(target)
    try:
        yield target
    finally:
        restore()
    REQUIRE()


def test_unchanged_authority_is_accepted() -> None:
    REQUIRE()


def test_in_place_value_replacement_is_rejected(kwdefaults: dict[str, Any]) -> None:
    name = next(iter(kwdefaults))
    kwdefaults[name] = object()
    with pytest.raises(ValueError, match="operation authority changed"):
        REQUIRE()


class _EqualToEverything:
    def __eq__(self, other: object) -> bool:
        return True

    __hash__ = object.__hash__


def test_equal_but_not_identical_value_is_rejected(kwdefaults: dict[str, Any]) -> None:
    # Values are compared by identity, never by equality.
    name = next(iter(kwdefaults))
    kwdefaults[name] = _EqualToEverything()
    with pytest.raises(ValueError, match="operation authority changed"):
        REQUIRE()


def test_renamed_key_with_same_length_is_rejected(kwdefaults: dict[str, Any]) -> None:
    name = next(iter(kwdefaults))
    value = kwdefaults.pop(name)
    kwdefaults[name + "_renamed"] = value
    with pytest.raises(ValueError, match="operation authority changed"):
        REQUIRE()


def test_reinserted_key_in_new_order_is_accepted(kwdefaults: dict[str, Any]) -> None:
    # The original check is order-insensitive: same names, same objects.
    name = next(iter(kwdefaults))
    kwdefaults[name] = kwdefaults.pop(name)
    REQUIRE()


class _CollidingKey:
    """Hash-colliding stand-in whose equality answers from a script."""

    def __init__(self, name: str, answers: list[bool]) -> None:
        self.name = name
        self.answers = answers
        self.calls = 0

    def __hash__(self) -> int:
        return hash(self.name)

    def __eq__(self, other: object) -> bool:
        if other != self.name:
            return False
        self.calls += 1
        return self.answers.pop(0) if self.answers else False


def test_colliding_key_that_flips_keeps_original_key_error(
    kwdefaults: dict[str, Any],
) -> None:
    # Original generator: ``name in d`` (True) then ``d[name]`` (False ->
    # KeyError).  The non-str key keeps that exact path and exception.
    name = next(iter(kwdefaults))
    value = kwdefaults.pop(name)
    key = _CollidingKey(name, [True, False])
    kwdefaults[key] = value  # type: ignore[index]
    with pytest.raises(KeyError):
        REQUIRE()
    # The probe count depends on the hash seed (slot collisions); the
    # original membership test and item lookup both reach the key.
    assert key.calls >= 2


class _StrLikeMeta(type):
    """A key class that hashes like ``str`` and compares equal to it."""

    def __hash__(cls) -> int:
        return hash(str)

    def __eq__(cls, other: object) -> bool:
        return other is str or type.__eq__(cls, other)


class _StrLikeCollidingKey(_CollidingKey, metaclass=_StrLikeMeta):
    pass


class _UnhashableMeta(type):
    __hash__ = None  # type: ignore[assignment]


class _UnhashableTypeCollidingKey(_CollidingKey, metaclass=_UnhashableMeta):
    __hash__ = _CollidingKey.__hash__


class _RaisingMeta(type):
    def __hash__(cls) -> int:
        raise RuntimeError("key type hashed")

    def __eq__(cls, other: object) -> bool:
        raise RuntimeError("key type compared")


class _RaisingTypeKey(_CollidingKey, metaclass=_RaisingMeta):
    __hash__ = _CollidingKey.__hash__


def test_str_like_key_type_cannot_enter_the_fast_path(
    kwdefaults: dict[str, Any],
) -> None:
    # Codex r1 P1: a key whose *type* hashes and compares like ``str`` must
    # still take the original generator (membership True, lookup KeyError).
    name = next(iter(kwdefaults))
    value = kwdefaults.pop(name)
    kwdefaults[_StrLikeCollidingKey(name, [True, False])] = value  # type: ignore[index]
    with pytest.raises(KeyError):
        REQUIRE()


def test_unhashable_key_type_keeps_original_acceptance(
    kwdefaults: dict[str, Any],
) -> None:
    # Codex r1 P2: the original guard accepts a colliding key that always
    # matches and keeps the original value; key types are never hashed.
    name = next(iter(kwdefaults))
    value = kwdefaults.pop(name)
    key = _UnhashableTypeCollidingKey(name, [True] * 64)
    kwdefaults[key] = value  # type: ignore[index]
    REQUIRE()
    assert key.calls >= 2


def test_earlier_mismatch_short_circuits_before_hostile_key_type(
    kwdefaults: dict[str, Any],
) -> None:
    # Codex r1 P2: an earlier identity mismatch raises ValueError before any
    # later key could run code; the key type is never hashed or compared.
    names = list(kwdefaults)
    first, last = names[0], names[-1]
    kwdefaults[first] = object()
    value = kwdefaults.pop(last)
    kwdefaults[_RaisingTypeKey(last, [True] * 64)] = value  # type: ignore[index]
    with pytest.raises(ValueError, match="operation authority changed"):
        REQUIRE()


def test_colliding_key_that_never_matches_is_rejected(
    kwdefaults: dict[str, Any],
) -> None:
    name = next(iter(kwdefaults))
    value = kwdefaults.pop(name)
    kwdefaults[_CollidingKey(name, [])] = value  # type: ignore[index]
    with pytest.raises(ValueError, match="operation authority changed"):
        REQUIRE()
