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
    assert key.calls == 2


def test_colliding_key_that_never_matches_is_rejected(
    kwdefaults: dict[str, Any],
) -> None:
    name = next(iter(kwdefaults))
    value = kwdefaults.pop(name)
    kwdefaults[_CollidingKey(name, [])] = value  # type: ignore[index]
    with pytest.raises(ValueError, match="operation authority changed"):
        REQUIRE()
