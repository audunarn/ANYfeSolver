"""The canonical-JSON fast paths must equal the general definition exactly.

``_canonical_value`` in the S3 and S3 V2D state modules gained exact-type fast
paths (exact builtins, exact float64 ndarrays).  The reference functions below
are verbatim copies of the implementations before that change; every input in
the corpus must give the identical tree (bitwise floats), identical canonical
bytes and digests, or the identical exception type and message.
"""

from __future__ import annotations

import collections
import json
import math
from typing import Any, Mapping as TypingMapping, Sequence as TypingSequence
from collections.abc import Mapping as AbcMapping, Sequence as AbcSequence

import numpy as np
import pytest

from anyfesolver import e4_pl_s3_state as s3_state
from anyfesolver import e4_pl_s3_v2d_state as v2d_state


def _v2d_reference(value: Any, *, path: str = "$") -> Any:
    if isinstance(value, np.ndarray):
        return _v2d_reference(value.tolist(), path=path)
    if isinstance(value, np.generic):
        return _v2d_reference(value.item(), path=path)
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise v2d_state.V2DStateError(f"nonfinite canonical value at {path}")
        return 0.0 if value == 0.0 else value
    if isinstance(value, TypingMapping):
        made: dict[str, Any] = {}
        for key, member in value.items():
            if not isinstance(key, str):
                raise v2d_state.V2DStateError(f"non-string canonical key at {path}")
            if key in made:
                raise v2d_state.V2DStateError(f"duplicate canonical key {key!r} at {path}")
            made[key] = _v2d_reference(member, path=f"{path}.{key}")
        return made
    if isinstance(value, TypingSequence) and not isinstance(value, (str, bytes, bytearray)):
        return [
            _v2d_reference(member, path=f"{path}[{index}]")
            for index, member in enumerate(value)
        ]
    raise v2d_state.V2DStateError(
        f"unsupported canonical value {type(value).__name__} at {path}"
    )


def _s3_reference(value: Any, *, path: str = "$") -> Any:
    if isinstance(value, np.ndarray):
        return _s3_reference(value.tolist(), path=path)
    if isinstance(value, np.generic):
        return _s3_reference(value.item(), path=path)
    if value is None or isinstance(value, (str, bool)):
        return value
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise s3_state.S3CommittedStateError(f"nonfinite canonical value at {path}")
        return 0.0 if value == 0.0 else value
    if isinstance(value, AbcMapping):
        result: dict[str, Any] = {}
        for key, member in value.items():
            if not isinstance(key, str):
                raise s3_state.S3CommittedStateError(f"non-string canonical key at {path}")
            if key in result:
                raise s3_state.S3CommittedStateError(f"duplicate canonical key {key!r} at {path}")
            result[key] = _s3_reference(member, path=f"{path}.{key}")
        return result
    if isinstance(value, AbcSequence) and not isinstance(value, (str, bytes, bytearray)):
        return [
            _s3_reference(member, path=f"{path}[{index}]")
            for index, member in enumerate(value)
        ]
    raise s3_state.S3CommittedStateError(
        f"unsupported canonical value {type(value).__name__} at {path}"
    )


def _fingerprint(tree: Any) -> Any:
    """Type-exact, bit-exact structural fingerprint of a canonical tree."""

    if type(tree) is float:
        return ("float", tree.hex(), math.copysign(1.0, tree))
    if type(tree) is dict:
        return ("dict", tuple((key, type(key).__name__, _fingerprint(member))
                              for key, member in tree.items()))
    if type(tree) is list:
        return ("list", tuple(_fingerprint(member) for member in tree))
    return (type(tree).__name__, repr(tree))


def _outcome(function: Any, value: Any) -> Any:
    try:
        tree = function(value)
    except Exception as exc:  # compared by exact type and message
        return ("error", type(exc), str(exc))
    text = json.dumps(tree, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False)
    return ("ok", _fingerprint(tree), text)


class _ArraySubclass(np.ndarray):
    def tolist(self):  # a hostile subclass must keep its general-path behaviour
        return [1.0, -0.0]


class _StrKey(str):
    pass


class _MappingView(AbcMapping):
    def __init__(self, data):
        self._data = dict(data)

    def __getitem__(self, key):
        return self._data[key]

    def __iter__(self):
        return iter(self._data)

    def __len__(self):
        return len(self._data)


def _corpus() -> list[Any]:
    rng = np.random.default_rng(20261124)
    special = np.array([0.0, -0.0, 1.0, -1.0, 5e-324, -5e-324, 1.7976931348623157e308,
                        np.nextafter(1.0, 2.0), 1e-300, 0.1, 1.0 / 3.0])
    arrays = [
        special,
        special.reshape(1, -1),
        special[::2],
        special[::-1],
        np.asfortranarray(rng.standard_normal((3, 4))),
        rng.standard_normal((2, 3, 4)) * 1e-12,
        np.zeros((0, 3)),
        np.zeros(0),
        np.array(-0.0),
        np.array(2.5),
        np.array([1.0, np.nan]),
        np.array([[1.0], [np.inf]]),
        np.array([-np.inf]),
        np.array([np.nan, np.inf]).reshape(1, 2),
        special.astype(">f8"),
        special.astype(np.float32),
        np.arange(6, dtype=np.int64).reshape(2, 3),
        np.array([True, False]),
        np.array([1.0, "a"], dtype=object),
        np.array([[0.5, None]], dtype=object),
        special.view(_ArraySubclass),
        np.frombuffer(special.tobytes(), dtype=np.float64),
    ]
    scalars = [0.0, -0.0, 1.5, float("nan"), float("inf"), -float("inf"), 5e-324,
               0, -3, 2**80, True, False, None, "", "x", _StrKey("k"), b"bytes",
               bytearray(b"b"), np.float64(-0.0), np.float32(0.25), np.int32(7),
               np.bool_(True), np.float64("nan"), complex(1, 2), object(), {1, 2}]
    containers = [
        [], (), {}, [1, [2.0, [-0.0]]], (0.0, -0.0, (1, "a")),
        {"a": 1.0, "b": [-0.0, special], "c": {"d": np.array([[-0.0, 3.0]])}},
        {"a": {"b": [1.0, float("nan")]}},
        {"x": [0, 1, {"y": np.array([np.inf])}]},
        {1: 2.0},
        {_StrKey("a"): 1.0, "b": 2.0},
        collections.OrderedDict([("z", 1.0), ("a", -0.0)]),
        _MappingView({"q": [special, -0.0]}),
        [special, special.astype(">f8"), np.float64(-0.0), {"k": special[::3]}],
        "string-root",
        [b"x"],
        [set()],
        rng.standard_normal(200).tolist(),
    ]
    state_like = {
        "schema": "s",
        "element_id": 7,
        "node_ids": [1, 2, 3],
        "committed_total_u": rng.standard_normal(18),
        "plastic_strain": np.zeros((15, 3)),
        "alpha": -np.zeros(15),
        "layer_stress": rng.standard_normal((15, 3)),
        "station_generalized_strain": rng.standard_normal((3, 8)),
        "initial_field_provenance": {},
        "flag": None,
    }
    return arrays + scalars + containers + [state_like, [state_like, state_like]]


@pytest.mark.parametrize(
    "module, reference",
    [(v2d_state, _v2d_reference), (s3_state, _s3_reference)],
    ids=["v2d", "s3"],
)
def test_fast_canonical_value_equals_reference(module: Any, reference: Any) -> None:
    corpus = _corpus()
    assert len(corpus) > 60
    for index, value in enumerate(corpus):
        assert _outcome(module._canonical_value, value) == _outcome(reference, value), index


@pytest.mark.parametrize("module", [v2d_state, s3_state], ids=["v2d", "s3"])
def test_canonical_bytes_and_digest_unchanged_on_random_states(module: Any) -> None:
    rng = np.random.default_rng(7)
    reference = _v2d_reference if module is v2d_state else _s3_reference
    for _ in range(200):
        values = rng.standard_normal(rng.integers(0, 40))
        values[rng.random(values.shape) < 0.2] = -0.0
        values[rng.random(values.shape) < 0.1] = 0.0
        state = {"u": values, "m": values.reshape(-1, 1), "s": values.tolist()}
        expected = (json.dumps(reference(state), sort_keys=True, separators=(",", ":"),
                               ensure_ascii=False, allow_nan=False) + "\n").encode("utf-8")
        assert module.canonical_json_bytes(state) == expected


def test_fast_path_does_not_call_hostile_array_subclass_differently() -> None:
    calls = []

    class Counting(np.ndarray):
        def tolist(self):
            calls.append(1)
            return [0.0]

    value = np.zeros(3).view(Counting)
    for module in (v2d_state, s3_state):
        calls.clear()
        assert module._canonical_value(value) == [0.0]
        assert calls == [1]
