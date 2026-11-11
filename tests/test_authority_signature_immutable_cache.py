"""Authority cache must not hide mutable data or changed conversion work."""

import ast
import inspect
from pathlib import Path
import struct
import sys
from types import MappingProxyType, SimpleNamespace
from typing import Any

import numpy as np
import pytest

from anyfesolver import e4_pl_s3_state as state


def fresh_signer():
    # The production factory is intentionally deleted after initialization.
    tree = ast.parse(Path(state.__file__).read_text(encoding="utf-8"))
    factory = next(node for node in tree.body
                   if isinstance(node, ast.FunctionDef)
                   and node.name == "_make_module_authority_signer")
    namespace = {"Any": Any}
    exec(compile(ast.Module(body=[factory], type_ignores=[]),
                 state.__file__, "exec"), namespace)
    numpy_proxy = SimpleNamespace(sctypeDict=np.sctypeDict, generic=np.generic,
                                  ndarray=np.ndarray, asarray=np.asarray,
                                  float64=np.float64)
    signer = namespace[factory.name](numpy_proxy, MappingProxyType)
    cached = inspect.getclosurevars(signer).nonlocals["cached_immutable"]
    cache = inspect.getclosurevars(cached).nonlocals["immutable_cache"]
    return signer, numpy_proxy, cache


def retained(cache, value):
    return any(entry is not None and entry[0] is value for entry in cache)


def test_float_bits_order_and_immutable_hit():
    signer, _, cache = fresh_signer()
    nan = struct.unpack("=d", bytes.fromhex("341200000000f87f"))[0]
    values = (0.0, -0.0, nan, (3, "value", None), frozenset({2, 1}))
    expected = ("tuple", (
        *(('float', struct.pack("=d", value)) for value in values[:3]),
        ("tuple", (("int", 3), ("str", "value"), ("NoneType", None))),
        ("frozenset", (("int", 1), ("int", 2))),
    ))
    assert signer(values) == expected
    assert signer(values) == expected
    assert retained(cache, values)


@pytest.mark.parametrize("kind", ["list", "dict", "proxy", "array"])
def test_nested_mutable_leaf_stays_live(kind):
    signer, _, cache = fresh_signer()
    mapping = {"value": 1}
    leaf = {"list": [1], "dict": mapping, "proxy": MappingProxyType(mapping),
            "array": np.array([1.0])}[kind]
    root = ("constant", (leaf,))
    before = signer(root)
    assert signer(root) == before and not retained(cache, root)
    if kind in {"dict", "proxy"}:
        mapping["value"] = 2
    else:
        leaf[0] = 2
    assert signer(root) != before


def test_array_metadata_remains_live():
    signer, _, _ = fresh_signer()
    array = np.arange(4.0)
    root = (array,)
    before = signer(root)
    array.shape = (2, 2)
    assert signer(root) != before
    shaped = signer(root)
    array.flags.writeable = False
    assert signer(root) != shaped


def test_subclasses_and_opaque_objects_are_not_cached():
    signer, _, cache = fresh_signer()
    class Integer(int):
        pass
    value = Integer(3)
    root = (value,)
    expected = ("tuple", (("identity", Integer.__module__,
                           Integer.__qualname__, id(value)),))
    assert signer(root) == expected
    assert signer(root) == expected and not retained(cache, root)


@pytest.mark.parametrize("binding", ["asarray", "float64"])
def test_changed_numpy_dependency_is_observed_after_hit(binding):
    signer, proxy, cache = fresh_signer()
    root = (1.25,)
    expected = signer(root)
    assert signer(root) == expected and retained(cache, root)
    if binding == "asarray":
        def changed(*args, **kwargs):
            raise RuntimeError("observed conversion")
        proxy.asarray = changed
        with pytest.raises(RuntimeError, match="observed conversion"):
            signer(root)
    else:
        proxy.float64 = np.float32
        assert signer(root) == ("tuple", (("float", np.asarray(1.25,
                                                    dtype=np.float32).tobytes()),))
    setattr(proxy, binding, getattr(np, binding))
    assert signer(root) == expected


def test_changed_numpy_array_dispatch_precedes_cache():
    signer, proxy, _ = fresh_signer()
    root = (1.25,)
    signer(root)
    proxy.ndarray = tuple
    result = signer(root)
    assert result[0] == "ndarray" and result[1] == np.dtype(float).str


def test_cache_is_bounded_owns_identity_and_rejects_large_aggregates():
    signer, _, cache = fresh_signer()
    roots = [(index, "value") for index in range(600)]
    originals = [signer(root) for root in roots]
    assert len(cache) == 256
    for index, entry in enumerate(cache):
        if entry is not None:
            assert index == (id(entry[0]) >> 4) % 256
    assert signer(roots[0]) == originals[0]
    equal_replacement = tuple(list(roots[-1]))
    assert equal_replacement is not roots[-1]
    assert signer(equal_replacement) == originals[-1]
    assert retained(cache, equal_replacement)
    for large in (tuple(range(300)), ("x" * 5000,), (1 << 40000,)):
        signer(large)
        assert not retained(cache, large)


def test_live_authority_tables_match_their_frozen_signatures():
    from anyfesolver.current_state_tangent import _QUALIFIED_PROFILES
    signer, _, _ = fresh_signer()
    checked = 0
    for profile in _QUALIFIED_PROFILES.values():
        modules = [(sys.modules[profile["module_name"]], profile["module_data_authority"])]
        modules.extend((row["module"], row["data_bindings"])
                       for row in profile["dependency_module_authority"])
        for module, bindings in modules:
            for name, (_, expected) in bindings.items():
                observed = vars(module)[name]
                assert signer(observed) == expected
                assert signer(observed) == expected
                checked += 1
    assert checked > 100
