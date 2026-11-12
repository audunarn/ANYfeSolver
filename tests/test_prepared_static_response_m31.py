"""M31: identical-result levers, each compared with the code it replaces (M31b: the
lean selected reactions were reverted to the pre-M31 code after review)."""
import collections
import copy
import sys

import numpy as np
import pytest

from anyfesolver import LoadCase, RecoveryConfig, prepare_static_evaluator, prepare_static_response_evaluator
from anyfesolver.prepared_static import _plain_deepcopy
from anyfesolver.prepared_static_loads import load_packet_storage_bytes
from anyfesolver.static_batch import StaticCase
import prepared_static_response_m16_assessment  # noqa: F401  (adds scripts/ to sys.path)
from benchmark_static_batches import mixed_free_model
from test_prepared_static_response_m26 import _grid_model

STRESS = RecoveryConfig(include_displacements=False, include_stresses=True, include_reactions=True)

# Observe the running stdlib independently of the solver's private seams.
# A future interpreter's product behavior remains unverified until these
# behavioral comparisons run there; this probe does not certify Python 3.15.
_memo_probe = [[]]
_MEMO_NONE_IS_A_HIT = copy.deepcopy(_memo_probe, {id(_memo_probe): None}) is None
del _memo_probe


# --------------------------------------------------------------------------- deepcopy

def _assert_same_copy(original, ours, theirs, seen=None):
    """ours and theirs (copy.deepcopy) have the same types, atomic identities,
    container freshness and aliasing pattern relative to the original."""
    seen = {} if seen is None else seen
    assert type(ours) is type(theirs)
    key = id(original)
    if key in seen:
        assert seen[key] == (id(ours), id(theirs)) or (ours is original and theirs is original)
        return
    seen[key] = (id(ours), id(theirs))
    assert (ours is original) == (theirs is original)
    if isinstance(original, np.ndarray):
        assert ours.dtype == theirs.dtype and ours.shape == theirs.shape and ours.tobytes() == theirs.tobytes()
        assert ours.flags.c_contiguous == theirs.flags.c_contiguous
        return
    if type(original) in (dict, collections.OrderedDict):
        assert list(ours) == list(theirs)
        for (k1, v1), (k2, v2), (k0, v0) in zip(ours.items(), theirs.items(), original.items()):
            _assert_same_copy(k0, k1, k2, seen)
            _assert_same_copy(v0, v1, v2, seen)
        return
    if type(original) in (list, tuple):
        assert len(ours) == len(theirs) == len(original)
        for a, b, o in zip(ours, theirs, original):
            _assert_same_copy(o, a, b, seen)
        return
    if isinstance(original, float):
        assert ours.hex() == theirs.hex() if ours == ours else theirs != theirs
        assert ours is original and theirs is original
        return
    assert ours == theirs


def _samples():
    shared = [1, 2, 3]
    nested = {'k': shared}
    yield {'a': 1, 'b': [1.5, 'x', None, True], 'c': (1, 2), 'd': (1, [2])}
    yield {'shared': shared, 'again': shared, 'nested': nested, 'tuple': (shared, shared)}
    yield {'array': np.arange(6.0).reshape(2, 3), 'scalar': np.float64(2.5), 'od': collections.OrderedDict(a=[1])}
    yield {'nodes': list(range(5000)), 'components': [{'nodes': list(range(100)), 'size': 100}] * 2}
    yield [float('nan'), -0.0, 1e308, (), [], {}]


@pytest.mark.parametrize('index', range(5))
def test_plain_deepcopy_matches_deepcopy(index):
    original = list(_samples())[index]
    _assert_same_copy(original, _plain_deepcopy(original), copy.deepcopy(original))


@pytest.mark.parametrize('builder', ['grid', 'mixed_free'])
def test_plain_deepcopy_of_the_real_operator_info(builder):
    if builder == 'grid':
        model, patterns = _grid_model()
        owner = prepare_static_response_evaluator(model, patterns, recovery_config=STRESS, stress_execution='certified_vectorized')
    else:
        owner = prepare_static_evaluator(mixed_free_model()[0], recovery_config=RecoveryConfig(include_stresses=False))
    with owner:
        info = owner._operator_info
        _assert_same_copy(info, _plain_deepcopy(info), copy.deepcopy(info))


def test_plain_deepcopy_honours_a_memo_entry_of_none():
    # M31b (review): a memo value of None is treated as the running copy.deepcopy
    # treats it: a hit up to Python 3.14, a miss on entry from 3.15.
    shared = [1, [2]]
    pair = (shared, 3)
    for value in (shared, {'k': shared}, [shared, shared], pair):
        for target in (shared, pair):
            memo_ours, memo_theirs = {id(target): None}, {id(target): None}
            ours = _plain_deepcopy(value, memo_ours)
            theirs = copy.deepcopy(value, memo_theirs)
            assert repr(ours) == repr(theirs)
            assert repr(memo_ours[id(target)]) == repr(memo_theirs[id(target)])


def test_plain_deepcopy_memo_none_rule_follows_the_interpreter():
    assert _MEMO_NONE_IS_A_HIT is (sys.version_info < (3, 15))
    shared = [1, [2]]
    copied = _plain_deepcopy(shared, {id(shared): None})
    if _MEMO_NONE_IS_A_HIT:
        assert copied is None
    else:
        assert copied == shared and copied is not shared and copied[1] is not shared[1]


def test_plain_deepcopy_tuple_rechecks_the_memo_after_its_items():
    # copy.deepcopy looks a tuple up again after copying its items; an item's
    # __deepcopy__ may have registered it (here: as None).
    class Planter:
        def __init__(self):
            self.target = None

        def __deepcopy__(self, memo):
            memo[id(self.target)] = None
            return 'planted'

    for make in (_plain_deepcopy, copy.deepcopy):
        planter = Planter()
        value = (planter, [1])
        planter.target = value
        assert make([value, value]) == [None, None]


# --------------------------------------------------------------------------- packet storage

def _reference_storage(case):  # pre-M31 walk
    from anyfesolver.boundary import _exact_load_case_instance_namespace
    if case.load_case is None:
        return 256
    state = _exact_load_case_instance_namespace(case.load_case)

    def size(value):
        if type(value) is np.ndarray:
            return 256 + int(value.size) * 16
        if type(value) in (list, tuple):
            return 64 + sum(size(item) for item in value)
        if type(value) is dict:
            return 256 + sum(128 + size(key) + size(item) for key, item in value.items())
        if value is None or type(value) in (bool, int, float, str):
            return 64 + (4 * len(value) if type(value) is str else 0)
        return 64
    return 256 + sum(size(state[name]) for name in ("nodal_loads", "element_loads", "pressure_loads", "gravity",
                                                    "added_node_masses", "name"))


def _cases():
    plain = LoadCase('plain')
    for k in range(1, 400):
        plain.pressure_loads[k] = 1000.0 + k
    plain.add_nodal_load(3, [0., 0., 1., 0., 0., 0.])
    yield plain
    tuples = LoadCase('tuples')
    tuples.pressure_loads[1] = (1.0, 2.0, 3.0)
    tuples.pressure_loads[2] = 4.0
    yield tuples
    numpy_values = LoadCase('numpy values')
    numpy_values.pressure_loads[5] = np.float64(2.0)
    numpy_values.pressure_loads[6] = 1.0
    yield numpy_values
    yield LoadCase('empty')


@pytest.mark.parametrize('index', range(4))
def test_packet_storage_bytes_unchanged(index):
    load = list(_cases())[index]
    case = StaticCase(load.name, load, 0.0)
    assert load_packet_storage_bytes(case) == _reference_storage(case)
