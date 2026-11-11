"""Regression coverage for the owned B2 payload copy protocol.

Copy/deepcopy/reconstruction of _Array/_CSR must preserve the owned
immutable data, detach public materialized metadata, keep object payloads
safe from cross-copy mutation, and keep __deepcopy__ memo semantics
consistent for repeated references.
"""
import copy

import numpy as np
import pytest
from scipy import sparse

from anyfesolver._stiffened_shell_b2_definition_ownership import (
    _Array, _CSR, _Coordinates,
)


def _object_array(*elements):
    array = np.empty(len(elements), dtype=object)
    for i, element in enumerate(elements):
        array[i] = element
    return array


def test_numeric_array_copy_deepcopy_and_reconstruction_preserve_payload():
    source = np.arange(6, dtype=np.float64).reshape(2, 3)
    owned = _Array(source)
    source[:] = 99.
    expected = np.arange(6, dtype=np.float64).reshape(2, 3)
    for clone in (copy.copy(owned), copy.deepcopy(owned), _Array(owned)):
        view = clone.materialize()
        np.testing.assert_array_equal(view, expected)
        assert view.dtype == np.float64 and view.shape == (2, 3)
        with pytest.raises(ValueError):
            view[0, 0] = 17.
        np.testing.assert_array_equal(clone.materialize(), expected)
        np.testing.assert_array_equal(owned.materialize(), expected)


def test_materialized_metadata_mutation_is_detached_per_view():
    owned = _Array(np.arange(6, dtype=np.float64))
    first = owned.materialize()
    second = owned.materialize()
    assert first is not second
    first.shape = (2, 3)
    assert second.shape == (6,)
    assert owned.materialize().shape == (6,)
    np.testing.assert_array_equal(owned.materialize(), np.arange(6.))


def test_object_payload_deepcopy_isolated_from_original_and_shallow_copy():
    payload = [0.]
    owned = _Array(_object_array(payload, [1.]))
    payload.append(99.)
    assert owned.materialize()[0] == [0.]
    deep = copy.deepcopy(owned)
    shallow = copy.copy(owned)
    deep_view = deep.materialize()
    deep_view[0].append(99.)
    assert owned.materialize()[0] == [0.]
    assert shallow.materialize()[0] == [0.]
    assert deep.materialize()[0] == [0.]


def test_object_payload_materializations_are_independent():
    owned = _Array(_object_array([0.], [1.]))
    first = owned.materialize()
    second = owned.materialize()
    first[0].append(99.)
    assert second[0] == [0.]
    assert owned.materialize()[0] == [0.]


def test_deepcopy_memo_preserves_repeated_array_references():
    owned = _Array(np.arange(3, dtype=np.float64))
    memo = {}
    first = copy.deepcopy(owned, memo)
    second = copy.deepcopy(owned, memo)
    assert first is second
    cloned = copy.deepcopy((owned, owned))
    assert cloned[0] is cloned[1]
    assert cloned[0] is not owned
    np.testing.assert_array_equal(cloned[0].materialize(), owned.materialize())


def test_array_reconstruction_adopts_owned_payload_without_nesting():
    inner = _Array(_object_array([0.], [1.]))
    adopted = _Array(inner)
    assert adopted._bytes is inner._bytes
    view = adopted.materialize()
    assert isinstance(view[0], list)


def test_object_array_of_owned_arrays_snapshots_without_deepcopy_failure():
    # The pre-repair crash: an object-dtype element that is itself an owned
    # _Array made copy.deepcopy reconstruct via setattr and raise.
    inner = _Array(np.arange(3, dtype=np.float64))
    owned = _Array(np.array([inner], dtype=object))
    view = owned.materialize()
    np.testing.assert_array_equal(view[0].materialize(), np.arange(3, dtype=np.float64))
    cloned = copy.deepcopy(owned)
    np.testing.assert_array_equal(
        cloned.materialize()[0].materialize(), np.arange(3, dtype=np.float64))


def test_csr_copy_deepcopy_and_reconstruction_preserve_sparse_payload():
    source = sparse.eye(4, format='csr')
    owned = _CSR(source)
    source.data[:] = 99.
    for clone in (copy.copy(owned), copy.deepcopy(owned), _CSR(owned.materialize())):
        fresh = clone.materialize()
        np.testing.assert_array_equal(fresh.toarray(), np.eye(4))
        np.testing.assert_array_equal(owned.materialize().toarray(), np.eye(4))


def test_csr_materialized_mutation_and_attribute_replacement_are_isolated():
    owned = _CSR(sparse.eye(4, format='csr'))
    fresh = owned.materialize()
    fresh.data[:] = 17.
    fresh.indices[:] = 0
    fresh.indptr[:] = 0
    fresh._shape = (1, 1)
    np.testing.assert_array_equal(owned.materialize().toarray(), np.eye(4))
    np.testing.assert_array_equal(owned.materialize().data, np.ones(4))


def test_deepcopy_memo_preserves_repeated_csr_references():
    owned = _CSR(sparse.eye(3, format='csr'))
    memo = {}
    first = copy.deepcopy(owned, memo)
    second = copy.deepcopy(owned, memo)
    assert first is second
    cloned = copy.deepcopy((owned, owned))
    assert cloned[0] is cloned[1]
    assert cloned[0] is not owned
    np.testing.assert_array_equal(cloned[0].materialize().toarray(), np.eye(3))


def test_coordinates_copy_and_deepcopy_are_identity_stable():
    coords = _Coordinates({1: (0., 0., 0.), 2: (1., 0., 0.)})
    assert copy.copy(coords) is coords
    assert copy.deepcopy(coords) is coords
    cloned = copy.deepcopy((coords, coords))
    assert cloned[0] is cloned[1] is coords


def test_definition_deepcopy_preserves_owned_views_and_admission():
    from dataclasses import replace
    from test_stiffened_shell_b2_mixed_panel_experimental import _definition, _load
    from anyfesolver import stiffened_shell as api
    base, _ = _definition('q4')
    cloned = copy.deepcopy(base)
    assert cloned is not base
    np.testing.assert_array_equal(cloned.prescribed_offset, base.prescribed_offset)
    np.testing.assert_array_equal(
        cloned.transformation.toarray(), base.transformation.toarray())
    for node in base.shell_nodes_m:
        assert cloned.shell_nodes_m[node] == base.shell_nodes_m[node]
    rebuilt = replace(cloned)
    np.testing.assert_array_equal(rebuilt.prescribed_offset, base.prescribed_offset)
    np.testing.assert_array_equal(
        rebuilt.transformation.toarray(), base.transformation.toarray())
    owner = api.create_analysis(definition=cloned)
    try:
        result = owner.solve_linear(_load())
        assert result.status == 'completed', result.failure_reason
    finally:
        owner.close()
