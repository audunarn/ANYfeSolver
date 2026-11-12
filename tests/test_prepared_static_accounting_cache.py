"""Exact retained accounting and bounded empty reducers, without timing claims."""
import sys
import numpy as np
import pytest
from anyfesolver.prepared_static_envelopes import (
    CanonicalEnvelopePlan, EnvelopeAccumulator, _storage)


def accumulator():
    plan = CanonicalEnvelopePlan((), (), False, False, 0, 0)
    result = EnvelopeAccumulator(plan)
    result._allocate('stress', 1, (('value', (2,)), ('tensor', (2, 2))))
    result._allocate('stress', 2, (('value', ()),))
    return result


def exact(owner):
    expected = _storage((owner.entries, owner.entity_keys))
    assert owner.nbytes() == expected
    assert owner.nbytes() == expected
    return expected


def fields(scale):
    return {'value': np.array([scale, -scale]),
            'tensor': np.array([[scale, 2*scale], [3*scale, -scale]])}


def test_stable_accounting_reuses_exact_storage_walk():
    owner = accumulator()
    calls = []
    previous = sys.getprofile()
    def observe(frame, event, arg):
        if event == 'call' and frame.f_code is _storage.__code__:
            calls.append(1)
    try:
        sys.setprofile(observe)
        first = owner.nbytes()
        walked = len(calls)
        assert walked > 0
        assert owner.nbytes() == first
        assert len(calls) == walked
    finally:
        sys.setprofile(previous)
    exact(owner)


def test_accounting_tracks_allocation_labels_times_cow_merge_and_detached_results():
    owner = accumulator()
    initial = exact(owner)
    owner.fold('stress', 1, fields(2.), 'short', None, 0)
    exact(owner)
    tx = owner.transaction()
    assert tx._nbytes_cache is None
    exact(tx)
    reduced = tx.empty_subset(('stress',), 1)
    exact(reduced)
    reduced.fold('stress', 1, fields(4.), 'long case name ' * 37, 123.456, 1)
    exact(reduced)
    tx.merge_reduced(reduced)
    assert tx._nbytes_cache is None
    changed = exact(tx)
    assert changed > initial
    exact(owner)
    assert owner.entries[('stress', 1, 'value')][1][0]['case_ids'][0] == 'short'
    assert tx.entries[('stress', 1, 'value')][1][0]['case_ids'][1] == 'long case name ' * 37
    # A losing merge does not invalidate a still-exact stable state.
    tx.merge_reduced(reduced)
    assert tx._nbytes_cache == changed
    exact(tx)
    summary = tx.result('snapshot')
    for entry in summary.extrema:
        entry.minimum.values[...] = 999
        entry.maximum.case_ids[...] = 'detached mutable output'
        for values in entry.absolute.associated_fields.values():
            values[...] = 888
    assert exact(tx) == changed
    tx._allocate('stress', 3, (('extra', (3,)),))
    assert tx._nbytes_cache is None
    exact(tx)
    exact(owner)
    assert ('stress', 3) not in owner.entity_keys


def test_partial_fold_failure_invalidates_only_transaction_and_preserves_committed():
    owner = accumulator()
    before = exact(owner)
    tx = owner.transaction()
    exact(tx)
    # First field changes before the second field fails schema validation.
    broken = {'value': np.array([[2., -2.]]), 'tensor': np.array([[1., 2.]])}
    with pytest.raises(Exception, match='schema or finiteness'):
        tx.fold_tile('stress', 1, broken, (('failure case', 8., 0),))
    assert tx._nbytes_cache is None
    exact(tx)
    assert exact(owner) == before
    assert np.isnan(owner.entries[('stress', 1, 'value')][1][0]['values']).all()


def test_empty_entity_subset_never_copies_unrelated_entries():
    owner = accumulator()
    original = owner.entries
    class LookupOnly:
        def __getitem__(self, key):
            return original[key]
        def keys(self):
            raise AssertionError('empty subset copied the complete entries table')
        def __iter__(self):
            raise AssertionError('empty subset enumerated the complete entries table')
    owner.entries = LookupOnly()
    subset = owner.empty_subset(('stress',), 1)
    assert tuple(subset.entity_keys) == (('stress', 1),)
    assert tuple(subset.entries) == (('stress', 1, 'value'), ('stress', 1, 'tensor'))
    exact(subset)
