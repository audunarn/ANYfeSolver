"""Focused development tests for capsule export layout-signature scaling.

The stored-row bookkeeping must stay compact for many cases: one immutable
signature object per layout revision shared by every stored row of that
revision, an indexed entity-id -> size lookup, and no numeric payload
retained after the sink has consumed it.
"""
import numpy as np
import pytest

from anyfesolver import (PatternCase, RecoveryCapsuleError, RecoveryConfig,
                       prepare_static_response_evaluator)
from anyfesolver.result_capsule import _CapsuleExport, _Layout
from test_prepared_static_evaluator import _shell_case, _shell_model


class _CountingSink:
    """Owner-neutral sink recording the structural event stream."""

    def __init__(self):
        self.begun = None
        self.cases = []
        self.stored_events = []
        self.final = None

    def capsule_begin(self, header):
        self.begun = dict(header)

    def capsule_case(self, record, coefficients):
        self.cases.append((dict(record), np.array(coefficients, copy=True)))

    def capsule_stored(self, key, case_indices, values):
        self.stored_events.append((key, np.asarray(case_indices),
                                   np.asarray(values, copy=True)))

    def capsule_end(self, final):
        self.final = dict(final)


def test_layout_signature_is_one_object_per_revision():
    layout = _Layout()
    layout.add(1, 3)
    layout.add(2, 4)
    first = layout.signature()
    # The signature object is cached and shared, not rebuilt per call.
    assert layout.signature() is first
    layout.add(3, 1)
    second = layout.signature()
    # An actual layout change invalidates the cached signature.
    assert second is not first and second != first
    assert layout.signature() is second
    # Indexed entity-id -> size lookup: exact sizes, None for unknown ids.
    assert layout.size_of(1) == 3 and layout.size_of(2) == 4 and layout.size_of(3) == 1
    assert layout.size_of(9) is None
    assert layout.count() == 8
    np.testing.assert_array_equal(layout.association(),
                                  np.array([1, 1, 1, 2, 2, 2, 2, 3]))


def test_many_stored_rows_share_one_signature_and_release_numeric_payload():
    model = _shell_model()
    pattern = _shell_case(2, pressure=1.0).load_case
    recovery = RecoveryConfig(include_stresses=True, include_reactions=True)
    cases = [PatternCase(f'c{index}', {'p': 0.5 + 0.01 * index}) for index in range(12)]
    with prepare_static_response_evaluator(model, {'p': pattern},
                                           recovery_config=recovery) as evaluator:
        export = _CapsuleExport(evaluator, cases, model_identity='shell-model',
                                mesh_identity='shell-mesh', batch_size=5)
        sink = _CountingSink()
        export.run(sink)

        assert len(sink.cases) == 12
        assert sink.final['case_count'] == 12
        # The scaling invariant is the shared immutable signature and the
        # released numeric payload, not an all-cases fallback: a component
        # stores exactly the cases that need an explicit row.  Every stored
        # row of one component shares the SAME signature object, and it is
        # the layout's current cached signature.
        assert export.stored_rows
        for key, entries in export.stored_rows.items():
            assert entries, key
            signatures = {id(signature) for signature, _index in entries}
            assert len(signatures) == 1, key
            assert export.layouts[key].signature() is entries[0][0]
            indices = [index for _signature, index in entries]
            assert len(set(indices)) == len(indices), key
            if key.startswith('displacement:'):
                # Displacement overrides exist exactly for the cases whose
                # per-case bitwise parity failed; the truly portable cases
                # reconstruct from the basis and keep no stored row.
                assert indices == [position for position, flag
                                   in enumerate(export.displacement_portable)
                                   if flag is False], key
            else:
                # Non-portable certified cases always keep explicit rows.
                assert indices == list(range(12)), key
        # Retained metadata is compact: one signature object per component
        # plus one (signature, index) pair per row, never a fresh entity
        # tuple per stored row.
        row_count = sum(len(entries) for entries in export.stored_rows.values())
        signature_objects = {id(signature) for entries in export.stored_rows.values()
                             for signature, _index in entries}
        assert row_count >= len(export.stored_rows)
        assert len(signature_objects) == len(export.stored_rows)
        # No numeric payload is retained after the sink consumed it.
        assert export._pending_numeric_bytes() == 0
        assert all(not pending for pending in export.pending_rows.values())
        # The sink received every stored row exactly once, batch by batch.
        for key, entries in export.stored_rows.items():
            emitted = sum(indices.shape[0] for name, indices, _values in sink.stored_events
                          if name == key)
            assert emitted == len(entries), key
        # Superseded-layout fail-closed behavior is preserved: a layout
        # change after stored rows were captured is refused.
        key = next(iter(export.stored_rows))
        with pytest.raises(RecoveryCapsuleError):
            export._grow_layout(key, export.layouts[key], 99, 1)
