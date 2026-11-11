"""Exact native fibre chart transport parity; scoped development only.

This pins the two lifecycle invariants the cyclic chart-energy comparison
relies on, at byte level:

1. A shell chart rebase carries the accepted fibre material and energy
   records through unchanged (no recomputation at the chart event).
2. A chart checkpoint round-trip reproduces the rebased branch's accepted
   increments bit-for-bit: identical accepted results and identical
   checkpoint bytes after every increment.

It deliberately does NOT compare against a non-rebased control branch and is
not a qualification of the control-vs-rebase energy parity tolerance.
"""
from __future__ import annotations

import numpy as np

from anyfesolver import stiffened_shell as ss

from test_stiffened_shell_native_plastic_chart import (
    PREFIX, definition, has_flow, programme,
)
from test_stiffened_shell_staged_material_history import _canonical

# Two post-rebase increments bound the cost; transport exactness does not
# require the full cyclic following sequence.
FOLLOWING = (1.125, 1.5)
OPTIONS = dict(max_iterations=20, max_backtracks=12)
CONTROL = ss.AdaptiveIncrementControl(4, 64)


def test_rebase_and_chart_checkpoint_transport_are_bitwise(request):
    model = definition()

    def owner():
        made = ss.create_analysis(definition=model)
        request.addfinalizer(made.close)
        return made

    source = owner()
    loads = programme(source)
    prefix = source.solve(loads, PREFIX, increment_control=CONTROL, **OPTIONS)
    assert has_flow(prefix), 'no active fibre branch at the rebase predecessor'
    raw = source.checkpoint()

    reset = owner()
    reset.restore(raw)
    assert _canonical(reset.recover().beam_fibre_material) == _canonical(
        prefix.beam_fibre_material)
    assert _canonical(reset.recover().beam_fibre_energy) == _canonical(
        prefix.beam_fibre_energy)
    event = reset.rebase_rotations(prefix.generation)
    assert event.generation == prefix.generation + 1
    np.testing.assert_array_equal(
        event.shell_chart_coordinates.reshape(-1, 6)[:, 3:], 0.)
    # The chart event must not recompute or perturb the accepted records.
    assert _canonical(event.beam_fibre_material) == _canonical(
        prefix.beam_fibre_material)
    assert _canonical(event.beam_fibre_energy) == _canonical(
        prefix.beam_fibre_energy)

    chart_raw = reset.checkpoint()
    restored = owner()
    restored.restore(chart_raw)
    assert restored.checkpoint() == chart_raw

    records = {}
    for label, branch in (('reset', reset), ('restored', restored)):
        rows = []

        def observe(result, branch=branch, rows=rows):
            rows.append((result, branch.checkpoint()))

        finished = branch.solve(loads, FOLLOWING, increment_control=CONTROL,
                                on_accepted=observe, **OPTIONS)
        assert finished.status == 'completed'
        records[label] = rows
    assert len(records['reset']) == len(records['restored']) == len(FOLLOWING)
    for (left, left_bytes), (right, right_bytes) in zip(
            records['reset'], records['restored'], strict=True):
        assert _canonical(left) == _canonical(right)
        assert left_bytes == right_bytes
    assert reset.checkpoint() == restored.checkpoint()
