"""Focused regression: native beam chart rebase replays from the stored predecessor.

The shell chart rebase is the same accepted constitutive generation. Its exact
native beam history guard must re-evaluate the accepted update from the
stored predecessor (driver._beam_predecessors); re-evaluating from the
accepted history re-runs the return mapping on a zero increment and rounds
fibre entries at the last ulp. The accepted history itself stays unchanged,
physical beam frames/resultants/translations stay identical across the
rebase, stale generations and cancellation are refused, and the untouched
checkpoint round-trips byte-for-byte.
"""
import numpy as np
import pytest

from anyfesolver import stiffened_shell as ss
from anyfesolver.control import CancellationToken, SolveCancelled
from test_stiffened_shell_native_plastic_chart import (
    definition, programme, same, has_flow, PREFIX, OPTIONS, CONTROL,
)
from test_stiffened_shell_staged_material_history import _capture


def test_rebase_replays_native_beam_history_from_stored_predecessor(
        request, tmp_path):
    model = definition()
    control = ss.create_analysis(definition=model)
    request.addfinalizer(control.close)
    loads = programme(control)
    prefix = control.solve(loads, PREFIX, increment_control=CONTROL, **OPTIONS)
    assert has_flow(prefix), "active fibre branch required at the rebase predecessor"
    raw = control.checkpoint()
    reset = ss.create_analysis(definition=model)
    request.addfinalizer(reset.close)
    reset.restore(raw)
    before = _capture(reset, reset.recover(), tmp_path / "before-rebase")
    origins = (reset._finite_driver._shell_origins,
               reset._finite_driver._beam_origins)
    token = CancellationToken()
    token.cancel("before plastic chart reset")
    with pytest.raises(SolveCancelled):
        reset.rebase_rotations(prefix.generation, cancellation_token=token)
    assert reset.checkpoint() == raw
    with pytest.raises(ValueError, match="stale"):
        reset.rebase_rotations(prefix.generation - 1)
    assert reset.checkpoint() == raw
    event = reset.rebase_rotations(prefix.generation)
    assert event.generation == prefix.generation + 1
    after = _capture(reset, event, tmp_path / "after-rebase")
    assert reset._finite_driver._shell_origins is origins[0]
    assert reset._finite_driver._beam_origins is origins[1]
    same(before, after, model, exact=True)
    for field in ('beam_positions', 'beam_frames', 'cell_frames', 'cell_resultants'):
        np.testing.assert_array_equal(getattr(before['result'], field),
                                      getattr(after['result'], field))
    np.testing.assert_array_equal(
        event.shell_chart_coordinates.reshape(-1, 6)[:, 3:], 0.)
    chart = ss.create_analysis(definition=model)
    request.addfinalizer(chart.close)
    chart.restore(reset.checkpoint())
    restored = _capture(chart, chart.recover(), tmp_path / "chart-restored")
    same(after, restored, model, exact=True)
    assert chart.checkpoint() == reset.checkpoint()
    assert control.checkpoint() == raw
