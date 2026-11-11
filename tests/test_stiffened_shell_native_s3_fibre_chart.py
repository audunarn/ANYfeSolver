"""Scoped simultaneous S3-J2/fibre chart ownership from a saved accepted prefix."""
from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from anyfesolver import stiffened_shell as ss
from anyfesolver._stiffened_shell_displacement_adaptive import decode
from anyfesolver._stiffened_shell_displacement_checkpoint import decode_snapshot
from anyfesolver._stiffened_shell_beam_history import (
    arrays as native_history_arrays, histories as native_histories,
)
from anyfesolver._stiffened_shell_ge_experimental import NodalAction
from anyfesolver._ge_beam3_fibre_section import PhysicalFibreSection
from examples.stiffened_shell_coupled_s3_adaptive_history import (
    CONTROL, DIRECTION, OPTIONS, definition,
)
from test_stiffened_shell_native_point_displacement import owner
from test_stiffened_shell_native_plastic_s3_displacement import compare
from test_stiffened_shell_ge_physical_fibre_finite import _yielding_definition
from test_stiffened_shell_staged_material_history import _canonical, _capture, _material_close


def _compare_physical(left, right, model):
    """Use the existing Q4-J2/fibre chart test's normalized physical limits."""
    a, b = left["result"], right["result"]
    assert a.status == b.status == "completed"
    assert a.target == b.target
    assert len(a.history) == len(b.history)
    np.testing.assert_array_equal(
        [row[0] for row in a.history], [row[0] for row in b.history])
    np.testing.assert_allclose(
        np.asarray([row[1] for row in a.history]) / 3000.,
        np.asarray([row[1] for row in b.history]) / 3000., rtol=0., atol=1e-9)
    np.testing.assert_allclose(a.applied_dead_load / 3000.,
                               b.applied_dead_load / 3000.,
                               rtol=0., atol=1e-9)
    np.testing.assert_allclose(a.beam_positions / 2., b.beam_positions / 2.,
                               rtol=0., atol=1e-9)
    np.testing.assert_allclose(
        left["q"][:6 * len(model.shell.node_ids)].reshape(-1, 6)[:, :3] / 2.,
        right["q"][:6 * len(model.shell.node_ids)].reshape(-1, 6)[:, :3] / 2.,
        rtol=0., atol=1e-9)
    for x, y in ((left["shell_frames"], right["shell_frames"]),
                 (a.beam_frames, b.beam_frames), (a.cell_frames, b.cell_frames)):
        np.testing.assert_allclose(x, y, rtol=0., atol=1e-9)
    # Native p18: two force triples and four moment triples per retained cell.
    resultant_scale = np.tile((3000.,) * 6 + (6000.,) * 12,
                              (a.cell_resultants.shape[0], 1))
    np.testing.assert_allclose(a.cell_resultants / resultant_scale,
                               b.cell_resultants / resultant_scale,
                               rtol=0., atol=1e-9)
    wrench_scale = np.array((3000.,) * 3 + (6000.,) * 3)
    for name in ("multipliers", "support_reactions"):
        np.testing.assert_allclose(
            getattr(a, name).reshape(-1, 6) / wrench_scale,
            getattr(b, name).reshape(-1, 6) / wrench_scale,
            rtol=0., atol=1e-9)
    _material_close(left["shell"], right["shell"], model)
    _material_close(left["beam"], right["beam"], model)
    for name in ("shell_s3_material", "beam_fibre_material", "beam_fibre_energy"):
        assert len(getattr(a, name)) == len(getattr(b, name))
        for x, y in zip(getattr(a, name), getattr(b, name), strict=True):
            _material_close(x, y, model)


def test_simultaneous_s3_fibre_rebase_restore_and_signed_continuation(request, tmp_path):
    model = definition()
    direct = owner(request, model)
    reset = owner(request, model)
    reopened = owner(request, model)
    raw = (Path(__file__).parent / "data/native_s3_fibre_history/unloaded-v1.bin").read_bytes()
    prefix = direct.restore(raw)
    reset.restore(raw)
    assert prefix.shell_s3_material and prefix.beam_fibre_material
    assert max(max(row.alpha) for row in prefix.shell_s3_material) > 0.

    context, inner = decode(direct, raw)
    before_metadata, before_arrays = decode_snapshot(direct, inner)
    assert before_metadata["policy"] == "B3_GE_NATIVE_POINT_SHELL_J2_S3_DISPLACEMENT_V1_DEV"
    before_q = before_arrays["q"][:reset._shell_count].reshape(-1, 6)
    assert np.any(before_q[:, 3:] != 0.)
    accepted_shell = _canonical(reset._finite_driver._shell_origins)
    accepted_beam = _canonical(reset._finite_driver._beam_origins)

    charted = reset.rebase_rotations(expected_generation=prefix.generation)
    assert charted.generation == prefix.generation + 1
    compare(prefix, charted)
    assert _canonical(reset._finite_driver._shell_origins) == accepted_shell
    assert _canonical(reset._finite_driver._beam_origins) == accepted_beam
    chart_raw = reset.checkpoint()
    _, chart_inner = decode(reset, chart_raw)
    chart_metadata, chart_arrays = decode_snapshot(reset, chart_inner)
    assert chart_metadata["chart_events"] == before_metadata.get("chart_events", 0) + 1
    np.testing.assert_array_equal(
        chart_arrays["q"][:reset._shell_count].reshape(-1, 6)[:, 3:], 0.)
    np.testing.assert_array_equal(chart_arrays["shell_anchors"], prefix.shell_frames)
    compare(charted, reopened.restore(chart_raw))
    assert reopened.checkpoint() == chart_raw

    kwargs = dict(OPTIONS, displacement_control=CONTROL,
                  increment_control=context.state.limits)
    targets = context.state.targets
    control = direct.solve(DIRECTION, targets, **kwargs)
    after_rebase = reset.solve(DIRECTION, targets, **kwargs)
    after_restore = reopened.solve(DIRECTION, targets, **kwargs)
    assert control.status == after_rebase.status == after_restore.status == "completed"
    original = _capture(direct, control, tmp_path / "direct")
    reset_state = _capture(reset, after_rebase, tmp_path / "rebased")
    restored_state = _capture(reopened, after_restore, tmp_path / "rebased-restored")
    _compare_physical(original, reset_state, model)
    _compare_physical(reset_state, restored_state, model)
    compare(after_rebase, after_restore)
    assert reset.checkpoint() == reopened.checkpoint()
    assert control.free_equilibrium_ratio <= 1e-8
    assert after_rebase.free_equilibrium_ratio <= 1e-8
    assert after_restore.free_equilibrium_ratio <= 1e-8
    assert max(control.constraint_max, after_rebase.constraint_max,
               after_restore.constraint_max) <= 1e-9
    assert not control.production_qualified
    (tmp_path / "s3-fibre-chart.json").write_text(json.dumps({
        "status": "SCOPED_S3_FIBRE_CHART_PARITY_NOT_QUALIFIED",
        "source_prefix_generation": prefix.generation,
        "rebased_generation": charted.generation,
        "target_count": len(targets),
        "final_target": control.target,
        "accepted_step_count": len(control.history),
        "equilibrium_max": max(control.free_equilibrium_ratio,
                               after_rebase.free_equilibrium_ratio,
                               after_restore.free_equilibrium_ratio),
    }, indent=2) + "\n", encoding="utf-8")


@pytest.mark.parametrize("triangles", (False, True))
def test_rebase_refuses_changed_native_fibre_history(request, monkeypatch, triangles):
    model = _yielding_definition(mixed=True, triangles=triangles)
    if not triangles:
        model = replace(model, q4_pullback_policy=ss.Q4_ALGORITHMIC_PULLBACK_V1)
    analysis = owner(request, model)
    prefix = analysis.solve((NodalAction('beam', 105,
                        (0., 0., 1000., 0., 0., 0.)),), targets=(.5, 1.))
    raw = analysis.checkpoint()
    driver = analysis._finite_driver
    saved = driver._beam_origins
    predecessor = driver._beam_predecessors
    saved_arrays = native_history_arrays(analysis._beam_parts, saved, "saved")
    prior_arrays = native_history_arrays(analysis._beam_parts, predecessor, "prior")
    differences = [float(np.max(np.abs(left[1] - right[1]), initial=0.))
                   for left, right in zip(saved_arrays, prior_arrays, strict=True)]
    scales = [max(float(np.max(np.abs(left[1]), initial=0.)),
                  float(np.max(np.abs(right[1]), initial=0.)),
                  np.finfo(float).tiny)
              for left, right in zip(saved_arrays, prior_arrays, strict=True)]
    assert differences[-1] > 64.*np.finfo(float).eps*scales[-1], (differences, scales)
    changed = tuple(prior if type(part[0].section) is PhysicalFibreSection else current
                    for part, current, prior in zip(analysis._beam_parts, saved,
                                                    predecessor, strict=True))
    original_assemble = driver._assemble

    def changed_history(*args, **kwargs):
        response = list(original_assemble(*args, **kwargs))
        response[-1] = changed
        return tuple(response)

    monkeypatch.setattr(driver, "_assemble", changed_history)
    with pytest.raises(ValueError, match="changed (exact )?native beam material history"):
        analysis.rebase_rotations(prefix.generation)
    assert analysis.checkpoint() == raw
    assert analysis.recover() is prefix

    packed = dict(native_history_arrays(analysis._beam_parts, saved, "saved"))
    fibres = packed["saved_fibres"]
    low_changed = None
    for index, component in np.argwhere(fibres[:, (1, 3)] != 0.):
        column = (1, 3)[component]
        trial = fibres.copy()
        trial[index, column] = 0.
        try:
            low_changed = native_histories(analysis._beam_parts,
                {**packed, "saved_fibres": trial}, "saved")
        except ValueError:
            continue
        assert abs(fibres[index, column]) < 64.*np.finfo(float).eps*max(
            float(np.max(np.abs(fibres))), np.finfo(float).tiny)
        break
    assert low_changed is not None, "fixture needs a valid nonzero compensated fibre half"
    changed = low_changed
    with pytest.raises(ValueError, match="changed (exact )?native beam material history"):
        analysis.rebase_rotations(prefix.generation)
    assert analysis.checkpoint() == raw
    assert analysis.recover() is prefix
