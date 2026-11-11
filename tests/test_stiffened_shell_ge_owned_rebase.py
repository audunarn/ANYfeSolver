"""Actual accepted shell-anchor events, not total-vector recanonicalization."""

from dataclasses import replace

import numpy as np
import pytest
from scipy.spatial.transform import Rotation
from scipy.sparse.linalg import spsolve

from anymaterial import DNVC208MaterialCurve
from anyfesolver import stiffened_shell
from anyfesolver._stiffened_shell_ge_experimental import FixedCoordinate
from anyfesolver._stiffened_shell_ge_checkpoint_binary import (
    CHART_MAGIC, decode_chart, encode_chart,
)
from anyfesolver._stiffened_shell_rotation_charts import total_coordinates
from anyfesolver.control import CancellationToken, SolveCancelled
from test_stiffened_shell_ge_experimental import _panel
from test_stiffened_shell_ge_finite_experimental import _actions, _q4_shell_j2_panel


def _owner(definition):
    return stiffened_shell.create_analysis(definition=definition)


@pytest.fixture(scope="module")
def loaded_elastic():
    definition = _panel()
    owner = _owner(definition)
    result = owner.solve(_actions(), (.5, 1.))
    return definition, owner.checkpoint(), result


def _restored(fixture):
    definition, raw, _ = fixture
    owner = _owner(definition)
    owner.restore(raw)
    return owner


def _poses(owner):
    driver = owner._finite_driver
    # Independent reconstruction rather than the production pose helper.
    rows = driver._state.q[:owner._shell_count].reshape(-1, 6)
    frames = Rotation.from_rotvec(rows[:, 3:].copy()).as_matrix()
    if driver._state.shell_anchors is not None:
        frames = frames@driver._state.shell_anchors
    return frames


def _compare_physical(left, right):
    np.testing.assert_allclose(_poses(left), _poses(right), rtol=0., atol=1e-10)
    for name in ("beam_positions", "beam_frames", "cell_frames",
                 "cell_resultants", "multipliers", "support_reactions"):
        np.testing.assert_allclose(getattr(left.recover(), name), getattr(right.recover(), name),
                                   rtol=1e-8, atol=1e-6)
    np.testing.assert_allclose(left.recover().shell_displacement.reshape(-1, 6)[:, :3],
                               right.recover().shell_displacement.reshape(-1, 6)[:, :3],
                               rtol=1e-8, atol=1e-10)
    assert left.recover().accepted_factors == right.recover().accepted_factors


def test_rebase_preserves_physical_prefix_and_continuation(loaded_elastic):
    control, owner = _restored(loaded_elastic), _restored(loaded_elastic)
    old = owner.recover()
    driver = owner._finite_driver
    state = driver._state
    shell_origins, beam_origins = driver._shell_origins, driver._beam_origins
    rebased = owner.rebase_rotations(old.generation)
    assert rebased.generation == old.generation+1
    assert rebased.accepted_factors == old.accepted_factors
    assert driver._shell_origins is shell_origins and driver._beam_origins is beam_origins
    for name in ("frames", "cells", "resultants", "multipliers"):
        assert getattr(driver._state, name) is getattr(state, name)
    np.testing.assert_array_equal(rebased.shell_chart_coordinates.reshape(-1, 6)[:, 3:], 0.)
    np.testing.assert_allclose(rebased.shell_rotation_anchors, _poses(control), rtol=0., atol=1e-15)
    assert not rebased.shell_rotation_anchors.flags.writeable
    assert not rebased.shell_frames.flags.writeable
    for value in (rebased.shell_rotation_anchors, rebased.shell_frames,
                  rebased.shell_chart_coordinates):
        with pytest.raises(ValueError):
            value.setflags(write=True)
    _compare_physical(owner, control)
    saved = owner.checkpoint()
    assert saved.startswith(CHART_MAGIC)
    resumed = _owner(loaded_elastic[0])
    restored = resumed.restore(saved)
    assert restored.generation == rebased.generation
    _compare_physical(owner, resumed)
    for target in (control, resumed):
        target.solve(_actions(), (1.1,))
    _compare_physical(control, resumed)
    # A second event binds the new physical pose and survives a second restore.
    resumed.rebase_rotations(resumed.recover().generation)
    second = _owner(loaded_elastic[0])
    second.restore(resumed.checkpoint())
    _compare_physical(second, control)


def test_stale_cancel_and_partial_support_leave_exact_prefix(loaded_elastic):
    owner = _restored(loaded_elastic)
    raw = owner.checkpoint()
    with pytest.raises(ValueError, match="stale"):
        owner.rebase_rotations(owner.recover().generation-1)
    assert owner.checkpoint() == raw

    class AtCommit(CancellationToken):
        def raise_if_cancelled(self, stage=""):
            if stage == "stiffened-shell.rebase.commit":
                self.cancel("cancel before chart commit")
            super().raise_if_cancelled(stage)

    with pytest.raises(SolveCancelled):
        owner.rebase_rotations(owner.recover().generation, cancellation_token=AtCommit())
    assert owner.checkpoint() == raw
    definition = replace(loaded_elastic[0], fixed=loaded_elastic[0].fixed+
                         (FixedCoordinate("shell", 2, 5),))
    partial = _owner(definition)
    partial.solve(_actions(), (.5,))
    raw = partial.checkpoint()
    with pytest.raises(ValueError, match="partial supports"):
        partial.rebase_rotations(partial.recover().generation)
    assert partial.checkpoint() == raw


def test_chart_snapshot_rejects_semantic_pose_corruption(loaded_elastic):
    owner = _restored(loaded_elastic)
    owner.rebase_rotations(owner.recover().generation)
    raw = owner.checkpoint()
    generation, anchors, witness, inner = decode_chart(raw, owner._identity, len(owner._shell_ids))
    changed = anchors.copy()
    changed[1] = Rotation.from_rotvec((.001, 0., 0.)).as_matrix()@changed[1]
    resealed = encode_chart(owner._identity, generation, changed, witness, inner)
    fresh = _owner(loaded_elastic[0])
    with pytest.raises(ValueError, match="pose"):
        fresh.restore(resealed)
    assert fresh._generation == 0
    with pytest.raises(RuntimeError, match="accepted"):
        fresh.recover()
    with pytest.raises(ValueError):
        fresh.restore(inner)
    corrupted = bytearray(raw); corrupted[-1] ^= 1
    with pytest.raises(ValueError, match="digest"):
        fresh.restore(bytes(corrupted))


@pytest.mark.parametrize("triangles", [False, True])
def test_yielded_histories_survive_actual_anchor_reset_and_restore(triangles):
    definition = _q4_shell_j2_panel()
    if triangles:
        pressure = 20_000.
        curve = DNVC208MaterialCurve(pressure, 1.05*pressure, 1.1*pressure,
                                     .005, .01, 4.*pressure, .2)
        definition = replace(definition, shell=replace(definition.shell,
            q4_elements=(), s3_elements=((1, 2, 5), (1, 5, 4), (2, 3, 6), (2, 6, 5)),
            j2_curve=curve))
    owner = _owner(definition)
    factors, next_factor = ((.05,), .06) if triangles else ((.5, 1.), 1.1)
    result = owner.solve(_actions(), factors, max_iterations=20, max_backtracks=12)
    assert max(max(row) for row in result.shell_layer_alpha) > 0.
    if not triangles:
        assert max(max(row) for row in result.beam_station_accumulated) > 0.
    driver = owner._finite_driver
    origins = (driver._shell_origins, driver._beam_origins)
    control = _owner(definition)
    control.restore(owner.checkpoint())
    owner.rebase_rotations(result.generation)
    assert driver._shell_origins is origins[0] and driver._beam_origins is origins[1]
    fresh = _owner(definition)
    fresh.restore(owner.checkpoint())
    _compare_physical(owner, fresh)
    assert fresh.recover().beam_station_accumulated == owner.recover().beam_station_accumulated
    assert fresh.recover().shell_layer_alpha == owner.recover().shell_layer_alpha
    if triangles:
        for a, b in zip(fresh.recover().shell_s3_material, owner.recover().shell_s3_material):
            np.testing.assert_array_equal(a.alpha, b.alpha)
            np.testing.assert_array_equal(a.plastic_strain, b.plastic_strain)
    for target in (owner, fresh, control):
        target.solve(_actions(), (next_factor,), max_iterations=20, max_backtracks=12)
    _compare_physical(owner, fresh)
    _compare_physical(owner, control)


@pytest.mark.parametrize("triangles", [False, True])
def test_assembled_anchored_actual_update_derivative(loaded_elastic, triangles):
    definition = loaded_elastic[0]
    if triangles:
        definition = replace(definition, shell=replace(definition.shell,
            q4_elements=(), s3_elements=((1, 2, 5), (1, 5, 4), (2, 3, 6), (2, 6, 5))))
        owner = _owner(definition)
        owner.solve(_actions(), (.5,))
    else:
        owner = _restored(loaded_elastic)
    owner.rebase_rotations(owner.recover().generation)
    driver = owner._finite_driver
    state, args = driver._state, (1.1, driver._shell_origins, driver._beam_origins)
    residual, tangent, *_ = driver._assemble(state, *args)
    count = owner._K.shape[0]
    free = np.r_[owner._free, np.arange(count, tangent.shape[0])]
    direction = np.zeros(tangent.shape[0])
    direction[free] = spsolve(tangent[free, :][:, free], -residual[free])
    for amount in (0., .37):
        trial = driver._advance(state, direction, amount)
        _, tangent, *_ = driver._assemble(trial, *args)
        predicted = tangent@direction
        # This Newton direction is a tiny physical load correction. Smaller
        # parameter steps fall below the legacy elastic fit's binary64 floor;
        # use two resolved central-difference levels with unchanged tolerance.
        for h in (.5, .2):
            plus = driver._assemble(driver._advance(trial, direction, h), *args)[0]
            minus = driver._assemble(driver._advance(trial, direction, -h), *args)[0]
            error = (plus-minus)/(2*h)-predicted
            assert np.linalg.norm(error[free])/np.linalg.norm(predicted[free]) < 1e-6
            assert np.linalg.norm(error[count:]) < 1e-9


def test_rebase_preserves_rejected_increment_prefix(monkeypatch):
    base = _q4_shell_j2_panel()
    definition = replace(base, shell=replace(base.shell, q4_elements=(),
        s3_elements=((1, 2, 5), (1, 5, 4), (2, 3, 6), (2, 6, 5))))
    baseline = _owner(definition)
    baseline.solve(_actions(), (.5,), max_iterations=20, max_backtracks=12)
    rebased = _owner(definition)
    rebased.restore(baseline.checkpoint())
    rebased.rebase_rotations(rebased.recover().generation)
    # A physical .55 increment need not fail on every admitted numerical
    # platform. Exercise deterministic chart rejection after forming an actual
    # candidate, rather than requiring an old observed convergence failure.
    from anyfesolver._stiffened_shell_ge_finite_experimental import (
        FinitePointDriver, _RotationStepTooLarge,
    )
    advance=FinitePointDriver._advance
    def reject_candidate(self,*args,**kwargs):
        advance(self,*args,**kwargs)
        raise _RotationStepTooLarge('injected chart rejection')
    monkeypatch.setattr(FinitePointDriver,'_advance',reject_candidate)
    for target in (baseline, rebased):
        prefix = target.checkpoint()
        with pytest.raises(RuntimeError, match="backtrack budget exhausted"):
            target.solve(_actions(), (.55,), max_iterations=20, max_backtracks=2)
        assert target.checkpoint() == prefix


@pytest.mark.parametrize("plastic", [False, True])
def test_mixed_mesh_chart_checkpoint_continues_the_same_physical_history(plastic):
    base = _q4_shell_j2_panel() if plastic else _panel()
    definition = replace(base, shell=replace(base.shell,
        q4_elements=((1, 2, 5, 4),), s3_elements=((2, 3, 6), (2, 6, 5))))
    owner = _owner(definition)
    result = owner.solve(_actions(), (.5, 1.), max_iterations=20, max_backtracks=12)
    control = _owner(definition)
    control.restore(owner.checkpoint())
    owner.rebase_rotations(result.generation)
    restored = _owner(definition)
    restored.restore(owner.checkpoint())
    _compare_physical(owner, restored)
    for target in (control, restored):
        target.solve(_actions(), (1.05,), max_iterations=20, max_backtracks=12)
    _compare_physical(control, restored)


def test_unexpected_evaluation_error_does_not_publish_a_chart(loaded_elastic, monkeypatch):
    owner = _restored(loaded_elastic)
    prefix = owner.checkpoint()
    def failed(*args, **kwargs):
        raise MemoryError("injected resource failure")
    monkeypatch.setattr(owner._finite_driver, "_assemble", failed)
    with pytest.raises(MemoryError, match="resource"):
        owner.rebase_rotations(owner.recover().generation)
    assert owner.checkpoint() == prefix


def test_total_chart_chain_handles_common_rotation_without_relative_log_guard():
    anchors = Rotation.from_rotvec((0., 3., 0.)).as_matrix()[None]
    q = np.array((0., 0., 0., .03, -.01, .02))
    values, differential = total_coordinates(q, anchors)
    direction = np.array((0., 0., 0., .4, -.3, .2))
    h = 1e-6
    plus = total_coordinates(q+h*direction, anchors)[0]
    minus = total_coordinates(q-h*direction, anchors)[0]
    np.testing.assert_allclose((plus-minus)/(2*h), differential@direction, rtol=1e-8, atol=1e-9)
    assert np.linalg.norm(values[3:]) > .9*np.pi
    with pytest.raises(ValueError, match="pi branch"):
        total_coordinates(np.zeros(6), Rotation.from_rotvec((0., np.pi, 0.)).as_matrix()[None])
