"""Elastic footprint chart-selection parity, not broad F06 qualification."""
from dataclasses import fields, is_dataclass, replace
from hashlib import sha256
import json

import numpy as np
import pytest

from anyfesolver import stiffened_shell as ss
from anyfesolver import _stiffened_shell_ge_finite_experimental as finite
from anyfesolver.control import CancellationToken, SolveCancelled
from anyfesolver._stiffened_shell_rotation_charts import physical_frames
from test_stiffened_shell_nodal_rigid_footprint import _definition, _owner, _group_checks


def _charted_definition(mixed=False):
    return replace(_definition(mixed), footprint_chart_policy=ss.GE_NODAL_FOOTPRINT_INCREMENT_ORIGIN_CHART_V2)


def _signature(value):
    if isinstance(value, np.ndarray):
        return (value.shape, value.dtype.str, sha256(value.tobytes()).hexdigest())
    if is_dataclass(value):
        return tuple((field.name, _signature(getattr(value, field.name))) for field in fields(value))
    if isinstance(value, dict):
        return tuple((key, _signature(part)) for key, part in sorted(value.items()))
    if isinstance(value, (tuple, list)):
        return tuple(_signature(part) for part in value)
    return value


def _histories(driver):
    return _signature(tuple(getattr(driver, name) for name in (
        "_shell_origins", "_beam_origins", "_shell_predecessors",
        "_beam_predecessors", "_shell_predecessor_q", "_shell_predecessor_anchors")))


@pytest.mark.parametrize("mixed", (False, True, "all-s3"))
def test_public_four_branch_successor_preserves_strict_physical_parity(request, mixed):
    definition = _charted_definition(mixed)
    original = _owner(request, definition)
    actions = (ss.NodalAction("beam", 105, (0., 0., 10., 0., 0., 0.)),)
    prefix = original.solve(actions, (.25, .5), max_iterations=20, max_backtracks=12)
    saved = original.checkpoint()
    restored = _owner(request, definition); restored.restore(saved)
    reset = _owner(request, definition); reset.restore(saved)
    driver = reset._finite_driver
    before = _histories(driver)
    q = driver._state.q.copy()
    poses = physical_frames(q[:reset._shell_count], driver._state.shell_anchors)
    event = reset.rebase_rotations(expected_generation=prefix.generation)
    assert _histories(driver) == before
    assert event.generation == prefix.generation+1
    np.testing.assert_array_equal(physical_frames(driver._state.q[:reset._shell_count],
                                                driver._state.shell_anchors), poses)
    np.testing.assert_array_equal(driver._state.q[:reset._shell_count].reshape(-1, 6)[:, :3],
                                  q[:reset._shell_count].reshape(-1, 6)[:, :3])
    reset_restored = _owner(request, definition); reset_restored.restore(reset.checkpoint())
    branches = (original, restored, reset, reset_restored)
    results = [branch.solve(actions, (1.,), max_iterations=20, max_backtracks=12) for branch in branches]
    reference = results[0]
    for branch, result in zip(branches, results):
        _group_checks(branch, result)
        assert result.status == "completed" and result.accepted_factors == (.25, .5, 1.)
        assert result.iterations == reference.iterations
        assert result.free_equilibrium_ratio <= 1e-8 and result.constraint_max <= 1e-9
        np.testing.assert_allclose(result.beam_positions/2., reference.beam_positions/2., rtol=0., atol=1e-9)
        np.testing.assert_allclose(result.shell_displacement.reshape(-1, 6)[:, :3]/2.,
                                  reference.shell_displacement.reshape(-1, 6)[:, :3]/2.,
                                  rtol=0., atol=1e-9)
        for name in ("beam_frames", "cell_frames", "shell_frames"):
            np.testing.assert_allclose(getattr(result, name), getattr(reference, name), rtol=0., atol=1e-9)
        np.testing.assert_allclose(result.cell_resultants/np.array((10.,)*6+(20.,)*12),
                                  reference.cell_resultants/np.array((10.,)*6+(20.,)*12), rtol=0., atol=1e-9)
        for name in ("multipliers", "support_reactions"):
            scale = np.array((10.,)*3+(20.,)*3)
            np.testing.assert_allclose(getattr(result, name).reshape(-1, 6)/scale,
                                      getattr(reference, name).reshape(-1, 6)/scale, rtol=0., atol=1e-9)
        assert _signature(branch._finite_driver._beam_origins) == _signature(original._finite_driver._beam_origins)
        successor = _owner(request, definition); successor.restore(branch.checkpoint())
        assert successor.checkpoint() == branch.checkpoint()


@pytest.mark.parametrize("mixed", (False, True, "all-s3"))
def test_detached_increment_origin_owns_chart_and_preserves_predecessors(request, mixed):
    owner = _owner(request, _charted_definition(mixed))
    actions = (ss.NodalAction("beam", 105, (0., 0., 10., 0., 0., 0.)),)
    owner.solve(actions, (.25, .5), max_iterations=20, max_backtracks=12)
    driver = owner._finite_driver; state = driver._state
    before = _histories(driver); saved = owner.checkpoint(); generation = owner._generation
    detached = driver._increment_origin(state)
    assert detached is not state and driver._state is state
    assert _histories(driver) == before and owner.checkpoint() == saved and owner._generation == generation
    assert not np.shares_memory(detached.q, state.q)
    assert not detached.q.flags.writeable and not detached.shell_anchors.flags.writeable
    np.testing.assert_array_equal(physical_frames(state.q[:owner._shell_count], state.shell_anchors),
        physical_frames(detached.q[:owner._shell_count], detached.shell_anchors))
    for name in ("frames", "cells", "resultants", "multipliers"):
        assert getattr(detached, name) is getattr(state, name)
    # The fixed anchor belongs to this increment, not to each perturbed trial.
    delta = np.zeros(len(state.q)+len(state.multipliers)); delta[6*2+4] = .1
    moved = driver._advance(detached, delta, .01)
    assert moved.shell_anchors is detached.shell_anchors
    assert np.any(moved.q[:owner._shell_count].reshape(-1, 6)[:, 3:] != 0.)


@pytest.mark.parametrize("mixed", (False, True, "all-s3"))
def test_partial_rotation_support_rejects_finite_work_atomically(request, mixed):
    definition = _charted_definition(mixed)
    definition = replace(definition, fixed=definition.fixed+(ss.FixedCoordinate("shell", 2, 3),))
    owner = _owner(request, definition)
    actions = (ss.NodalAction("beam", 105, (0., 0., 10., 0., 0., 0.)),)
    with pytest.raises(ValueError, match="full rotational triples"):
        owner.solve(actions, (.25, .5), max_iterations=20, max_backtracks=12)
    assert owner._finite_driver is None and owner._generation == 0 and owner._accepted is None


@pytest.mark.parametrize("stage", ("stiffened-shell.rebase.prepare", "stiffened-shell.rebase.commit"))
def test_cancel_chart_event_preserves_complete_accepted_prefix(request, monkeypatch, stage):
    owner = _owner(request, _charted_definition())
    actions = (ss.NodalAction("beam", 105, (0., 0., 10., 0., 0., 0.)),)
    result = owner.solve(actions, (.25, .5), max_iterations=20, max_backtracks=12)
    saved = owner.checkpoint(); state = owner._finite_driver._state; before = _histories(owner._finite_driver)
    token = CancellationToken(); safe_point = finite.cancellation_safe_point
    def cancel(value, where):
        if where == stage: token.cancel("chart cancellation")
        return safe_point(value, where)
    monkeypatch.setattr(finite, "cancellation_safe_point", cancel)
    with pytest.raises(SolveCancelled, match="chart cancellation"):
        owner.rebase_rotations(expected_generation=result.generation, cancellation_token=token)
    assert owner._generation == result.generation and owner.recover() is result
    assert owner._finite_driver._state is state and owner.checkpoint() == saved
    assert _histories(owner._finite_driver) == before


@pytest.mark.parametrize("mixed", (False, True, "all-s3"))
def test_rejected_and_cancelled_successor_discard_detached_chart(request, monkeypatch, mixed):
    owner = _owner(request, _charted_definition(mixed))
    actions = (ss.NodalAction("beam", 105, (0., 0., 10., 0., 0., 0.)),)
    result = owner.solve(actions, (.25, .5), max_iterations=20, max_backtracks=12)
    saved = owner.checkpoint(); state = owner._finite_driver._state; before = _histories(owner._finite_driver)
    with pytest.raises(ss.FiniteIncrementFailure):
        owner.solve(actions, (1.,), max_iterations=1, max_backtracks=12)
    assert owner.checkpoint() == saved and owner._finite_driver._state is state
    token = CancellationToken(); safe_point = finite.cancellation_safe_point
    def cancel(value, where):
        if where == "stiffened-shell.finite.publish": token.cancel("successor cancellation")
        return safe_point(value, where)
    monkeypatch.setattr(finite, "cancellation_safe_point", cancel)
    with pytest.raises(SolveCancelled, match="successor cancellation"):
        owner.solve(actions, (1.,), cancellation_token=token, max_iterations=20, max_backtracks=12)
    assert owner._generation == result.generation and owner.recover() is result
    assert owner._finite_driver._state is state and owner.checkpoint() == saved
    assert _histories(owner._finite_driver) == before


@pytest.mark.parametrize("mixed", (False, True, "all-s3"))
def test_loaded_origin_chart_actual_update_derivative(request, record_property, mixed):
    owner = _owner(request, _charted_definition(mixed))
    actions = (ss.NodalAction("beam", 105, (0., 0., 10., 0., 0., 0.)),)
    owner.solve(actions, (.25, .5), max_iterations=20, max_backtracks=12)
    driver = owner._finite_driver; state = driver._increment_origin(driver._state)
    response = driver._assemble(state, .5, driver._shell_origins, driver._beam_origins)
    count = len(state.q); shell = owner._shell_count; nodal = shell+driver._beam_nodal
    input_scale = np.ones(len(response[0])); row_scale = np.ones_like(input_scale)
    input_scale[:nodal] = np.tile((2., 2., 2., 1., 1., 1.), nodal//6)
    row_scale[:nodal] = np.tile((100., 100., 100., 200., 200., 200.), nodal//6)
    for member in range(len(owner._beam_parts)):
        base = nodal+24*member
        input_scale[base:base+24] = (1.,)*6+(100.,)*6+(200.,)*12
        row_scale[base:base+24] = (200.,)*6+(2.,)*6+(1.,)*12
    input_scale[count:] = np.tile((100., 100., 100., 200., 200., 200.), len(owner._connections))
    row_scale[count:] = np.tile((2., 2., 2., 1., 1., 1.), len(owner._connections))
    roles = (slice(0, shell), slice(shell, nodal), slice(nodal, count), slice(count, None))
    errors = []
    for phase in (.3, 1.1):
        direction = input_scale*np.sin(np.arange(len(input_scale))+.4+phase)
        direction[owner._fixed] = 0.
        predicted = response[1]@direction/row_scale
        for step in (1e-5, 3e-6, 1e-6):
            plus = driver._advance(state, direction, step)
            minus = driver._advance(state, direction, -step)
            assert plus.shell_anchors is state.shell_anchors and minus.shell_anchors is state.shell_anchors
            rp = driver._assemble(plus, .5, driver._shell_origins, driver._beam_origins)[0]
            rm = driver._assemble(minus, .5, driver._shell_origins, driver._beam_origins)[0]
            actual = (rp-rm)/(2*step)/row_scale
            errors.append([float(np.linalg.norm((actual-predicted)[role])/
                max(1e-12, np.linalg.norm(predicted[role]))) for role in roles])
    record_property("loaded_origin_actual_update_role_errors", json.dumps(errors))
    assert np.isfinite(errors).all() and np.max(errors) <= 1e-6


@pytest.mark.parametrize("mixed", (False, True, "all-s3"))
def test_current_chart_corruption_and_cross_policy_restore_reject_atomically(request, mixed):
    from anyfesolver._stiffened_shell_ge_checkpoint_binary import decode_chart, encode_chart
    owner = _owner(request, _charted_definition(mixed))
    legacy = _owner(request, _definition(mixed))
    actions = (ss.NodalAction("beam", 105, (0., 0., 10., 0., 0., 0.)),)
    result = owner.solve(actions, (.25, .5, 1.), max_iterations=20, max_backtracks=12)
    saved = owner.checkpoint(); before = _histories(owner._finite_driver); state = owner._finite_driver._state
    generation, anchors, witness, inner = decode_chart(saved, owner._identity, len(owner._shell_ids))
    assert generation == result.generation == 3
    bad_witness = witness.copy(); bad_witness[2] = np.eye(3)
    corrupt_inner = inner[:-1]+bytes((inner[-1]^1,))
    candidates = (encode_chart(owner._identity, generation, anchors, bad_witness, inner),
                  encode_chart(owner._identity, 2, anchors, witness, inner),
                  encode_chart(owner._identity, generation, anchors, witness, corrupt_inner))
    for raw in candidates:
        assert sha256(raw[:-32]).digest() == raw[-32:]
        target = _owner(request, _charted_definition(mixed))
        with pytest.raises(ValueError): target.restore(raw)
        assert target._generation == 0 and target._accepted is None and target._finite_driver is None
        assert owner.checkpoint() == saved and owner.recover() is result
        assert owner._finite_driver._state is state and _histories(owner._finite_driver) == before
    legacy.solve(actions, (.25, .5), max_iterations=20, max_backtracks=12)
    legacy_saved = legacy.checkpoint()
    target = _owner(request, _charted_definition(mixed))
    with pytest.raises(ValueError): target.restore(legacy_saved)
    legacy_target = _owner(request, _definition(mixed))
    with pytest.raises(ValueError): legacy_target.restore(saved)
    assert owner.checkpoint() == saved and legacy.checkpoint() == legacy_saved
    # Valid digest and the V1 model identity still cannot grant chart admission.
    framed_v1 = encode_chart(legacy._identity, generation, anchors, witness, inner)
    with pytest.raises(ValueError, match="chart checkpoint restore is unsupported"):
        legacy_target.restore(framed_v1)
    assert legacy.checkpoint() == legacy_saved
