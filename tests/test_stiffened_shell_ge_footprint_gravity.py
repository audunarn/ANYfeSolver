"""Bounded gravity/selected-node integration, not M4 qualification."""
from dataclasses import replace
from copy import copy
from types import SimpleNamespace
import json

import numpy as np
import pytest

from anyfesolver import b3_ge, stiffened_shell as ss
from anyfesolver.control import CancellationToken, SolveCancelled
from anyfesolver import _stiffened_shell_ge_finite_experimental as finite
from anyfesolver._stiffened_shell_ge_body_load import admits_owner
from anyfesolver._ge_beam3_native_definition import NativeBeamDefinition
from anyfesolver._stiffened_shell_rotation_charts import physical_frames
from anyfesolver._stiffened_shell_nodal_footprint import CHART_ADMISSION_POLICY
from test_stiffened_shell_nodal_rigid_footprint import _definition as footprint_definition
from test_stiffened_shell_native_gravity import inertia
from test_stiffened_shell_footprint_increment_charts import _histories, _signature
from test_stiffened_shell_ge_gravity_owner import balance


def _definition(*, only=False, mixed=False, yield_force=None):
    base = footprint_definition(mixed)
    beams = []
    for declaration in base.beams:
        element, _ = declaration.instantiate()
        if yield_force is not None:
            section = element.section
            reference = element.operator.reference
            beams.append(b3_ge.define_beam('b3-ge', element.element_id, element.node_ids,
                reference.coordinates, reference.nodal_triads,
                b3_ge.EllipsoidalGeneralizedSection(section.elastic, section.metric,
                                                   yield_force, section.hardening),
                inertia(offset=(0., .03, -.02))))
        else:
            beams.append(NativeBeamDefinition.capture(element, inertia(offset=(0., .03, -.02))))
    return replace(base, beams=tuple(beams), points=() if only else base.points,
                   footprint_chart_policy=ss.GE_NODAL_FOOTPRINT_INCREMENT_ORIGIN_CHART_V2)


def _owner(request, definition=None):
    owner = ss.create_analysis(definition=definition or _definition())
    request.addfinalizer(owner.close)
    return owner


def _load(owner, acceleration=(0., 0., -9.81)):
    return ss.GEBodyLoad(owner.preflight()['identity'], acceleration)


def _solve(owner, load, targets):
    return owner.solve(load, targets, max_iterations=20, max_backtracks=12)


def _group(owner, result):
    group, = owner.recover(selection=ss.NodalRigidFootprintSelection(('tip-pad',)))
    assert group.generation == result.generation and group.configuration == 'current'
    expected = tuple(indices for _, indices in owner._footprint_bindings)[0]
    assert tuple(p.point_index for p in group.point_wrenches) == expected
    np.testing.assert_allclose((np.asarray(group.shell_action)+group.beam_action)/
                              np.array((100.,)*3+(200.,)*3), 0., rtol=0., atol=1e-8)
    assert np.linalg.norm(group.beam_action) > 1e-3
    return group


@pytest.mark.parametrize('only', (False, True))
def test_admitted_gravity_balance_and_selected_recovery(request, only):
    owner = _owner(request, _definition(only=only))
    assert admits_owner(owner) and owner.capabilities['body_acceleration']
    load = _load(owner)
    context = load.compile(owner)
    assert context.freeze_trial_load_norm is True
    assert any(np.linalg.norm(kernel.cell_first_moments) > 0. for kernel, _ in context.beams)
    linear = owner.solve_linear(load)
    balance(owner, linear)
    # A fresh finite owner retains the same primal coordinates and rigid rows.
    owner = _owner(request, _definition(only=only))
    result = _solve(owner, load, (.1, .2))
    balance(owner, result)
    _group(owner, result)
    assert np.linalg.norm(result.multipliers) > 1e-3
    assert result.free_equilibrium_ratio <= 1e-8 and result.constraint_max <= 1e-9


def test_full_actual_update_tangent_and_conservative_body_work(request, record_property):
    owner = _owner(request)
    result = _solve(owner, _load(owner), (.1, .2))
    _group(owner, result)
    driver = owner._finite_driver
    state = driver._increment_origin(driver._state)
    assembled = driver._assemble(state, .2, driver._shell_origins, driver._beam_origins)
    count = len(state.q); shell = owner._shell_count; nodal = shell+driver._beam_nodal
    inputs = np.ones(len(assembled[0])); rows = np.ones_like(inputs)
    inputs[:nodal] = np.tile((2., 2., 2., 1., 1., 1.), nodal//6)
    rows[:nodal] = np.tile((100., 100., 100., 200., 200., 200.), nodal//6)
    for member in range(len(owner._beam_parts)):
        base = nodal+24*member
        inputs[base:base+24] = (1.,)*6+(100.,)*6+(200.,)*12
        rows[base:base+24] = (200.,)*6+(2.,)*6+(1.,)*12
    inputs[count:] = np.tile((100.,)*3+(200.,)*3, len(owner._connections))
    rows[count:] = np.tile((2.,)*3+(1.,)*3, len(owner._connections))
    roles = (slice(0, shell), slice(shell, nodal), slice(nodal, count), slice(count, None))
    errors = []; work_errors = []
    force, jacobian, _ = driver._external.evaluate(state)
    assert np.linalg.norm(force[nodal:nodal+6]) > .01
    for phase in (.3, 1.1):
        direction = inputs*np.sin(np.arange(len(inputs))+.4+phase)
        direction[owner._fixed] = 0.
        predicted = assembled[1]@direction/rows
        for step in (1e-5, 3e-6, 1e-6):
            plus = driver._advance(state, direction, step)
            minus = driver._advance(state, direction, -step)
            rp = driver._assemble(plus, .2, driver._shell_origins, driver._beam_origins)[0]
            rm = driver._assemble(minus, .2, driver._shell_origins, driver._beam_origins)[0]
            actual = (rp-rm)/(2*step)/rows
            errors.append([float(np.linalg.norm((actual-predicted)[role])/
                max(1e-12, np.linalg.norm(predicted[role]))) for role in roles])
            fp, _, wp = driver._external.evaluate(plus)
            fm, _, wm = driver._external.evaluate(minus)
            np.testing.assert_allclose((fp-fm)/(2*step), jacobian@direction[:count], rtol=1e-7, atol=1e-8)
            derivative = (wp.work-wm.work)/(2*step)
            expected = force@direction[:count]
            work_errors.append(abs(derivative-expected)/max(1., abs(expected)))
    record_property('complete_gravity_footprint_role_errors', json.dumps(errors))
    record_property('body_work_derivative_errors', json.dumps(work_errors))
    assert np.isfinite(errors).all() and np.max(errors) <= 1e-6
    assert np.max(work_errors) <= 1e-6


def _physical_parity(left_owner, left, right_owner, right):
    for name, scale in (('beam_positions', 2.), ('beam_frames', 1.), ('cell_frames', 1.)):
        np.testing.assert_allclose(getattr(left, name)/scale, getattr(right, name)/scale, rtol=0., atol=1e-9)
    np.testing.assert_allclose(left.cell_resultants/np.array((100.,)*6+(200.,)*12),
                              right.cell_resultants/np.array((100.,)*6+(200.,)*12), rtol=0., atol=1e-9)
    for name in ('multipliers', 'support_reactions'):
        np.testing.assert_allclose(getattr(left, name).reshape(-1, 6)/np.array((100.,)*3+(200.,)*3),
                                  getattr(right, name).reshape(-1, 6)/np.array((100.,)*3+(200.,)*3), rtol=0., atol=1e-9)
    for owner, reference in ((left_owner, right_owner),):
        state = owner._finite_driver._state; other = reference._finite_driver._state
        np.testing.assert_allclose(state.q[:owner._shell_count].reshape(-1, 6)[:, :3]/2.,
                                  other.q[:reference._shell_count].reshape(-1, 6)[:, :3]/2., rtol=0., atol=1e-9)
        np.testing.assert_allclose(physical_frames(state.q[:owner._shell_count], state.shell_anchors),
                                  physical_frames(other.q[:reference._shell_count], other.shell_anchors), rtol=0., atol=1e-9)
    ld, rd = left_owner._finite_driver, right_owner._finite_driver
    # Material histories are identical; predecessor coordinates may use different
    # charts after a rebase. Compare their physical configuration independently.
    for name in ('_shell_origins', '_beam_origins', '_shell_predecessors', '_beam_predecessors'):
        assert _signature(getattr(ld, name)) == _signature(getattr(rd, name))
    np.testing.assert_allclose(ld._shell_predecessor_q.reshape(-1, 6)[:, :3]/2.,
        rd._shell_predecessor_q.reshape(-1, 6)[:, :3]/2., rtol=0., atol=1e-9)
    np.testing.assert_allclose(physical_frames(ld._shell_predecessor_q, ld._shell_predecessor_anchors),
        physical_frames(rd._shell_predecessor_q, rd._shell_predecessor_anchors), rtol=0., atol=1e-9)
    assert left.accepted_factors == right.accepted_factors
    np.testing.assert_allclose(left.body_load.applied_moment_about_origin,
                              right.body_load.applied_moment_about_origin, rtol=0., atol=2e-5)
    assert left.body_load.work == pytest.approx(right.body_load.work, rel=1e-8, abs=1e-8)


def test_checkpoint_rebase_continuation_and_history_parity(request):
    owner = _owner(request); load = _load(owner)
    prefix = _solve(owner, load, (.1, .2)); saved = owner.checkpoint()
    restored = _owner(request); restored.restore(saved)
    assert restored.checkpoint() == saved
    reset = _owner(request); reset.restore(saved)
    histories = _histories(reset._finite_driver)
    event = reset.rebase_rotations(prefix.generation)
    assert _histories(reset._finite_driver) == histories
    np.testing.assert_allclose(event.body_load.applied_force, prefix.body_load.applied_force, rtol=0., atol=0.)
    np.testing.assert_allclose(event.body_load.applied_moment_about_origin,
                              prefix.body_load.applied_moment_about_origin, rtol=0., atol=0.)
    assert event.body_load.work == prefix.body_load.work
    charted = _owner(request); charted.restore(reset.checkpoint())
    assert charted.checkpoint() == reset.checkpoint()
    final = _solve(owner, load, (.3,))
    for branch in (restored, reset, charted):
        successor = _solve(branch, load, (.3,))
        _physical_parity(owner, final, branch, successor)
        balance(branch, successor); _group(branch, successor)


@pytest.mark.parametrize('stage', ('stiffened-shell.finite.publish', 'stiffened-shell.rebase.commit'))
def test_cancellation_failure_and_changed_acceleration_preserve_prefix(request, monkeypatch, stage):
    owner = _owner(request); load = _load(owner)
    prefix = _solve(owner, load, (.1, .2)); saved = owner.checkpoint()
    state = owner._finite_driver._state; histories = _histories(owner._finite_driver)
    changed = _load(owner, (0., 0., -9.8))
    with pytest.raises(ValueError, match='changed after first increment'):
        _solve(owner, changed, (.3,))
    fresh = _owner(request)
    fresh.solve_linear(_load(fresh, (0., 0., -9.8)))
    fresh_saved = fresh.checkpoint()
    with pytest.raises(RuntimeError, match='requires a fresh'):
        fresh.restore(saved)
    assert fresh.checkpoint() == fresh_saved
    assemble = owner._finite_driver._assemble
    def fail(*args, **kwargs):
        assemble(*args, **kwargs)
        raise RuntimeError('retained injected successor failure')
    with monkeypatch.context() as patch:
        patch.setattr(owner._finite_driver, '_assemble', fail)
        with pytest.raises(RuntimeError, match='retained injected'):
            _solve(owner, load, (.3,))
    assert owner.checkpoint() == saved and owner._finite_driver._state is state
    token = CancellationToken(); safe_point = finite.cancellation_safe_point
    def cancel(value, where):
        if where == stage: token.cancel('gravity footprint cancellation')
        return safe_point(value, where)
    monkeypatch.setattr(finite, 'cancellation_safe_point', cancel)
    with pytest.raises(SolveCancelled, match='gravity footprint cancellation'):
        if stage.endswith('publish'):
            owner.solve(load, (.3,), cancellation_token=token, max_iterations=20, max_backtracks=12)
        else:
            owner.rebase_rotations(prefix.generation, cancellation_token=token)
    assert owner._generation == prefix.generation and owner.recover() is prefix
    assert owner.checkpoint() == saved and owner._finite_driver._state is state
    assert _histories(owner._finite_driver) == histories


@pytest.mark.parametrize('variant', ('mixed', 's3', 'unrebased', 'density', 'inertia', 'law', 'curved', 'material', 'line'))
def test_narrow_admission_and_capability_parity(request, variant):
    definition = _definition(mixed=True if variant == 'mixed' else 'all-s3' if variant == 's3' else False)
    if variant == 'unrebased':
        definition = replace(definition, footprint_chart_policy=CHART_ADMISSION_POLICY)
    if variant == 'inertia':
        beams = []
        for declaration in definition.beams:
            element, _ = declaration.instantiate(); beams.append(NativeBeamDefinition.capture(element, np.diag((1., 2., 1., 1., 1., 1.))))
        definition = replace(definition, beams=tuple(beams))
    # Unsupported combinations that construction already excludes are tested
    # in detached views, without weakening those earlier constructor guards.
    owner = _owner(request, definition)
    if variant in ('density', 'curved', 'material', 'line'):
        view = copy(owner)
        if variant == 'density': view._definition = replace(definition, shell=replace(definition.shell, density_kg_m3=0.))
        elif variant == 'curved': view._definition = replace(definition, shell_reference_policy='curved-unsupported')
        elif variant == 'material': view._definition = replace(definition, shell=replace(definition.shell, j2_curve=object()))
        else: view._definition = replace(definition, lines=(SimpleNamespace(multiplier_space='unsupported'),))
        owner = view
    if variant == 'law':
        section = owner._beam_parts[0][0].section
        object.__setattr__(section, 'yield_force', section.yield_force/2.)
    assert not admits_owner(owner)
    assert not owner.capabilities['body_acceleration']
    with pytest.raises(ValueError, match='GE body'):
        _load(owner).compile(owner)
    assert owner._generation == 0 and owner._finite_driver is None


def test_yielding_generalized_response_rejected_without_publish(request):
    owner = _owner(request, _definition(yield_force=1e-5))
    assert owner.capabilities['body_acceleration']
    with pytest.raises(ValueError, match='elastic beam response only'):
        _solve(owner, _load(owner), (.1,))
    assert owner._generation == 0 and owner._finite_driver is None


def test_body_and_staged_programme_remain_uncomposable(request):
    owner = _owner(request); load = _load(owner)
    _solve(owner, load, (.1,))
    saved = owner.checkpoint()
    with pytest.raises(ValueError, match='cannot be mixed'):
        _solve(owner, (), (.2,))
    assert owner.checkpoint() == saved
