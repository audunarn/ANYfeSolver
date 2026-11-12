"""Separate candidate evidence for explicit director-integrated S355 coupling.

The frozen default module and its caches are not changed. Its exact test/oracle
code is loaded into a private module namespace with a private fixture cache;
only the interpolated public constructor is supplied the opt-in trace keyword.
These candidate checks do not qualify the default trace or analytical families.
"""
from __future__ import annotations

import copy
import importlib.util
from pathlib import Path

import numpy as np
import pytest

from anyfesolver import DisplacementControl, solve_static_nonlinear
from anyfesolver.control import CancellationToken, SolveCancelled
from anyfesolver.mesh_gen import InterpolatedBeamShellMPCElement


_spec = importlib.util.spec_from_file_location(
    '_slice3_director_private_oracles',
    Path(__file__).with_name('test_nonlinear_beam_shell_slice3_continuation.py'))
assert _spec is not None and _spec.loader is not None
base = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(base)
# This is the new private module's cache, never the collected default module's.
assert base._CACHE == {}
_original_build = base._build_ep_model


def _director_constructor(*args, **kwargs):
    assert 'connection_trace' not in kwargs
    return InterpolatedBeamShellMPCElement(*args, **kwargs,
                                          connection_trace='director_integrated')


base.InterpolatedBeamShellMPCElement = _director_constructor


def _build(coupling, scale=1.):
    assert coupling == 'interp', 'candidate requires actual three-beam nonmatching topology'
    model, load, probe, grid = _original_build(coupling, scale)
    interpolated = [el for el in model.mesh.elements.values()
                    if isinstance(el, InterpolatedBeamShellMPCElement)]
    assert len(interpolated) == 2
    for element in interpolated:
        assert element.connection_trace == 'director_integrated'
        descriptor = element.get_nonlinear_mpc_descriptor(model.mesh)
        # The descriptor encodes policy through admitted geometry, not a label.
        assert descriptor is not None and descriptor['director_trace_geometry'] is not None
        assert element.rotating_eccentricity is True
    return model, load, probe, grid


base._build_ep_model = _build


def test_director_preyield_matches_high_yield_control():
    base.test_preyield_plastic_matches_elastic_control_exactly('interp')


def test_director_preyield_matches_distinct_linear_elastic_stored_stress():
    base.test_preyield_plastic_matches_distinct_linear_elastic_material('interp')


def test_director_full_target_plasticity_equilibrium_and_rhs_budget(monkeypatch, tmp_path):
    # Exact existing assertions and independent per-attempt RHS instrumentation;
    # accepted and terminal raw arrays survive a failure/timeout.
    base.test_dc_full_target_plasticity_and_equilibrium('interp', monkeypatch, tmp_path)
    case = base._full_case('interp')
    mapping = base._equilibrium_oracle(case)['mpc_map']
    assert sum(entry.director_trace_geometry is not None for entry in mapping.entries) == 2


def test_director_restart_reproduces_full_physical_history():
    # Original 6-to-.10 +5-to-11/60, original milestone/history tolerances.
    base.test_dc_resume_reproduces_uninterrupted_run('interp')


def test_director_cancelled_same_model_and_inputs_still_deliver(monkeypatch):
    half = base._half_case('interp')
    base._assert_completed_dc_case(half, -.10)
    model, load, probe, grid = _build('interp')
    base._add_tip_load(model, load, grid)
    u = half['result'].displacements.copy()
    states = copy.deepcopy(dict(half['result'].element_states))
    before_u, before_states = u.copy(), copy.deepcopy(states)
    token = CancellationToken()
    events, snapshots = [], []
    import anyfesolver.nonlinear_static as ns
    real_snapshot = ns._increment_snapshot

    def observe(*args, **kwargs):
        value = real_snapshot(*args, **kwargs)
        snapshots.append((value, value.displacements.copy(), copy.deepcopy(dict(value.element_states))))
        return value

    def cancel(event):
        if event.event_type == 'nonlinear_static_step':
            events.append(event)
            if len(events) == 2:
                token.cancel('director candidate cancellation after two accepted increments')

    common = dict(control='displacement', displacement_control=DisplacementControl(
        node_id=probe, dof='uz', target_displacement=base.DC_TARGET),
        kinematics='corotational', num_steps=base.RESUME_STEPS,
        max_iterations=base.DC_MAX_ITER, tolerance=base.DC_TOL,
        initial_displacements=u, initial_element_states=states,
        equilibrate_initial_state=False, record_increment_snapshots=True)
    with monkeypatch.context() as context:
        context.setattr(ns, '_increment_snapshot', observe)
        with pytest.raises(SolveCancelled):
            solve_static_nonlinear(model, load, **common, cancellation_token=token,
                                   progress_callback=cancel)
    assert len(events) == len(snapshots) == 2
    np.testing.assert_array_equal(u, before_u)
    base._assert_states_equal(states, before_states)
    for snapshot, saved_u, saved_states in snapshots:
        np.testing.assert_array_equal(snapshot.displacements, saved_u)
        base._assert_states_equal(dict(snapshot.element_states), saved_states)
        base._assert_material_schema(half, snapshot.element_states)
    # Same model and identical caller objects; no replacement seed/history.
    reused = solve_static_nonlinear(model, load, **common)
    case = dict(model=model,load=load,probe=probe,grid=grid,result=reused,coupling='interp')
    base._assert_completed_dc_case(case, base.DC_TARGET)
    np.testing.assert_array_equal(u, before_u)
    base._assert_states_equal(states, before_states)
    clean = base._resumed_case('interp')['result']
    np.testing.assert_allclose(reused.displacements, clean.displacements, rtol=0., atol=1e-12)
    assert abs(reused.load_factor-clean.load_factor) <= 1e-12
    base._assert_states_allclose(reused.element_states, clean.element_states, physical=True)
    for snapshot, saved_u, saved_states in snapshots:
        np.testing.assert_array_equal(snapshot.displacements, saved_u)
        base._assert_states_equal(dict(snapshot.element_states), saved_states)


def test_director_incompatible_restart_refused_without_input_mutation():
    half = base._half_case('interp')
    model, load, probe, grid = _build('interp')
    base._add_tip_load(model, load, grid)
    u = half['result'].displacements.copy()
    # Actual interior slave, not a coincident/direct endpoint.
    u[model.mesh.dof_manager.get_node_dofs(101)[2]] += 1e-3
    states = copy.deepcopy(dict(half['result'].element_states))
    before_u, before_states = u.copy(), copy.deepcopy(states)
    with pytest.raises(ValueError):
        solve_static_nonlinear(model, load, control='displacement',
            displacement_control=DisplacementControl(node_id=probe,dof='uz',target_displacement=base.DC_TARGET),
            kinematics='corotational',num_steps=base.RESUME_STEPS,
            max_iterations=base.DC_MAX_ITER,tolerance=base.DC_TOL,
            initial_displacements=u,initial_element_states=states,equilibrate_initial_state=False)
    np.testing.assert_array_equal(u, before_u)
    base._assert_states_equal(states, before_states)


def test_director_stopped_negative_run_excludes_rejected_history(monkeypatch):
    """Deliberately insufficient1/1 controls are a separate negative test."""
    import inspect
    import anyfesolver.nonlinear_static as ns
    import anyfesolver.nonlinear_performance as performance

    half = base._half_case('interp')
    model,load,probe,grid = _build('interp')
    base._add_tip_load(model,load,grid)
    u=half['result'].displacements.copy()
    states=copy.deepcopy(dict(half['result'].element_states))
    before_u,before_states=u.copy(),copy.deepcopy(states)
    accepted_origin=[]
    real_assemble=ns._assemble_nonlinear_system
    original_dc=(performance._ORIGINAL_DISPLACEMENT_SOLVER
                 if performance._ORIGINAL_DISPLACEMENT_SOLVER is not None
                 else ns._solve_static_displacement_control)
    original_dc_code=original_dc.__code__

    def observe_origin(*args, **kwargs):
        frame=inspect.currentframe().f_back
        try:
            # The persistent performance provider inserts wrapper frames and
            # retains the original DC callable captured above.
            while frame is not None and frame.f_code is not original_dc_code:
                frame=frame.f_back
            if not accepted_origin and frame is not None:
                # Initial load recovery observes the already projected accepted
                # origin before any Newton trial. LSQR admission is not an
                # exact identity on the caller's compatible displacement.
                assert kwargs['tangent'] is False
                assert not frame.f_locals['steps'] and not frame.f_locals['snapshots']
                np.testing.assert_array_equal(args[1],frame.f_locals['initial_u'])
                np.testing.assert_array_equal(frame.f_locals['q'],
                    frame.f_locals['initial_reduced_displacements'])
                base._assert_states_equal(args[2],before_states)
                accepted_origin.append((args[1].copy(),copy.deepcopy(args[2])))
        finally:
            del frame
        return real_assemble(*args, **kwargs)

    with monkeypatch.context() as context:
        context.setattr(ns,'_assemble_nonlinear_system',observe_origin)
        result=solve_static_nonlinear(model,load,control='displacement',
            displacement_control=DisplacementControl(node_id=probe,dof='uz',target_displacement=base.DC_TARGET),
            kinematics='corotational',num_steps=1,max_iterations=1,tolerance=base.DC_TOL,
            initial_displacements=u,initial_element_states=states,equilibrate_initial_state=False,
            record_increment_snapshots=True)
    assert len(accepted_origin)==1
    np.testing.assert_array_equal(u,before_u)
    base._assert_states_equal(states,before_states)
    assert result.status != 'completed'
    assert result.info['displacement_control_adaptation']['cutbacks'] > 0
    assert len(result.snapshots)==len(result.steps)
    if result.steps:
        assert [s.control_value for s in result.snapshots]==[s.control_value for s in result.steps]
        assert abs(base._probe_uz(dict(model=model,result=result,probe=probe))-result.steps[-1].control_value)<=1e-8
    else:
        assert result.snapshots==()
        assert abs(base._probe_uz(dict(model=model,result=result,probe=probe))+.1)<=1e-7
        np.testing.assert_allclose(result.displacements,accepted_origin[0][0],rtol=1e-14,atol=1e-15)
        np.testing.assert_allclose(result.displacements,before_u,rtol=1e-14,atol=1e-15)
        base._assert_states_allclose(dict(result.element_states),accepted_origin[0][1])
    case=dict(model=model,load=load,probe=probe,grid=grid,result=result,coupling='interp')
    checked=base._independent_equilibrium(case)
    assert checked['relative_residual']<=1e-7
    assert np.linalg.norm(checked['balance'])<=1e-6*max(np.linalg.norm(checked['applied_wrench']),1e-30)
    mapping=base._equilibrium_oracle(case)['mpc_map']
    linear_u=mapping.to_linear_displacement(result.displacements)
    assert base.constraint_residual_summary(model,linear_u)['status']=='passed'
    for snapshot in result.snapshots:
        checked=base._independent_equilibrium(case,displacements=snapshot.displacements.copy(),
            load_factor=snapshot.load_factor,element_states=copy.deepcopy(dict(snapshot.element_states)))
        assert checked['relative_residual']<=1e-7
