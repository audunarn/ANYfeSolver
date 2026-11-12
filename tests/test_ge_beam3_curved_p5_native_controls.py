"""Native P5 nonlinear control paths and cancellation state-safety checks."""

import numpy as np
import pytest

from anyfesolver.arc_length import ArcLengthControl, solve_static_arc_length
from anyfesolver.control import CancellationToken, SolveCancelled
from anyfesolver.nonlinear_static import (
    solve_static_nonlinear, DisplacementControl, NonlinearLoadStage, NonlinearLoadProgram,
    NonlinearConvergenceSettings,
)
from anyfesolver.boundary import LoadCase
from anyfesolver.nonlinear_restart import canonical_checkpoint_json_bytes
from docs.reference_cases.ge_beam3_curved_p5_native_material_probe import canonical
from test_ge_beam3_curved_p5_native_driver_probe import problem


def test_actual_displacement_control_reaches_curved_plastic_state():
    model, element, load=problem()
    result=solve_static_nonlinear(model, load, control='displacement',
        displacement_control=DisplacementControl(node_id=3, dof='ux', target_displacement=.03),
        num_steps=2, max_iterations=10, tolerance=1e-10, num_layers=1, min_step_fraction=1.,
        record_increment_snapshots=True, emit_restart_checkpoint=True)
    assert result.status=='completed', (result.status, result.failure_reason)
    assert abs(result.displacements[12]-.03)<=1e-11
    state=result.element_states[1]
    element.validate_model_bound_nonlinear_state(model.mesh, element.core.section, state, 1,
                                                expected_committed_total_u=result.displacements)
    assert any(h.accumulated>0 for h in state['material_state']['histories'])


def test_actual_arc_length_reaches_native_committed_state():
    model, element, load=problem()
    control=ArcLengthControl(initial_load_increment=.5, minimum_load_increment=.5,
        maximum_load_increment=.5, growth_factor=1., max_steps=2, max_retries_per_step=1,
        stop_after_peak_steps=20)
    result=solve_static_arc_length(model, load, control=control, max_iterations=10,
        tolerance=1e-10, arc_tolerance=1e-10, num_layers=1, record_increment_snapshots=True,
        emit_restart_checkpoint=True)
    assert result.status=='maximum_steps_reached', (result.status, result.info)
    assert result.load_factor>0
    state=result.element_states[1]
    element.validate_model_bound_nonlinear_state(model.mesh, element.core.section, state, 1,
                                                expected_committed_total_u=result.displacements)
    assert state['material_state']['epoch']>=2


@pytest.mark.parametrize('mode', ['force', 'displacement', 'arc_length'])
def test_cancellation_during_unaccepted_native_trial_discards_it(monkeypatch, mode):
    model, element, load=problem();cancellation=CancellationToken();captured=[]
    original=element.compute_nonlinear_response
    def cancel_after_response(*args, **kwargs):
        response=original(*args, **kwargs)
        context=kwargs['native_material_context'];view=kwargs['native_rotation_trial']
        if not captured and context.store.generation==1 and not np.array_equal(view.trial_coordinates, view.committed_coordinates):
            captured.append((context.store, canonical(context.store.materialize())))
            cancellation.cancel('cancel unaccepted P5 candidate')
        return response
    monkeypatch.setattr(element, 'compute_nonlinear_response', cancel_after_response)
    with pytest.raises(SolveCancelled):
        if mode=='arc_length':
            settings=ArcLengthControl(initial_load_increment=.5, minimum_load_increment=.5,
                maximum_load_increment=.5, growth_factor=1., max_steps=2, max_retries_per_step=1)
            solve_static_arc_length(model, load, control=settings, max_iterations=10,
                tolerance=1e-10, arc_tolerance=1e-10, num_layers=1, cancellation_token=cancellation)
        else:
            options={} if mode=='force' else dict(control='displacement',
                displacement_control=DisplacementControl(node_id=3, dof='ux', target_displacement=.03))
            solve_static_nonlinear(model, load, num_steps=2, max_iterations=10,
                tolerance=1e-10, num_layers=1, min_step_fraction=1., cancellation_token=cancellation, **options)
    assert captured
    store, before=captured[0]
    assert store.generation==store.native_rotation_store.generation==1
    assert canonical(store.materialize())==before
    assert not store.has_active_trial
    assert not store.native_rotation_store.has_active_trial


def test_native_loading_unloading_reversal_program_preserves_accepted_origins():
    model, element, load=problem()
    unload=LoadCase('unload');unload.add_nodal_load(3, forces=np.array([-.025, .01, -.005]))
    reverse=LoadCase('reverse');reverse.add_nodal_load(3, forces=np.array([-.025, .01, -.005]))
    program=NonlinearLoadProgram([NonlinearLoadStage('loading', load),
        NonlinearLoadStage('unloading', unload), NonlinearLoadStage('reversal', reverse)])
    result=solve_static_nonlinear(model, load_program=program, num_steps=6, max_iterations=10,
        tolerance=1e-10, num_layers=1, min_step_fraction=1., record_increment_snapshots=True,
        convergence_settings=NonlinearConvergenceSettings(profile='legacy', growth_factor=1.,
            max_step_factor=1., line_search='always', max_line_search_cuts=4))
    assert result.status=='completed', (result.status, result.failure_reason)
    assert result.load_factor==3.
    snapshots=result.snapshots
    for previous, current in zip(snapshots, snapshots[1:]):
        a=previous.element_states[1]['material_state'];b=current.element_states[1]['material_state']
        assert b['origins']==a['histories']
        assert all(new.accumulated>=old.accumulated for new, old in zip(b['histories'], a['histories']))
    element.recover_native_fields(model.mesh, result.element_states[1], expected_committed_total_u=result.displacements)


def test_native_displacement_checkpoint_split_matches_uninterrupted():
    def run(target, steps, checkpoint=None):
        model, _, load=problem()
        return solve_static_nonlinear(model, load, control='displacement',
            displacement_control=DisplacementControl(node_id=3, dof='ux', target_displacement=target),
            num_steps=steps, max_iterations=10, tolerance=1e-10, num_layers=1, min_step_fraction=1.,
            restart_checkpoint=checkpoint, emit_restart_checkpoint=True)
    continuous=run(.03, 2);first=run(.015, 1)
    assert continuous.status==first.status=='completed'
    resumed=run(.03, 1, canonical_checkpoint_json_bytes(first.restart_checkpoint))
    assert resumed.status=='completed'
    assert np.array_equal(resumed.displacements, continuous.displacements)
    assert canonical(resumed.element_states)==canonical(continuous.element_states)
    assert canonical_checkpoint_json_bytes(resumed.restart_checkpoint)==canonical_checkpoint_json_bytes(continuous.restart_checkpoint)


def test_native_arc_length_checkpoint_split_matches_uninterrupted():
    def run(steps, checkpoint=None):
        model, _, load=problem()
        settings=ArcLengthControl(initial_load_increment=.5, minimum_load_increment=.5,
            maximum_load_increment=.5, growth_factor=1., max_steps=steps, max_retries_per_step=1,
            stop_after_peak_steps=20)
        return solve_static_arc_length(model, load, control=settings, max_iterations=10,
            tolerance=1e-10, arc_tolerance=1e-10, num_layers=1,
            restart_checkpoint=checkpoint, emit_restart_checkpoint=True)
    continuous=run(4);first=run(2)
    assert continuous.status==first.status=='maximum_steps_reached'
    resumed=run(2, canonical_checkpoint_json_bytes(first.restart_checkpoint))
    assert resumed.status=='maximum_steps_reached'
    assert np.array_equal(resumed.displacements, continuous.displacements)
    assert canonical(resumed.element_states)==canonical(continuous.element_states)
    assert canonical_checkpoint_json_bytes(resumed.restart_checkpoint)==canonical_checkpoint_json_bytes(continuous.restart_checkpoint)
