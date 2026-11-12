"""Actual public Newton driver with private curved P5 material mechanics."""

from copy import deepcopy
from dataclasses import replace

import numpy as np
import pytest

from anyfesolver.fe_core import FEModel
from anyfesolver.boundary import BoundaryCondition, LoadCase
from anyfesolver.nonlinear_static import solve_static_nonlinear, _assemble_nonlinear_system, NonlinearConvergenceSettings
from anyfesolver.nonlinear_state import NonlinearStateStore, StateTransactionError, create_model_native_rotation_store
from anyfesolver._native_rotation_state import NativeRotationValidationError
from anyfesolver._native_material_protocol import NativeMaterialValidator, NativeMaterialContext
from anyfesolver.ge_beam3_curved_reference import CurvedBeam3ReferenceGeometry
from docs.reference_cases.ge_beam3_curved_p5_native_driver_probe import DriverP5Element
from docs.reference_cases.ge_beam3_curved_p5_native_material_probe import NativeMaterialError, canonical, seal
from docs.reference_cases.ge_beam3_curved_p5_algebra_probe import reference_hessians
from test_ge_beam3_curved_p5_algebra_probe import reference, assert_scaled_close
from test_ge_beam3_curved_p5_nonlinear_mixed_probe import law
from test_ge_beam3_curved_p5_native_chart_probe import sample, store_for


def problem(height=.4):
    model=FEModel('private-p5-actual-newton');ref=reference(height)
    for i, point in enumerate(ref.coordinates, 1): model.add_node(i, *point)
    element=DriverP5Element(1, (1, 2, 3), ref, law())
    model.add_element(1, element)
    model.materials[element.material_name]=element.core.section
    model.add_boundary_condition(BoundaryCondition('root', [1],
        {name: 0. for name in ('ux', 'uy', 'uz', 'rx', 'ry', 'rz')}))
    load=LoadCase('tip');load.add_nodal_load(3, forces=np.array([.025, -.01, .005]))
    return model, element, load


@pytest.mark.parametrize('height', [0., .4])
def test_actual_newton_driver_reaches_p5_material_state(height):
    model, element, load=problem(height)
    result=solve_static_nonlinear(model, load, num_steps=2, max_iterations=10,
        tolerance=1e-10, num_layers=1, min_step_fraction=1., record_increment_snapshots=True)
    assert result.status=='completed', (result.status, result.info)
    state=result.element_states[1]
    element.validate_model_bound_nonlinear_state(model.mesh, element.core.section, state, 1,
                                                expected_committed_total_u=result.displacements)
    assert state['material_state']['epoch']>=2
    assert any(h.accumulated>0 for h in state['material_state']['histories'])
    assert len(result.snapshots)==2
    prior=result.snapshots[0].element_states[1]['material_state']
    following=result.snapshots[1].element_states[1]['material_state']
    assert following['origins']==prior['histories']
    assert canonical(state)==canonical(result.snapshots[-1].element_states[1])


def make_store(model, element):
    states={1: element.init_model_bound_nonlinear_state(model.mesh, element.core.section, 1)}
    store=NonlinearStateStore.from_shell_layouts((), states)
    store.attach_native_rotation_store(create_model_native_rotation_store(model, states, np.zeros(18)))
    return store


def test_reference_stiffness_is_native_stationary_hessian():
    model, element, _=problem()
    made=element.compute_stiffness_matrix(model.mesh, element.core.section)
    full=reference_hessians(element.core.reference, element.core.section._elastic, 8)[0]
    expected=full[:18, :18]-full[:18, 18:]@np.linalg.solve(full[18:, 18:], full[18:, :18])
    assert_scaled_close(made, expected)
    eigen=np.linalg.eigvalsh(made);scale=np.linalg.norm(made)
    assert np.count_nonzero(np.abs(eigen)<1e-11*scale)==6
    assert eigen[6]>1e-11*scale


def test_store_itself_rejects_corrupted_candidate_before_any_commit():
    model, element, _=problem();store=make_store(model, element)
    before=canonical(store.materialize());_, _, payload=_assemble_nonlinear_system(model, sample(), store, 1)
    token=store.active_trial_token();candidate=deepcopy(payload[1])
    inner=candidate['material_state'];inner['response']=replace(inner['response'], potential=inner['response'].potential+.01)
    candidate['material_state']=seal(inner);candidate=seal(candidate)
    with pytest.raises(NativeMaterialError, match='replay'): store.set_trial_state(token, 1, candidate)
    # Also exercise commit-time protection against a corrupted internal candidate.
    store._fallback_trial[1]=candidate
    with pytest.raises(NativeMaterialError, match='replay'):
        store.commit(token, accepted_full_displacement=sample(),
                     accepted_full_coordinates=element.core.reference.coordinates+sample().reshape(3, 6)[:, :3])
    assert store.generation==store.native_rotation_store.generation==0
    assert canonical(store.materialize())==before
    store.discard_trial(token)


def test_native_protocol_requires_every_candidate_and_rejects_deletion():
    model, element, _=problem();store=make_store(model, element)
    token=store.begin_trial(full_displacement=np.zeros(18), full_coordinates=element.core.reference.coordinates)
    with pytest.raises(StateTransactionError, match='no candidate'):
        store.commit(token, accepted_full_displacement=np.zeros(18), accepted_full_coordinates=element.core.reference.coordinates)
    store.discard_trial(token)
    _assemble_nonlinear_system(model, sample(), store, 1);token=store.active_trial_token()
    store.freeze_deleted([1])
    with pytest.raises(StateTransactionError, match='deletion'):
        store.commit(token, accepted_full_displacement=sample(),
                     accepted_full_coordinates=element.core.reference.coordinates+sample().reshape(3, 6)[:, :3])
    assert store.generation==0;store.discard_trial(token)


def test_invalid_initial_material_state_rejected_before_store_attachment():
    model, element, _=problem();state=element.init_model_bound_nonlinear_state(model.mesh, element.core.section, 1)
    state['material_state']['response']=replace(state['material_state']['response'], potential=1.)
    state['material_state']=seal(state['material_state']);state=seal(state)
    with pytest.raises(NativeMaterialError, match='replay'):
        create_model_native_rotation_store(model, {1: state}, np.zeros(18))


@pytest.mark.parametrize('mutation', ['protocol', 'validator'])
def test_invalid_protocol_factory_fails_closed(mutation, monkeypatch):
    model, element, _=problem()
    if mutation=='protocol': monkeypatch.setattr(element, 'native_material_state_protocol', 'unknown')
    else: monkeypatch.setattr(element, 'create_native_material_validator', lambda mesh: object())
    with pytest.raises(NativeRotationValidationError): make_store(model, element)


def test_validator_receives_owned_state_not_store_history(monkeypatch):
    model, element, _=problem();original=element.create_native_material_validator
    def factory(mesh):
        base=original(mesh)
        def validate(state, **kwargs):
            base.validate(state, **kwargs)
            state['committed_total_u'][:]=123.
            if kwargs['previous_state'] is not None: kwargs['previous_state']['committed_total_u'][:]=456.
        return NativeMaterialValidator(validate)
    monkeypatch.setattr(element, 'create_native_material_validator', factory)
    store=make_store(model, element)
    _assemble_nonlinear_system(model, sample(), store, 1);token=store.active_trial_token()
    assert np.array_equal(store[1]['committed_total_u'], np.zeros(18))
    store.commit(token, accepted_full_displacement=sample(),
                 accepted_full_coordinates=element.core.reference.coordinates+sample().reshape(3, 6)[:, :3])
    assert np.array_equal(store[1]['committed_total_u'], sample())


def test_actual_driver_failed_increment_preserves_virgin_material_state():
    model, element, load=problem()
    result=solve_static_nonlinear(model, load, num_steps=1, max_iterations=1,
        tolerance=1e-12, num_layers=1, min_step_fraction=1.,
        convergence_settings=NonlinearConvergenceSettings(profile='legacy', line_search='never'))
    assert result.status!='completed'
    state=result.element_states[1]['material_state']
    assert state['epoch']==0
    assert not any(history.accumulated for history in state['histories'])
    assert np.array_equal(state['committed_total_u'], np.zeros(18))


def test_actual_driver_with_explicit_line_search_passes_fixed_origin_replay():
    model, element, load=problem()
    result=solve_static_nonlinear(model, load, num_steps=2, max_iterations=10,
        tolerance=1e-10, num_layers=1, min_step_fraction=1.,
        convergence_settings=NonlinearConvergenceSettings(profile='legacy', line_search='always', max_line_search_cuts=4))
    assert result.status=='completed'
    state=result.element_states[1]
    element.validate_model_bound_nonlinear_state(model.mesh, element.core.section, state, 1,
                                                expected_committed_total_u=result.displacements)


def test_unbound_store_cannot_issue_material_context():
    ref=reference();store=NonlinearStateStore.from_shell_layouts((), {})
    store.attach_native_rotation_store(store_for(ref))
    token=store.begin_trial(full_displacement=np.zeros(18), full_coordinates=ref.coordinates)
    view=store.native_element_rotation_view(token, 1, (0, 1, 2), ref.nodal_triads[:, :, 2])
    forged=NativeMaterialContext(store, token, 1, (0, 1, 2), ref.nodal_triads[:, :, 2])
    with pytest.raises(StateTransactionError, match='no native material'): forged.require_view(view)
    store.discard_trial(token)


def test_late_native_validator_failure_precedes_all_store_pointer_swaps(monkeypatch):
    model, first, _=problem();base=reference();ref=CurvedBeam3ReferenceGeometry(base.coordinates+[2., 0., 0.], base.nodal_triads)
    model.add_node(4, *ref.coordinates[1]);model.add_node(5, *ref.coordinates[2])
    second=DriverP5Element(2, (3, 4, 5), ref, law());model.add_element(2, second)
    model.materials[second.material_name]=second.core.section
    fail=[False];original=second.create_native_material_validator
    def factory(mesh):
        base_validator=original(mesh)
        def validate(state, **kwargs):
            if fail[0]: raise RuntimeError('late validation before commit')
            base_validator.validate(state, **kwargs)
        return NativeMaterialValidator(validate)
    monkeypatch.setattr(second, 'create_native_material_validator', factory)
    states={e.element_id: e.init_model_bound_nonlinear_state(model.mesh, e.core.section, 1) for e in (first, second)}
    store=NonlinearStateStore.from_shell_layouts((), states)
    store.attach_native_rotation_store(create_model_native_rotation_store(model, states, np.zeros(30)))
    before=canonical(store.materialize());u=np.sin(np.arange(30))*.01
    _assemble_nonlinear_system(model, u, store, 1);token=store.active_trial_token();fail[0]=True
    coords=np.array([node.coords()+u[np.array(node.dofs[:3])] for node in model.mesh.nodes.values()])
    with pytest.raises(RuntimeError, match='late validation'):
        store.commit(token, accepted_full_displacement=u, accepted_full_coordinates=coords)
    assert canonical(store.materialize())==before
    assert store.generation==store.native_rotation_store.generation==0
    store.discard_trial(token)
