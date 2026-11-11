"""Coordinate-state successor correctness; no production qualification claim."""

from copy import deepcopy
from fractions import Fraction

import numpy as np
import pytest

from anyfesolver.fe_core import FEModel
from anyfesolver.boundary import BoundaryCondition, LoadCase
from anyfesolver.ge_beam3_curved_reference import CurvedBeam3ReferenceGeometry
from anyfesolver.nonlinear_static import _assemble_nonlinear_system, solve_static_nonlinear
from anyfesolver.nonlinear_state import NonlinearStateStore, create_model_native_rotation_store
from anyfesolver.nonlinear_restart import canonical_checkpoint_json_bytes
from anyfesolver._ge_beam3_p5 import NativeP5BeamElement as V1
from anyfesolver._ge_beam3_p5_coordinates import NativeP5BeamElement, DirectedHardeningSection
from anyfesolver._ge_beam3_p5_coordinates import codec
from anyfesolver._ge_beam3_p5_coordinates.core import canonical, seal
from anyfesolver._ge_beam3_p5_coordinates.positions import from_total, validate_total_pair, POLICY, rest_mass
from anyfesolver._ge_beam3_p5_coordinates.committed_modal import prepare, solve_elastic_modes
from anyfesolver._ge_beam3_p5_coordinates.reference_modal import solve as reference_modes
from test_ge_beam3_curved_p5_algebra_probe import reference
from test_ge_beam3_curved_p5_nonlinear_mixed_probe import law
from test_ge_beam3_curved_p5_mass_probe import section_mass


def problem(*, curved=False, plastic=False, cls=NativeP5BeamElement):
    if curved:
        ref = reference(.4); material = law(.02 if plastic else 1000.)
        section = DirectedHardeningSection(material._elastic, material._direction, material._yield, material._hardening)
    else:
        points = np.array([[0.,0.,0.],[1.,0.,0.],[2.,0.,0.]])
        ref = CurvedBeam3ReferenceGeometry(points, np.tile(np.eye(3), (3,1,1)))
        section = DirectedHardeningSection(np.eye(6), np.array([1.,0.,0.,0.,0.,0.]), 1., 1.)
    model = FEModel('p5-coordinate-state')
    for index, point in enumerate(ref.coordinates, 1): model.add_node(index, *point)
    element = cls(1, (1,2,3), ref, section)
    model.add_element(1, element); model.materials[element.material_name] = element.core.section
    model.add_boundary_condition(BoundaryCondition('root', [1], {n:0. for n in ('ux','uy','uz','rx','ry','rz')}))
    return model, element


def make_store(model, element):
    initial = {1:element.init_model_bound_nonlinear_state(model.mesh, element.core.section, 1)}
    store = NonlinearStateStore.from_shell_layouts((), initial)
    store.attach_native_rotation_store(create_model_native_rotation_store(model, initial, np.zeros(18)))
    return store


def assemble(model, element, total, *, commit=True):
    store = make_store(model, element); before = canonical(store.materialize())
    try:
        force, matrix, payload = _assemble_nonlinear_system(model, total, store, 1)
        state = deepcopy(payload[1])
        if commit:
            store.commit(store.active_trial_token(), accepted_full_displacement=total,
                accepted_full_coordinates=element.core.reference.coordinates+total.reshape(3,6)[:, :3])
            assert canonical(store[1]) == canonical(state)
            assert store.generation == store.native_rotation_store.generation == 1
    finally:
        if store.has_active_trial: store.discard_trial(store.active_trial_token())
    if not commit:
        assert canonical(store.materialize()) == before
        assert store.generation == store.native_rotation_store.generation == 0
    return force, matrix, state


@pytest.mark.parametrize('strain', [2.**-20, 2.**-54, -2.**-55])
def test_actual_native_axial_commit_retains_force_work_and_replay(strain):
    model, element = problem(); total = np.zeros(18); total[[6,12]] = [strain, 2*strain]
    force, matrix, state = assemble(model, element, total)
    expected = np.zeros(18); expected[[0,12]] = [-strain,strain]
    assert np.linalg.norm(force-expected) <= 1e-11*np.linalg.norm(expected)
    assert np.linalg.norm(matrix@total-expected) <= 1e-11*np.linalg.norm(expected)
    inner = state['material_state']; response = inner['response']
    assert abs(response.potential-strain**2) <= 1e-11*strain**2
    assert inner['coordinate_policy'] == POLICY
    element.validate_model_bound_nonlinear_state(model.mesh, element.core.section, state, 1,
        expected_committed_total_u=total)
    for station in response.stations:
        assert abs(station.response.strain[0]-strain) <= 1e-11*abs(strain)
        assert abs(station.response.resultants[0]-strain) <= 1e-11*abs(strain)
    encoded = element.serialize_native_material_state(model.mesh, state)
    restored = element.validate_model_bound_nonlinear_state(model.mesh, element.core.section, encoded, 1,
        expected_committed_total_u=total)
    assert canonical(restored) == canonical(state)
    assert encoded['schema'] == codec.SCHEMA


def test_discard_preserves_virgin_state_even_when_high_pose_does_not_move():
    model, element = problem(); total = np.zeros(18); total[[6,12]] = [2.**-54, 2.**-53]
    _, _, state = assemble(model, element, total, commit=False)
    assert np.count_nonzero(state['material_state']['committed_position_low']) == 2


def test_recovery_retains_sub_ulp_station_motion_without_advancing_history():
    model, element = problem(); strain = 2.**-54
    total = np.zeros(18); total[[6,12]] = [strain,2*strain]
    virgin = element.init_model_bound_nonlinear_state(model.mesh, element.core.section, 1)
    _, _, state = assemble(model, element, total)
    before = canonical(state)
    origin = element.recover_native_fields(model.mesh, virgin)
    made = element.recover_native_fields(model.mesh, state)
    assert canonical(state) == before and made['coordinate_policy'] == POLICY
    assert made['current_positions'].shape == made['current_position_low'].shape == (16,3)
    points, _ = np.polynomial.legendre.leggauss(8)
    for index, (cell, station) in enumerate(made['station_ids']):
        t = float((points[station]+1)/2)
        actual = sum((Fraction(float(made[key][index,0]))-Fraction(float(origin[key][index,0]))
                      for key in ('current_positions','current_position_low')), Fraction())
        expected = Fraction(strain)*(Fraction(float(1-t))*cell+Fraction(t)*(cell+1))
        assert abs(actual-expected) <= abs(expected)*Fraction(1, 2**48)
    assert not made['current_position_low'].flags.writeable


@pytest.mark.parametrize('mutation', ['low','high','total','policy','missing','response'])
def test_resealed_coordinate_mutations_fail_before_commit(mutation):
    from dataclasses import replace
    model, element = problem(); total = np.zeros(18); total[[6,12]] = [2.**-54,2.**-53]
    store = make_store(model, element); before = canonical(store.materialize())
    try:
        _, _, payload = _assemble_nonlinear_system(model, total, store, 1)
        state = deepcopy(payload[1]); inner = state['material_state']
        if mutation == 'low': inner['committed_position_low'][:] = 0.
        if mutation == 'high': inner['committed_positions'][1,0] = np.nextafter(1.,2.)
        if mutation == 'total': inner['committed_total_u'][6] *= 2.
        if mutation == 'policy': inner['coordinate_policy'] = 'UNBOUND'
        if mutation == 'missing': del inner['committed_position_low']
        if mutation == 'response': inner['response'] = replace(inner['response'], potential=0.)
        state['material_state'] = seal(inner); state = seal(state)
        with pytest.raises(ValueError): store.set_trial_state(store.active_trial_token(), 1, state)
        assert store.generation == 0 and canonical(store.materialize()) == before
    finally:
        if store.has_active_trial: store.discard_trial(store.active_trial_token())


def test_v1_and_v2_histories_are_not_hot_restart_compatible():
    old, old_element = problem(cls=V1); new, new_element = problem()
    for source, se, destination, de in ((old,old_element,new,new_element),(new,new_element,old,old_element)):
        state = se.init_model_bound_nonlinear_state(source.mesh, se.core.section, 1)
        encoded = se.serialize_native_material_state(source.mesh, state)
        for data in (state, encoded):
            with pytest.raises(ValueError):
                de.validate_model_bound_nonlinear_state(destination.mesh, de.core.section, data, 1)


def test_zero_low_native_response_matches_frozen_v1_exactly():
    responses = []
    for cls in (V1, NativeP5BeamElement):
        model, element = problem(cls=cls); total = np.zeros(18); total[[6,12]] = [2.**-20, 2.**-19]
        force, matrix, state = assemble(model, element, total)
        responses.append((canonical(force), canonical(matrix.toarray()), canonical(state['material_state']['response'])))
    assert responses[0] == responses[1]


def run(model, *, num_steps=2, max_load_factor=1., restart_checkpoint=None):
    load = LoadCase('tip'); load.add_nodal_load(3, forces=np.array([.025,-.01,.005]))
    result = solve_static_nonlinear(model, load, num_steps=num_steps, max_iterations=12, tolerance=1e-12,
        num_layers=1, min_step_fraction=1., emit_restart_checkpoint=True, max_load_factor=max_load_factor,
        restart_checkpoint=restart_checkpoint)
    assert result.status == 'completed', (result.status, result.info)
    return result


@pytest.fixture(scope='module', params=[False, True])
def curved_path(request):
    model, element = problem(curved=True, plastic=request.param)
    return model, element, run(model), request.param


def test_curved_native_history_and_split_restart_are_exact(curved_path):
    model, element, continuous, plastic = curved_path
    first_model, _ = problem(curved=True, plastic=plastic)
    first = run(first_model, num_steps=1, max_load_factor=.5)
    fresh, fresh_element = problem(curved=True, plastic=plastic)
    continued = run(fresh, restart_checkpoint=canonical_checkpoint_json_bytes(first.restart_checkpoint))
    assert canonical_checkpoint_json_bytes(continued.restart_checkpoint) == canonical_checkpoint_json_bytes(continuous.restart_checkpoint)
    assert canonical(continued.element_states) == canonical(continuous.element_states)
    assert canonical(fresh_element.recover_native_fields(fresh.mesh, continued.element_states[1])) == canonical(
        element.recover_native_fields(model.mesh, continuous.element_states[1]))
    inner = continuous.element_states[1]['material_state']
    assert np.count_nonzero(inner['committed_position_low']) > 0
    assert any(h.accumulated > 0 for h in inner['histories']) == plastic


def test_current_operator_uses_retained_coordinate_state(curved_path):
    model, element, result, plastic = curved_path
    before = canonical(result.element_states)
    packet, guard = prepare(model, result.element_states, result.displacements, {1:section_mass()})
    inner = result.element_states[1]['material_state']
    assert np.linalg.norm(packet.internal_force[:18]-inner['response'].residual) <= 1e-11
    forces = np.zeros(18); forces[12:15] = [.025,-.01,.005]
    if plastic:
        with pytest.raises(ValueError, match='vibration branch'):
            solve_elastic_modes(model, result.element_states, result.displacements, {1:section_mass()}, forces)
    else:
        _, modes = solve_elastic_modes(model, result.element_states, result.displacements, {1:section_mass()}, forces)
        assert np.all(modes.eigenvalues > 0.) and modes.normalized_residual <= 1e-11
    guard(); assert canonical(result.element_states) == before


def test_current_operator_does_not_lose_tiny_axial_preload():
    model, element = problem(); total = np.zeros(18); strain = 2.**-54; total[[6,12]] = [strain,2*strain]
    force, _, state = assemble(model, element, total)
    packet, guard = prepare(model, {1:state}, total, {1:section_mass()})
    assert np.linalg.norm(packet.internal_force[:18]-force) <= 1e-11*np.linalg.norm(force)
    guard()


def test_reference_modes_and_rest_mass_are_unchanged_by_translation_pairs():
    model, element = problem()
    expected = reference_modes(model, {1:section_mass()})
    state = element.init_model_bound_nonlinear_state(model.mesh, element.core.section, 1)
    _, actual = solve_elastic_modes(model, {1:state}, np.zeros(18), {1:section_mass()}, np.zeros(18))
    assert np.linalg.norm(actual.eigenvalues-expected.eigenvalues) <= 1e-11*np.linalg.norm(expected.eigenvalues)
    ref = element.core.reference; rotations = np.tile(np.eye(3), (2,1,1))
    origin = rest_mass(ref, section_mass(), 8, ref.coordinates, np.zeros((3,3)), rotations)
    total = np.zeros(18); total.reshape(3,6)[:, :3] = 2.**-54
    high, low = from_total(ref.coordinates, total, ref.coordinates+total.reshape(3,6)[:, :3])
    changed = rest_mass(ref, section_mass(), 8, high, low, rotations)
    assert np.array_equal(origin, changed)


def test_coordinate_pairs_preserve_exact_binary64_input_sums():
    for a, b in ((1.,2.**-54), (1.,-2.**-55), (2.**-600,2.**-650), (1.,-1.),
                 (np.nextafter(0.,1.),np.nextafter(0.,1.)), (2.**400,-2.**400)):
        reference_nodes = np.full((3,3), a); total = np.zeros(18); total.reshape(3,6)[:, :3] = b
        high, low = from_total(reference_nodes, total, reference_nodes+b)
        assert Fraction(float(high[0,0]))+Fraction(float(low[0,0])) == Fraction(float(a))+Fraction(float(b))
        validate_total_pair(reference_nodes, total, high, low, POLICY)


@pytest.fixture
def successor_controls(monkeypatch):
    import test_ge_beam3_curved_p5_native_controls as controls
    def make_problem():
        model, element = problem(curved=True, plastic=True)
        load = LoadCase('tip'); load.add_nodal_load(3, forces=np.array([.025,-.01,.005]))
        return model, element, load
    monkeypatch.setattr(controls, 'problem', make_problem)
    return controls


@pytest.mark.parametrize('name', [
    'test_actual_displacement_control_reaches_curved_plastic_state',
    'test_actual_arc_length_reaches_native_committed_state',
    'test_native_loading_unloading_reversal_program_preserves_accepted_origins',
    'test_native_displacement_checkpoint_split_matches_uninterrupted',
    'test_native_arc_length_checkpoint_split_matches_uninterrupted'])
def test_existing_native_control_contracts_on_coordinate_successor(successor_controls, name):
    getattr(successor_controls, name)()


@pytest.mark.parametrize('mode', ['force','displacement','arc_length'])
def test_cancellation_cleans_successor_trial_without_losing_accepted_lows(successor_controls, monkeypatch, mode):
    successor_controls.test_cancellation_during_unaccepted_native_trial_discards_it(monkeypatch, mode)


@pytest.mark.parametrize('mutation', ['duplicate','nonfinite','extra_key','wrong_shape','wrong_number_type'])
def test_typed_coordinate_codec_rejects_malformed_payload(mutation):
    import json
    model, element = problem()
    state = element.init_model_bound_nonlinear_state(model.mesh, element.core.section, 1)
    envelope = element.serialize_native_material_state(model.mesh, state)
    raw = json.loads(envelope['payload'])
    if mutation == 'extra_key': raw['material_state']['other'] = None
    if mutation == 'wrong_shape': raw['material_state']['committed_position_low'] = [0.,0.,0.]
    if mutation == 'wrong_number_type': raw['material_state']['committed_position_low'][0][0] = 0
    envelope['payload'] = canonical(raw).decode('ascii')
    if mutation == 'duplicate': envelope['payload'] = envelope['payload'].replace('{','{"driver_schema":"duplicate",',1)
    if mutation == 'nonfinite': envelope['payload'] = envelope['payload'].replace('0.0','NaN',1)
    with pytest.raises(ValueError): codec.decode(envelope, order=8)


@pytest.mark.parametrize('curved', [False, True])
def test_common_finite_rigid_motion_remains_strain_free(curved):
    from docs.reference_cases.ge_beam3_curved_p5_algebra_probe import rotation
    model, element = problem(curved=curved)
    vector = np.array([.7,-.4,.8]); proper = rotation(vector)
    total = np.zeros((3,6)); points = element.core.reference.coordinates
    total[:, :3] = points@proper.T + np.array([2.,-1.,.5])-points
    total[:, 3:] = vector
    force, _, state = assemble(model, element, total.ravel())
    assert np.linalg.norm(force) <= 1e-11
    assert max(np.linalg.norm(s.response.strain) for s in state['material_state']['response'].stations) <= 1e-11
    element.recover_native_fields(model.mesh, state, expected_committed_total_u=total.ravel())


def test_two_element_native_coordinates_and_global_trace_modes():
    from docs.reference_cases.ge_beam3_curved_p5_continuum_probe import parabolic_references
    refs = parabolic_references(.4,2); ids = (7,23,55,81,103)
    model = FEModel('coordinate-successor-two-element-arch')
    for index, point in zip(ids, np.vstack((refs[0].coordinates,refs[1].coordinates[1:]))):
        model.add_node(index,*point)
    for index, ref in enumerate(refs):
        material = law(1000.)
        section = DirectedHardeningSection(material._elastic,material._direction,material._yield,material._hardening)
        element = NativeP5BeamElement(index+1,ids[2*index:2*index+3],ref,section)
        model.add_element(index+1,element); model.materials[element.material_name] = element.core.section
    model.add_boundary_condition(BoundaryCondition('root',[7], {n:0. for n in ('ux','uy','uz','rx','ry','rz')}))
    load = LoadCase('tip'); load.add_nodal_load(103,forces=np.array([.005,-.002,.001]))
    result = solve_static_nonlinear(model,load,num_steps=2,max_iterations=12,tolerance=1e-12,
        num_layers=1,min_step_fraction=1.,emit_restart_checkpoint=True)
    assert result.status == 'completed', (result.status,result.info)
    first = result.element_states[1]['material_state']; second = result.element_states[2]['material_state']
    for key in ('committed_positions','committed_position_low','committed_nodal_rotation_matrices'):
        assert np.array_equal(first[key][-1], second[key][0])
    force = np.zeros(30); force[24:27] = [.005,-.002,.001]
    before = canonical(result.element_states)
    packet, modes = solve_elastic_modes(model,result.element_states,result.displacements,
        {1:section_mass(),2:section_mass()},force)
    assert packet.stiffness.shape == (42,42) and modes.dynamic_map.shape == (42,24)
    assert modes.normalized_residual <= 1e-11
    assert canonical(result.element_states) == before


def test_successor_source_map_binds_all_modules_and_preserves_v1():
    import ast
    from docs.reference_cases import ge_beam3_curved_p5_coordinate_source_map as mapping
    record = mapping.build()
    assert mapping.source(mapping.MANIFEST) == mapping.canonical(record)
    assert len(record['outputs']) == 24
    assert record['independent_review_status'] == 'PENDING' and not record['production_qualified']
    assert len(record['derived_from']) == 5
    for name in mapping.FILES:
        tree = ast.parse(mapping.source(mapping.TARGET+'/'+name))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom): assert not (node.module or '').startswith(('docs','tests'))
    import anyfesolver
    assert not hasattr(anyfesolver,'NativeP5BeamElement')


def test_source_map_rejects_changed_preserved_mechanics(monkeypatch):
    from docs.reference_cases import ge_beam3_curved_p5_coordinate_source_map as mapping
    read = mapping.source
    def changed(path):
        raw = read(path)
        return raw+b'\n' if path == 'src/anyfesolver/_ge_beam3_p5/compensated.py' else raw
    monkeypatch.setattr(mapping,'source',changed)
    with pytest.raises(ValueError, match='preserved package mechanics'): mapping.build()


@pytest.mark.parametrize('plastic', [False,True])
def test_native_coordinate_tangent_remains_the_force_directional_derivative(plastic):
    from test_ge_beam3_curved_p5_native_chart_probe import sample
    model, element = problem(curved=True,plastic=plastic)
    total = sample(); direction = np.sin(np.arange(18)+.3); step = 1e-6
    _, matrix, _ = assemble(model,element,total,commit=False)
    plus, _, _ = assemble(model,element,total+step*direction,commit=False)
    minus, _, _ = assemble(model,element,total-step*direction,commit=False)
    expected = matrix@direction
    assert np.linalg.norm((plus-minus)/(2*step)-expected) <= 1e-7*max(1.,np.linalg.norm(expected))
    dense = matrix.toarray()
    assert np.linalg.norm(dense-dense.T) <= 1e-11*max(1.,np.linalg.norm(dense))
