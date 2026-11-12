"""Centered-reference adoption through real native solvers; development only."""

from copy import deepcopy

import numpy as np
import pytest

from anyfesolver.fe_core import FEModel
from anyfesolver.boundary import BoundaryCondition, LoadCase
from anyfesolver.nonlinear_static import solve_static_nonlinear
from anyfesolver.nonlinear_restart import canonical_checkpoint_json_bytes
from anyfesolver._ge_beam3_centered_reference import CenteredCurvedBeam3ReferenceGeometry as Centered
from anyfesolver._ge_beam3_p5_centered import NativeP5BeamElement, DirectedHardeningSection
from anyfesolver._ge_beam3_p5_centered.core import canonical, seal
from anyfesolver._ge_beam3_p5_centered.committed_modal import prepare, solve_elastic_modes
from anyfesolver._ge_beam3_p5_centered.reference_modal import solve as reference_modes
from anyfesolver._ge_beam3_p5_centered.mass import reference_kinetic_factors, current_rest_mass
from test_ge_beam3_curved_p5_nonlinear_mixed_probe import law
from test_ge_beam3_curved_p5_mass_probe import section_mass


def problem(*, shift=0., plastic=False, count=1):
    model = FEModel('centered-native-beam'); ids = (7,23,55) if count == 1 else (7,23,55,81,103)
    parameters = np.linspace(-1.,1.,2*count+1)
    nodes = np.array([[x,.5*(1-x*x),.25*(1-x*x)] for x in parameters])+shift
    frames = []
    for x in parameters:
        first = np.array([1.,-x,-.5*x]); first /= np.linalg.norm(first)
        second = np.array([0.,0.,1.]); second -= first*float(first@second); second /= np.linalg.norm(second)
        frames.append(np.column_stack((first,second,np.cross(first,second))))
    for i, point in zip(ids,nodes): model.add_node(i,*point)
    for index in range(count):
        ref = Centered(nodes[2*index:2*index+3],np.array(frames[2*index:2*index+3]))
        material = law(.02 if plastic else 1000.)
        section = DirectedHardeningSection(material._elastic,material._direction,material._yield,material._hardening)
        element = NativeP5BeamElement(index+1,ids[2*index:2*index+3],ref,section)
        model.add_element(index+1,element); model.materials[element.material_name] = element.core.section
    model.add_boundary_condition(BoundaryCondition('root',[ids[0]],{n:0. for n in ('ux','uy','uz','rx','ry','rz')}))
    load = LoadCase('tip'); load.add_nodal_load(ids[-1],forces=np.array([.025,-.01,.005]))
    return model,load


def run(model,load,**kwargs):
    result = solve_static_nonlinear(model,load,num_steps=kwargs.pop('num_steps',2),max_iterations=12,
        tolerance=1e-12,num_layers=1,min_step_fraction=1.,emit_restart_checkpoint=True,**kwargs)
    assert result.status == 'completed', (result.status,result.info)
    return result


@pytest.fixture(scope='module',params=[(False,1),(True,1),(False,2),(True,2)])
def translated_paths(request):
    plastic,count = request.param
    results = []
    for shift in (0.,2.**30,2.**40):
        model,load = problem(shift=shift,plastic=plastic,count=count)
        result = run(model,load)
        assert any(h.accumulated > 0. for state in result.element_states.values()
                   for h in state['material_state']['histories']) == plastic
        results.append((model,result))
    return results,plastic,count


def test_native_newton_and_station_response_are_translation_invariant(translated_paths):
    results,_,_ = translated_paths; baseline = results[0][1]
    for model,result in results:
        assert np.array_equal(result.displacements,baseline.displacements)
        for eid,element in model.mesh.elements.items():
            state = result.element_states[eid]
            assert type(element.core.reference) is Centered
            assert element.to_dict()['reference_evaluation'] == element.core.reference.evaluation_id
            assert canonical(state['material_state']['response']) == canonical(baseline.element_states[eid]['material_state']['response'])
            element.validate_model_bound_nonlinear_state(model.mesh,element.core.section,state,1,
                expected_committed_total_u=result.displacements[element.get_dof_mapping(model.mesh)])


def test_native_reference_and_current_operators_use_same_centered_geometry(translated_paths):
    results,plastic,count = translated_paths; compared = []
    for model,result in results:
        inertias = {eid:section_mass() for eid in model.mesh.elements}
        origin = reference_modes(model,inertias)
        before = canonical(result.element_states)
        packet,guard = prepare(model,result.element_states,result.displacements,inertias)
        guard(); assert canonical(result.element_states) == before
        if not plastic:
            force = np.zeros(len(result.displacements)); force[-6:-3] = [.025,-.01,.005]
            _,modes = solve_elastic_modes(model,result.element_states,result.displacements,inertias,force)
            assert modes.normalized_residual <= 1e-11 and np.all(modes.eigenvalues > 0.)
        compared.append((canonical(origin.full_stiffness),canonical(origin.full_mass),
            canonical(origin.eigenvalues),canonical(packet.stiffness),canonical(packet.mass),canonical(packet.internal_force)))
    assert all(values == compared[0] for values in compared)


def test_translated_native_recovery_keeps_physical_fields_and_split_positions(translated_paths):
    from fractions import Fraction as F
    results,_,_ = translated_paths; outputs = []
    for model,result in results:
        outputs.append({eid:element.recover_native_fields(model.mesh,result.element_states[eid])
                        for eid,element in model.mesh.elements.items()})
    for shift,made in zip((0.,2.**30,2.**40),outputs):
        for eid,fields in made.items():
            baseline = outputs[0][eid]
            for key in ('strains','resultants','global_resultants','reference_frames','current_frames'):
                assert np.array_equal(fields[key],baseline[key]), key
            assert fields['reference_evaluation'] == 'CENTERED_Q2_TWO_COMPONENT_COEFFICIENT_ANALYTIC_LIFT_V1'
            for index in np.ndindex(fields['current_positions'].shape):
                actual = sum((F(float(fields[k][index]))-F(float(baseline[k][index]))
                    for k in ('current_positions','current_position_low')),F())
                assert abs(actual-F(shift)) <= F(1,10**11)


def test_translated_plastic_restart_continues_exactly():
    model,load = problem(shift=2.**40,plastic=True); continuous = run(model,load)
    first_model,first_load = problem(shift=2.**40,plastic=True)
    first = run(first_model,first_load,num_steps=1,max_load_factor=.5)
    fresh,fresh_load = problem(shift=2.**40,plastic=True)
    continued = run(fresh,fresh_load,restart_checkpoint=canonical_checkpoint_json_bytes(first.restart_checkpoint))
    assert canonical_checkpoint_json_bytes(continued.restart_checkpoint) == canonical_checkpoint_json_bytes(continuous.restart_checkpoint)


def test_reference_evaluator_fingerprint_and_no_legacy_clone_are_enforced():
    from anyfesolver.ge_beam3_curved_reference import CurvedBeam3ReferenceGeometry as Old
    model,_ = problem(); element = model.mesh.elements[1]
    old = Old(element.core.reference.coordinates,element.core.reference.nodal_triads)
    with pytest.raises(ValueError,match='centered reference'):
        reference_kinetic_factors(old,element.core.section._elastic,section_mass(),order=8)
    element.core.reference = old
    with pytest.raises(ValueError,match='centered reference'):
        element.init_model_bound_nonlinear_state(model.mesh,element.core.section,1)


def test_v2_checkpoint_is_not_accepted_as_centered_v3():
    from anyfesolver._ge_beam3_p5_coordinates import NativeP5BeamElement as V2
    model,_ = problem(); element = model.mesh.elements[1]
    prior = V2(1,element.node_ids,element.core.reference,element.core.section)
    old_model = FEModel('v2-not-v3')
    for i,node in model.mesh.nodes.items(): old_model.add_node(i,*node.coords())
    old_model.add_element(1,prior); old_model.materials[prior.material_name] = prior.core.section
    state = prior.init_model_bound_nonlinear_state(old_model.mesh,prior.core.section,1)
    for value in (state,prior.serialize_native_material_state(old_model.mesh,state)):
        with pytest.raises(ValueError): element.validate_model_bound_nonlinear_state(model.mesh,element.core.section,value,1)


def test_centered_mass_retains_cell_rotations_without_guyan_or_trace_mass():
    model,_ = problem(shift=2.**40); element = model.mesh.elements[1]; ref = element.core.reference
    factors = reference_kinetic_factors(ref,element.core.section._elastic,section_mass(),order=8)
    assert factors.full.shape[1] == 24 and factors.uncondensed_stiffness_factor.shape == (18,24)
    assert not hasattr(factors,'static_map')
    traces = [3,4,5,9,10,11,15,16,17]
    assert not np.any(factors.full[:,traces])
    mass = current_rest_mass(ref,section_mass(),8,ref.coordinates,np.zeros((3,3)),np.tile(np.eye(3),(2,1,1)))
    assert np.linalg.norm(mass-factors.full.T@factors.full) <= 1e-11
    assert np.linalg.matrix_rank(mass,tol=1e-11*np.linalg.norm(mass)) == 15


def test_affine_split_recovery_does_not_amplify_weight_rounding():
    from fractions import Fraction as F
    from anyfesolver._ge_beam3_p5_centered.positions import station_position
    from anyfesolver._ge_beam3_p5_coordinates.positions import station_position as v2_station
    points = np.array([[0.,0.,0.],[1.,0.,0.],[2.,0.,0.]])
    t = float((np.polynomial.legendre.leggauss(8)[0][3]+1)/2)
    assert F(1.-t)+F(t) != 1  # The diagnostic requires a genuinely rounded complement.
    zero = np.zeros((3,3)); offset = np.zeros(3)
    original = station_position(points,zero,0,t,np.eye(3),offset)
    for shift in (2.**30,2.**40):
        moved = station_position(points+shift,zero,0,t,np.eye(3),offset)
        for k in range(3):
            assert sum((F(float(moved[j][k]))-F(float(original[j][k])) for j in (0,1)),F()) == F(shift)
        historical = v2_station(points+shift,zero,0,t,np.eye(3),offset)
        historical_origin = v2_station(points,zero,0,t,np.eye(3),offset)
        assert abs(sum((F(float(historical[j][0]))-F(float(historical_origin[j][0])) for j in (0,1)),F())-F(shift)) > F(1,10**11)


@pytest.fixture
def state_contracts(monkeypatch):
    import test_ge_beam3_curved_p5_coordinate_state as previous
    from anyfesolver._ge_beam3_p5_centered import codec
    from anyfesolver._ge_beam3_p5_centered.positions import rest_mass
    original = previous.problem
    def make_problem(*,curved=False,plastic=False,cls=NativeP5BeamElement):
        return original(curved=curved,plastic=plastic,cls=cls)
    for name,value in dict(problem=make_problem,NativeP5BeamElement=NativeP5BeamElement,codec=codec,
        prepare=prepare,solve_elastic_modes=solve_elastic_modes,reference_modes=reference_modes,rest_mass=rest_mass).items():
        monkeypatch.setattr(previous,name,value)
    return previous


@pytest.mark.parametrize('strain', [2.**-20,2.**-54,-2.**-55])
def test_preserved_exact_axial_state_contract_on_centered_native(state_contracts,strain):
    state_contracts.test_actual_native_axial_commit_retains_force_work_and_replay(strain)


@pytest.mark.parametrize('mutation', ['low','high','total','policy','missing','response'])
def test_resealed_coordinate_mutation_rejection_on_centered_native(state_contracts,mutation):
    state_contracts.test_resealed_coordinate_mutations_fail_before_commit(mutation)


@pytest.mark.parametrize('mutation', ['duplicate','nonfinite','extra_key','wrong_shape','wrong_number_type'])
def test_typed_codec_rejection_on_centered_native(state_contracts,mutation):
    state_contracts.test_typed_coordinate_codec_rejects_malformed_payload(mutation)


@pytest.mark.parametrize('plastic',[False,True])
def test_native_tangent_directional_agreement_after_centered_adoption(state_contracts,plastic):
    state_contracts.test_native_coordinate_tangent_remains_the_force_directional_derivative(plastic)


@pytest.fixture
def controls(monkeypatch,state_contracts):
    import test_ge_beam3_curved_p5_native_controls as previous
    def make_problem():
        model,element = state_contracts.problem(curved=True,plastic=True)
        load = LoadCase('tip'); load.add_nodal_load(3,forces=np.array([.025,-.01,.005]))
        return model,element,load
    monkeypatch.setattr(previous,'problem',make_problem)
    return previous


@pytest.mark.parametrize('name',[
    'test_actual_displacement_control_reaches_curved_plastic_state',
    'test_actual_arc_length_reaches_native_committed_state',
    'test_native_loading_unloading_reversal_program_preserves_accepted_origins',
    'test_native_displacement_checkpoint_split_matches_uninterrupted',
    'test_native_arc_length_checkpoint_split_matches_uninterrupted'])
def test_centered_native_preserves_control_and_history_contracts(controls,name):
    getattr(controls,name)()


@pytest.mark.parametrize('mode',['force','displacement','arc_length'])
def test_centered_native_cancellation_preserves_accepted_state(controls,monkeypatch,mode):
    controls.test_cancellation_during_unaccepted_native_trial_discards_it(monkeypatch,mode)


def test_mutated_reference_coefficients_are_detected_before_native_use():
    model,_ = problem(shift=2.**30); element = model.mesh.elements[1]
    before = element.to_dict()['reference_fingerprint']
    element.core.reference._coefficient_high.setflags(write=True)
    element.core.reference._coefficient_high[0,0] *= 1.001
    assert element.core.reference.fingerprint() != before
    with pytest.raises(ValueError,match='model identity'):
        element.init_model_bound_nonlinear_state(model.mesh,element.core.section,1)


def test_centered_rest_inertia_rejects_finite_velocity_route():
    from anyfesolver._ge_beam3_p5_centered.mass import _CenteredRestInertia
    model,_ = problem(); ref = model.mesh.elements[1].core.reference
    mass = _CenteredRestInertia(ref,section_mass(),order=8)
    velocity = np.zeros(24); velocity[0] = 1.
    with pytest.raises(ValueError,match='rest linearization only'):
        mass.evaluate(ref.coordinates,np.tile(np.eye(3),(2,1,1)),velocity,np.zeros(24))
