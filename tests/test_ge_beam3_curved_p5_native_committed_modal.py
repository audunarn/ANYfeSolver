"""Accepted-origin current operator checks; no prestressed qualification claim."""

from copy import deepcopy
from dataclasses import replace

import numpy as np
import pytest

from anyfesolver.fe_core import FEModel
from anyfesolver.boundary import BoundaryCondition, LoadCase
from anyfesolver.control import CancellationToken, SolveCancelled
from anyfesolver.nonlinear_static import solve_static_nonlinear
from anyfesolver._native_stationary_spectrum import solve_stationary_spectrum
from docs.reference_cases.ge_beam3_curved_p5_native_driver_probe import DriverP5Element
from docs.reference_cases.ge_beam3_curved_p5_native_material_probe import canonical, seal
from docs.reference_cases.ge_beam3_curved_p5_native_committed_modal_probe import prepare, solve_elastic_modes, _operator
from docs.reference_cases.ge_beam3_curved_p5_native_modal_probe import solve as reference_modes
from docs.reference_cases.ge_beam3_curved_p5_finite_probe import CurvedFiniteProbe
from docs.reference_cases.ge_beam3_curved_p5_section_probe import DirectedHardeningSectionProbe
from docs.reference_cases.ge_beam3_curved_p5_algebra_probe import rotation
from docs.reference_cases.ge_beam3_curved_p5_continuum_probe import parabolic_references
from test_ge_beam3_curved_p5_algebra_probe import reference, assert_scaled_close
from test_ge_beam3_curved_p5_mass_probe import section_mass
from test_ge_beam3_curved_p5_nonlinear_mixed_probe import law


def problem(*, curved=False, plastic=False, yield_force=1000.):
    ref = reference(.4 if curved else 0.)
    section = law(.02 if plastic else 1000.) if curved else DirectedHardeningSectionProbe(
        np.diag([100.,30.,30.,2.,3.,4.]), np.array([1.,0.,0.,0.,0.,0.]), yield_force, .4)
    model = FEModel('private-p5-committed-modes')
    for i, point in enumerate(ref.coordinates, 1): model.add_node(i, *point)
    element = DriverP5Element(1, (1,2,3), ref, section)
    model.add_element(1, element); model.materials[element.material_name] = element.core.section
    model.add_boundary_condition(BoundaryCondition('root', [1],
        {name: 0. for name in ('ux','uy','uz','rx','ry','rz')}))
    return model, element


def accepted(model, forces):
    load = LoadCase('spatial-dead-tip'); load.add_nodal_load(3, forces=np.array(forces))
    result = solve_static_nonlinear(model, load, num_steps=2, max_iterations=12, tolerance=1e-12,
                                   num_layers=1, min_step_fraction=1., emit_restart_checkpoint=True)
    assert result.status == 'completed', (result.status, result.info)
    force = np.zeros(18); force[12:15] = forces
    return result, force


@pytest.fixture(scope='module')
def curved_elastic():
    model, element = problem(curved=True)
    result, force = accepted(model, [.025,-.01,.005])
    return model, element, result, force


@pytest.fixture(scope='module')
def curved_plastic():
    model, element = problem(curved=True, plastic=True)
    result, force = accepted(model, [.025,-.01,.005])
    return model, element, result, force


def test_current_elastic_operator_and_modes_from_actual_native_newton(curved_elastic):
    model, element, result, force = curved_elastic
    before = canonical(result.element_states)
    packet, modes = solve_elastic_modes(model, result.element_states, result.displacements,
                                      {1:section_mass()}, force)
    assert packet.stiffness.shape == packet.mass.shape == (24,24)
    assert modes.dynamic_map.shape == (24,12)
    assert np.all(modes.eigenvalues > 0.)
    assert canonical(result.element_states) == before
    assert packet.operators[0][1].elastic_interior
    assert not packet.production_qualified and not modes.production_qualified
    assert modes.normalized_residual <= 1e-11
    assert modes.operator_identity == packet.identity
    assert np.array_equal(modes.spatial_dead_forces, force)
    assert len(modes.external_work_identity) == 64


def test_current_hessian_matches_separate_moment_reduced_elastic_potential(curved_elastic):
    model, element, result, _ = curved_elastic
    packet, _ = prepare(model, result.element_states, result.displacements, {1:section_mass()})
    inner = result.element_states[1]['material_state']; response = inner['response']
    oracle = CurvedFiniteProbe(element.core.reference, element.core.section._elastic, order=8)
    jet = oracle._jet(inner['committed_positions'],
        inner['committed_nodal_rotation_matrices']@element.core.reference.nodal_triads,
        response.local_rotations, external=True)
    assert_scaled_close(packet.stiffness, jet.hessian)
    assert_scaled_close(packet.internal_force, jet.gradient)
    assert_scaled_close(packet.operators[0][1].nodal_condensed_tangent, response.tangent)


def test_current_operator_is_not_reference_stiffness_or_reference_mass(curved_elastic):
    model, _, result, _ = curved_elastic
    packet, _ = prepare(model, result.element_states, result.displacements, {1:section_mass()})
    origin = reference_modes(model, {1:section_mass()})
    assert np.linalg.norm(packet.stiffness-origin.full_stiffness) > 1e-3
    assert np.linalg.norm(packet.mass-origin.full_mass) > 1e-4
    assert np.array_equal(packet.mass[:,[3,4,5,9,10,11,15,16,17]], np.zeros((24,9)))


def test_stress_free_origin_matches_native_reference_path():
    model, element = problem(curved=True)
    states = {1:element.init_model_bound_nonlinear_state(model.mesh, element.core.section, 1)}
    packet, made = solve_elastic_modes(model, states, np.zeros(18), {1:section_mass()}, np.zeros(18))
    expected = reference_modes(model, {1:section_mass()})
    assert_scaled_close(packet.stiffness, expected.full_stiffness)
    assert_scaled_close(packet.mass, expected.full_mass)
    assert_scaled_close(made.eigenvalues, expected.eigenvalues)


def test_straight_axial_tension_stiffens_and_compression_softens():
    values = []
    for axial in (-.5, 0., .5):
        model, element = problem()
        if axial:
            result, force = accepted(model, [axial,0.,0.])
            states, displacement = result.element_states, result.displacements
        else:
            states = {1:element.init_model_bound_nonlinear_state(model.mesh, element.core.section, 1)}
            displacement, force = np.zeros(18), np.zeros(18)
        _, modes = solve_elastic_modes(model, states, displacement, {1:section_mass()}, force)
        values.append(modes.eigenvalues[:2])
    assert np.all(values[0] < values[1])
    assert np.all(values[1] < values[2])


def test_accepted_plastic_hessian_is_preserved_but_not_called_modal_modulus(curved_plastic):
    model, _, result, force = curved_plastic
    before = canonical(result.element_states)
    packet, _ = prepare(model, result.element_states, result.displacements, {1:section_mass()})
    operator = packet.operators[0][1]
    assert not operator.elastic_interior
    assert_scaled_close(operator.nodal_condensed_tangent, result.element_states[1]['material_state']['response'].tangent)
    with pytest.raises(ValueError, match='vibration branch'):
        solve_elastic_modes(model, result.element_states, result.displacements, {1:section_mass()}, force)
    assert canonical(result.element_states) == before


@pytest.mark.parametrize('target', ['displacement','state_hash','resealed_response','missing_state','wrong_force','moment'])
def test_corrupt_or_unsupported_current_inputs_fail(curved_elastic, target):
    model, _, result, force = curved_elastic
    states = deepcopy(result.element_states); displacement = result.displacements.copy(); force = force.copy()
    if target == 'displacement': displacement[12] += .01
    elif target == 'state_hash': states[1]['state_sha256'] = '0'*64
    elif target == 'resealed_response':
        inner = states[1]['material_state']; inner['response'] = replace(inner['response'], potential=inner['response'].potential+.01)
        states[1]['material_state'] = seal(inner); states[1] = seal(states[1])
    elif target == 'missing_state': states = {}
    elif target == 'wrong_force': force[12] += .1
    else: force[15] = .01
    with pytest.raises(ValueError): solve_elastic_modes(model, states, displacement, {1:section_mass()}, force)


def test_current_packet_determinism_and_cancellation(curved_elastic):
    model, _, result, force = curved_elastic
    first = solve_elastic_modes(model, result.element_states, result.displacements, {1:section_mass()}, force)
    second = solve_elastic_modes(model, result.element_states, result.displacements, {1:section_mass()}, force)
    assert canonical(first) == canonical(second)
    with pytest.raises(ValueError): first[0].stiffness.setflags(write=True)
    token = CancellationToken(); token.cancel()
    with pytest.raises(SolveCancelled):
        solve_elastic_modes(model, result.element_states, result.displacements, {1:section_mass()}, force, cancellation_token=token)


def test_signed_spectrum_retains_instability_and_does_not_require_positive_k():
    k = np.diag([-2.,0.,3.,4.]); k[0,3] = k[3,0] = 1.
    m = np.diag([2.,1.,1.,0.])
    made = solve_stationary_spectrum(k,m,(0,1,2,3),(3,),num_modes=3)
    assert_scaled_close(made.eigenvalues, np.array([-1.125,0.,3.]))
    assert made.full_modes[3,0] != 0.


@pytest.mark.parametrize('target', ['algebraic_mass','singular_trace','negative_trace','undeclared_kernel','nonsymmetric','nonfinite','bad_dofs'])
def test_signed_pencil_validation(target):
    k = np.diag([2.,3.,4.]); m = np.diag([2.,1.,0.]); algebraic = (2,)
    if target == 'algebraic_mass': m[2,2] = 1e-100
    elif target == 'singular_trace': k[2,2] = 0.
    elif target == 'negative_trace': k[2,2] = -1.
    elif target == 'undeclared_kernel': algebraic = ()
    elif target == 'nonsymmetric': k[0,1] = 1.
    elif target == 'nonfinite': k[0,0] = np.nan
    else: algebraic = (2,2)
    with pytest.raises((ValueError,np.linalg.LinAlgError)):
        solve_stationary_spectrum(k,m,(0,1,2),algebraic,num_modes=2)


def test_augmented_operator_superposed_rigid_motion_covariance(curved_elastic):
    model, element, result, _ = curved_elastic
    original = result.element_states[1]['material_state']; before = canonical(result.element_states)
    base = _operator(element, original, section_mass())
    r = rotation([.2,-.3,.1]); shift = np.array([.2,.1,-.3])
    x = original['committed_positions']@r.T+shift
    operators = r@original['committed_nodal_rotation_matrices']
    response = element.core._solve(x, operators, original['origins'])
    total = original['committed_total_u'].copy(); total.reshape(3,6)[:,:3] = x-element.core.reference.coordinates
    transformed = element.core._state(original['epoch'],total,x,operators,original['origins'],response)
    made = _operator(element, transformed, section_mass())
    transport = np.kron(np.eye(8),r)
    assert_scaled_close(made.stiffness, transport@base.stiffness@transport.T)
    assert_scaled_close(made.mass, transport@base.mass@transport.T)
    assert_scaled_close(made.internal_force, transport@base.internal_force)
    assert canonical(result.element_states) == before


def test_committed_modal_guard_detects_change_after_packet_construction():
    model, element = problem()
    states = {1:element.init_model_bound_nonlinear_state(model.mesh, element.core.section, 1)}
    packet, guard = prepare(model, states, np.zeros(18), {1:section_mass()})
    model.boundary_conditions[0].dof_constraints.pop('rx')
    with pytest.raises(ValueError, match='frozen inputs'): guard()


def test_oversize_candidate_fails_before_factor_or_state_evaluation(monkeypatch):
    from docs.reference_cases import ge_beam3_curved_p5_native_modal_probe as reference_adapter
    model, _ = problem()
    for i in range(4, 48): model.add_node(i, float(i), 0., 0.)
    def forbidden(*args, **kwargs): raise AssertionError('factor should not be evaluated')
    monkeypatch.setattr(reference_adapter, 'reference_kinetic_factors', forbidden)
    with pytest.raises(ValueError, match='before factor'):
        prepare(model, {}, np.zeros(model.mesh.dof_manager.total_dofs), {1:section_mass()})


def test_yield_boundary_has_no_automatic_branch_authority():
    model, _ = problem(yield_force=1.)
    result, force = accepted(model, [1.,0.,0.])
    packet, _ = prepare(model, result.element_states, result.displacements, {1:section_mass()})
    assert not packet.operators[0][1].elastic_interior
    with pytest.raises(ValueError, match='yield-boundary'):
        solve_elastic_modes(model, result.element_states, result.displacements, {1:section_mass()}, force)


@pytest.fixture(scope='module')
def two_element_state():
    model = FEModel('private-two-element-current-modes')
    refs = parabolic_references(.4, 2)
    ids = (7,23,55,81,103)
    for i, point in zip(ids, np.vstack((refs[0].coordinates, refs[1].coordinates[1:]))):
        model.add_node(i, *point)
    for index, ref in enumerate(refs):
        element = DriverP5Element(index+1, ids[2*index:2*index+3], ref, law(1000.))
        model.add_element(index+1, element); model.materials[element.material_name] = element.core.section
    model.add_boundary_condition(BoundaryCondition('root', [7],
        {name:0. for name in ('ux','uy','uz','rx','ry','rz')}))
    load = LoadCase('dead-tip'); load.add_nodal_load(103, forces=np.array([.005,-.002,.001]))
    result = solve_static_nonlinear(model, load, num_steps=2, max_iterations=12, tolerance=1e-12,
                                   num_layers=1, min_step_fraction=1.)
    assert result.status == 'completed', (result.status, result.info)
    force = np.zeros(30); force[24:27] = [.005,-.002,.001]
    return model, result, force


def test_actual_two_element_current_modes_balance_shared_rotations(two_element_state):
    model, result, force = two_element_state
    before = canonical(result.element_states)
    packet, modes = solve_elastic_modes(model, result.element_states, result.displacements,
                                      {1:section_mass(),2:section_mass()}, force)
    assert packet.stiffness.shape == (42,42)
    assert modes.dynamic_map.shape == (42,24)
    assert len(packet.algebraic_dofs) == 12
    incident = []
    for (eid, operator), (_, local_dofs) in zip(packet.operators, packet.internal_layout):
        dofs = tuple(int(i) for i in model.mesh.elements[eid].get_dof_mapping(model.mesh))+local_dofs
        local_force = operator.stiffness@modes.full_modes[list(dofs)]
        incident.append(local_force[15:18] if eid == 1 else local_force[3:6])
    assert np.linalg.norm(incident[0]) > 1e-4
    assert_scaled_close(incident[0]+incident[1], np.zeros((3,6)))
    assert canonical(result.element_states) == before


def test_shared_accepted_rotation_mismatch_fails_before_modal_assembly(two_element_state):
    model, result, _ = two_element_state
    states = deepcopy(result.element_states)
    element = model.mesh.elements[2]; original = states[2]['material_state']
    matrices = original['committed_nodal_rotation_matrices'].copy()
    matrices[0] = rotation([.001,0.,0.])@matrices[0]
    response = element.core._solve(original['committed_positions'], matrices, original['origins'])
    # Mechanically replayable local state, but impossible as a shared global
    # nodal rotation. The native store must reject it before augmented assembly.
    inner = element.core._state(original['epoch'], original['committed_total_u'],
        original['committed_positions'], matrices, original['origins'], response)
    states[2] = element._wrap(inner)
    with pytest.raises(ValueError):
        prepare(model, states, result.displacements, {1:section_mass(),2:section_mass()})
