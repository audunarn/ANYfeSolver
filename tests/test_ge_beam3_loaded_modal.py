"""Loaded equilibrium/mass/pencil development checks, not modal qualification."""

from copy import deepcopy
import json

import numpy as np
import pytest

from anyfesolver import _ge_beam3_loaded_modal as modal
from anyfesolver._ge_beam3_load_program import ForceProgram, solve_force_program
from anyfesolver._ge_beam3_p5_loads.core import canonical
from anyfesolver._ge_beam3_centered_mixed import CenteredStationaryBeam
from anyfesolver._ge_beam3_p5_centered.mass import reference_kinetic_factors
from anyfesolver.control import CancellationToken, SolveCancelled
from docs.reference_cases import ge_beam3_centered_line_load_oracle as load_oracle
from docs.reference_cases.ge_beam3_curved_p5_finite_probe import CurvedFiniteProbe
from test_ge_beam3_native_load_state import problem
from test_ge_beam3_curved_p5_mass_probe import section_mass


def close(actual, expected, tolerance=1e-11):
    assert np.linalg.norm(np.asarray(actual)-expected) <= tolerance*max(1., np.linalg.norm(expected))


def accepted(model, targets=(.5, 1.), nodal=()):
    result = solve_force_program(model, ForceProgram(targets, nodal))
    assert result.status == 'completed', result.failure
    states = {row['element_id']: row['state'] for row in json.loads(result.checkpoint)['element_states']}
    force = np.zeros(len(result.displacements))
    for node, *vector in nodal:
        force[list(model.mesh.dof_manager.get_node_dofs(node)[:3])] = result.parameter*np.array(vector)
    return result, states, force


@pytest.fixture(scope='module')
def loaded():
    model = problem()
    result, states, force = accepted(model, nodal=((55, .025, -.01, .005),))
    return model, result, states, force


def test_actual_loaded_native_equilibrium_has_retained_cell_modes(loaded):
    model, result, states, force = loaded; before = canonical(states)
    packet, modes = modal.solve_elastic_modes(model, states, result.displacements, {1: section_mass()},
        force, load_parameter=result.parameter)
    assert packet.stiffness.shape == packet.mass.shape == (24, 24)
    assert modes.dynamic_map.shape == (24, 12) and np.all(modes.eigenvalues > 0.)
    assert modes.normalized_residual <= 1e-11 and modes.operator_identity == packet.identity
    assert canonical(states) == before
    assert packet.operators[0][1].elastic_interior
    assert not packet.production_qualified and not modes.production_qualified and not modes.buckling_factor_authorized
    rotations = [3, 4, 5, 9, 10, 11, 15, 16, 17]
    np.testing.assert_array_equal(packet.mass[:, rotations], np.zeros((24, 9)))
    assert np.linalg.norm(packet.mass[18:, 18:]) > 0.


def test_net_hessian_includes_independently_reconstructed_curved_load_work(loaded):
    model, result, states, force = loaded
    packet, _ = modal.prepare(model, states, result.displacements, {1: section_mass()}, load_parameter=result.parameter)
    element = model.mesh.elements[1]
    inner = element.validate_model_bound_nonlinear_state(model.mesh, element.core.section, states[1], 1)['material_state']
    response = inner['response']; ref = element.core.reference
    material = CenteredStationaryBeam(ref, element.core.section, order=8,
        position_low=inner['committed_position_low'], origins=inner['origins'])
    full = material.evaluate(inner['committed_positions'], inner['committed_nodal_rotation_matrices']@ref.nodal_triads,
        response.local_rotations, response.moments)
    # The load reconstruction uses rational Q2 geometry and closed-form
    # rotations, not the producer's Jet2 work or assembled load matrices.
    work = load_oracle.evaluate(ref.coordinates, result.displacements.reshape(3, 6)[:, :3],
        response.local_rotations, element.core.line_force)
    h = full.hessian
    material_k = h[:24, :24]-h[:24, 24:]@np.linalg.solve(h[24:, 24:], h[24:, :24])
    assert np.linalg.norm(work.hessian) > 1e-5
    close(packet.stiffness, material_k-result.parameter*work.hessian[:24, :24])
    close(packet.net_residual, full.residual[:24]-result.parameter*work.force[:24])
    assert np.linalg.norm(packet.stiffness-material_k) > 1e-5


def test_loaded_operator_matches_separate_elastic_potential_with_shared_AD(loaded):
    # This is a separately reduced expression, NOT an independent AD oracle.
    model, result, states, force = loaded; element = model.mesh.elements[1]
    packet, _ = modal.prepare(model, states, result.displacements, {1: section_mass()}, load_parameter=result.parameter)
    inner = element.validate_model_bound_nonlinear_state(model.mesh, element.core.section, states[1], 1)['material_state']
    probe = CurvedFiniteProbe(element.core.reference, element.core.section._elastic, order=8,
        line_force=result.parameter*element.core.line_force)
    jet = probe._jet(inner['committed_positions'], inner['committed_nodal_rotation_matrices']@element.core.reference.nodal_triads,
        inner['response'].local_rotations, external=True)
    close(packet.stiffness, jet.hessian)
    close(packet.net_residual, jet.gradient)


def test_current_mass_matches_separate_lifted_velocity_integral(loaded):
    model, result, states, force = loaded; element = model.mesh.elements[1]
    packet, _ = modal.prepare(model, states, result.displacements, {1: section_mass()}, load_parameter=result.parameter)
    inner = element.validate_model_bound_nonlinear_state(model.mesh, element.core.section, states[1], 1)['material_state']
    speed = np.cos(np.arange(24)+.3); ref = element.core.reference; energy = 0.
    points, weights = np.polynomial.legendre.leggauss(8)
    for cell in (0, 1):
        rotation = inner['response'].local_rotations[cell]; spin = speed[18+3*cell:21+3*cell]
        for point, weight in zip(points, weights):
            t = float((point+1)/2); xi = cell-1+t
            offset = rotation@ref.half_cell_lift(cell, t)
            velocity = (1-t)*speed[6*cell:6*cell+3]+t*speed[6*(cell+1):6*(cell+1)+3]+np.cross(spin, offset)
            frame = rotation@ref.frame(xi)
            material = np.r_[frame.T@velocity, frame.T@spin]
            energy += weight*ref.jacobian(xi)*float(material@section_mass()@material)/4
    close(float(speed@packet.mass@speed)/2, energy)


def test_stress_free_reference_pencil_and_six_free_body_modes():
    model = problem(); model.boundary_conditions.clear(); element = model.mesh.elements[1]
    states = {1: element.init_model_bound_nonlinear_state(model.mesh, element.core.section, 1)}
    packet, modes = modal.solve_elastic_modes(model, states, np.zeros(18), {1: section_mass()},
        np.zeros(18), load_parameter=0., num_modes=9)
    factors = reference_kinetic_factors(element.core.reference, element.core.section._elastic, section_mass(), order=8)
    close(packet.mass, factors.full.T@factors.full)
    close(packet.stiffness, factors.uncondensed_stiffness_factor.T@factors.uncondensed_stiffness_factor)
    scale = max(1., np.linalg.norm(packet.stiffness)/np.linalg.norm(packet.mass))
    assert np.max(np.abs(modes.eigenvalues[:6])) <= 1e-11*scale
    assert np.all(modes.eigenvalues[6:] > 1e-11*scale)


@pytest.mark.parametrize('target', ['missing_state', 'parameter', 'displacement', 'state_hash', 'inertia', 'force', 'moment'])
def test_malformed_or_inconsistent_inputs_fail(loaded, target):
    model, result, states, force = loaded
    states = deepcopy(states); total = result.displacements.copy(); p = result.parameter
    inertias = {1: section_mass()}; force = force.copy()
    if target == 'missing_state': states = {}
    elif target == 'parameter': p = .5
    elif target == 'displacement': total[12] += .01
    elif target == 'state_hash':
        value = json.loads(states[1]['payload']); value['state_sha256'] = '0'*64
        states[1]['payload'] = canonical(value).decode('ascii')
    elif target == 'inertia': inertias[1][0, 0] = -1.
    elif target == 'force': force[12] += .1
    elif target == 'moment': force[15] = .01
    with pytest.raises((ValueError, np.linalg.LinAlgError)):
        modal.solve_elastic_modes(model, states, total, inertias, force, load_parameter=p)


def test_plastic_accepted_tangent_is_not_mislabeled_as_modal_modulus():
    model = problem(plastic=True); result, states, force = accepted(model)
    before = canonical(states)
    packet, _ = modal.prepare(model, states, result.displacements, {1: section_mass()}, load_parameter=result.parameter)
    assert not packet.operators[0][1].elastic_interior
    with pytest.raises(ValueError, match='vibration branch'):
        modal.solve_elastic_modes(model, states, result.displacements, {1: section_mass()}, force, load_parameter=result.parameter)
    assert canonical(states) == before


def test_packet_and_state_snapshots_are_immutable(loaded):
    model, result, states, force = loaded
    packet, guard = modal.prepare(model, states, result.displacements, {1: section_mass()}, load_parameter=result.parameter)
    with pytest.raises(ValueError): packet.mass.setflags(write=True)
    object.__setattr__(packet, 'load_parameter', .5)
    with pytest.raises(ValueError, match='packet changed'): guard()


def test_model_guard_and_pre_cancel_fail_closed():
    model = problem(); element = model.mesh.elements[1]
    states = {1: element.init_model_bound_nonlinear_state(model.mesh, element.core.section, 1)}
    packet, guard = modal.prepare(model, states, np.zeros(18), {1: section_mass()}, load_parameter=0.)
    model.boundary_conditions[0].dof_constraints.pop('rx')
    with pytest.raises(ValueError, match='frozen inputs'): guard()
    token = CancellationToken(); token.cancel()
    with pytest.raises(SolveCancelled):
        modal.prepare(problem(), states, np.zeros(18), {1: section_mass()}, load_parameter=0., cancellation_token=token)


def straight_model():
    from anyfesolver.fe_core import FEModel
    from anyfesolver.boundary import BoundaryCondition
    from anyfesolver._ge_beam3_centered_reference import CenteredCurvedBeam3ReferenceGeometry
    from anyfesolver._ge_beam3_p5_loads import NativeP5BeamElement, DirectedHardeningSection
    model = FEModel('loaded-straight-native')
    points = np.array([[-1., 0., 0.], [0., 0., 0.], [1., 0., 0.]])
    for i, point in enumerate(points, 1): model.add_node(i, *point)
    ref = CenteredCurvedBeam3ReferenceGeometry(points, np.tile(np.eye(3), (3, 1, 1)))
    section = DirectedHardeningSection(np.diag([100., 30., 30., 2., 3., 4.]),
        np.array([1., 0., 0., 0., 0., 0.]), 1000., .4)
    element = NativeP5BeamElement(1, (1, 2, 3), ref, section, line_force=np.zeros(3))
    model.add_element(1, element); model.materials[element.material_name] = element.core.section
    model.add_boundary_condition(BoundaryCondition('root', [1], {name: 0. for name in ('ux', 'uy', 'uz', 'rx', 'ry', 'rz')}))
    return model


def test_tension_stiffens_compression_softens_without_clipping_instability(tmp_path):
    spectra = []
    for axial in (-5., -.5, 0., .5):
        model = straight_model(); result, states, force = accepted(model, nodal=((3, axial, 0., 0.),))
        packet, modes = modal.solve_elastic_modes(model, states, result.displacements, {1: section_mass()},
            force, load_parameter=result.parameter)
        spectra.append(modes.eigenvalues)
        assert modes.negative_eigenvalues_retained and not modes.buckling_factor_authorized
    assert spectra[0][0] < 0.
    assert np.all(spectra[1][:2] < spectra[2][:2]) and np.all(spectra[2][:2] < spectra[3][:2])
    with (tmp_path/'axial-spectra.json').open('xb') as stream:
        stream.write(canonical(dict(loads=[-5., -.5, 0., .5], eigenvalues=spectra, production_qualified=False)))
    # Coarse branch/sign diagnostic, not an Euler factor or a buckling bracket.


def test_two_element_shared_traces_and_large_translation_are_preserved(tmp_path):
    outputs = []
    for shift in (0., 2.**40):
        model = problem(count=2, shift=shift)
        result, states, force = accepted(model)
        before = canonical(states)
        packet, modes = modal.solve_elastic_modes(model, states, result.displacements,
            {i: section_mass() for i in model.mesh.elements}, force, load_parameter=result.parameter)
        assert packet.stiffness.shape == packet.mass.shape == (42, 42)
        assert modes.dynamic_map.shape == (42, 24)
        assert len(packet.algebraic_dofs) == 12 and len(packet.internal_layout) == 2
        assert canonical(states) == before
        outputs.append(canonical(dict(stiffness=packet.stiffness, mass=packet.mass,
            force=packet.net_residual, eigenvalues=modes.eigenvalues, modes=modes.full_modes,
            production_qualified=False)))
    assert outputs[0] == outputs[1]
    with (tmp_path/'translated-pencil.json').open('xb') as stream: stream.write(outputs[0])


def test_loaded_pencil_covariance_under_proper_coordinate_reexpression():
    from anyfesolver.fe_core import FEModel
    from anyfesolver._ge_beam3_centered_reference import CenteredCurvedBeam3ReferenceGeometry
    from anyfesolver._ge_beam3_p5_loads import NativeP5BeamElement
    old = problem(); rotation = np.array([[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]])
    new = FEModel('rotated-loaded-native')
    for i, node in old.mesh.nodes.items(): new.add_node(i, *(rotation@node.coords()))
    for i, previous in old.mesh.elements.items():
        ref = previous.core.reference
        moved = CenteredCurvedBeam3ReferenceGeometry(ref.coordinates@rotation.T, rotation@ref.nodal_triads)
        element = NativeP5BeamElement(i, previous.node_ids, moved, previous.core.section,
            line_force=rotation@previous.core.line_force)
        new.add_element(i, element); new.materials[element.material_name] = element.core.section
    for boundary in old.boundary_conditions: new.add_boundary_condition(deepcopy(boundary))
    packets = []
    for model in (old, new):
        result, states, force = accepted(model)
        packets.append(modal.solve_elastic_modes(model, states, result.displacements,
            {1: section_mass()}, force, load_parameter=result.parameter))
    transform = np.kron(np.eye(8), rotation)
    close(packets[1][0].stiffness, transform@packets[0][0].stiffness@transform.T)
    close(packets[1][0].mass, transform@packets[0][0].mass@transform.T)
    close(packets[1][0].net_residual, transform@packets[0][0].net_residual)
    close(packets[1][1].eigenvalues, packets[0][1].eigenvalues)


def test_elastic_unloading_after_plastic_history_is_not_rejected_as_virgin_only():
    model = problem(plastic=True)
    result, states, force = accepted(model, targets=(.5, 1., 0.))
    before = canonical(states)
    packet, modes = modal.solve_elastic_modes(model, states, result.displacements,
        {1: section_mass()}, force, load_parameter=result.parameter)
    assert all(op.elastic_interior for _, op in packet.operators)
    assert modes.normalized_residual <= 1e-11 and canonical(states) == before
    decoded = model.mesh.elements[1].validate_model_bound_nonlinear_state(model.mesh,
        model.mesh.elements[1].core.section, states[1], 1)
    assert any(h.accumulated > 0. for h in decoded['material_state']['histories'])


@pytest.mark.parametrize('kind', ['partial_rotation', 'nonzero_support', 'oversize'])
def test_unsupported_geometry_constraints_fail_before_operator_evaluation(monkeypatch, kind):
    model = problem()
    if kind == 'partial_rotation': model.boundary_conditions[0].dof_constraints.pop('rx')
    elif kind == 'nonzero_support': model.boundary_conditions[0].dof_constraints['ux'] = .1
    elif kind == 'oversize':
        for i in range(100, 150): model.add_node(i, float(i), 0., 0.)
    def forbidden(*args): raise AssertionError('must not evaluate operator')
    monkeypatch.setattr(modal, '_operator', forbidden)
    with pytest.raises(ValueError):
        modal.prepare(model, {1: {}}, np.zeros(model.mesh.dof_manager.total_dofs), {1: section_mass()}, load_parameter=0.)


def test_packet_guard_detects_changed_global_dof_count():
    model = problem(); element = model.mesh.elements[1]
    states = {1: element.init_model_bound_nonlinear_state(model.mesh, element.core.section, 1)}
    packet, guard = modal.prepare(model, states, np.zeros(18), {1: section_mass()}, load_parameter=0.)
    model.mesh.dof_manager._total_dofs += 6
    with pytest.raises(ValueError): guard()
