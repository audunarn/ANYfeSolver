"""Native signed-mode ownership, state safety and small covariance checks."""

from copy import deepcopy
import numpy as np
import pytest
import anyfesolver._ge_beam3_signed_loaded_modes as adapter
from anyfesolver._ge_beam3_loaded_modal import solve_elastic_modes
from anyfesolver._ge_beam3_p5_loads.core import canonical
from anyfesolver.control import CancellationToken, SolveCancelled
from test_ge_beam3_native_load_state import problem
from test_ge_beam3_loaded_modal import accepted
from test_ge_beam3_curved_p5_mass_probe import section_mass


@pytest.fixture(scope='module')
def loaded():
    model = problem()
    result, states, force = accepted(model, nodal=((55, .025, -.01, .005),))
    return model, result, states, force


def solve(model, result, states, force, **kwargs):
    return adapter.solve_signed_loaded_modes(model, states, result.displacements,
        {i: section_mass() for i in model.mesh.elements}, force, load_parameter=result.parameter,
        bounds=(-10000., 1000000.), **kwargs)


def test_curved_coupled_native_modes_match_preserved_moderate_dense_solution(loaded, tmp_path):
    model, result, states, force = loaded; saved = canonical(states)
    packet, modes = solve(*loaded)
    old_packet, old = solve_elastic_modes(model, states, result.displacements, {1: section_mass()},
        force, load_parameter=result.parameter)
    assert packet.identity == old_packet.identity == modes.operator_identity
    np.testing.assert_allclose(modes.eigenvalues, old.eigenvalues, rtol=1e-11, atol=1e-10)
    correlation = old.full_modes.T@packet.mass@modes.full_modes
    np.testing.assert_allclose(np.abs(correlation), np.eye(6), rtol=1e-11, atol=1e-11)
    assert canonical(states) == saved and modes.spectral_residual <= 1e-11
    assert not modes.buckling_factor_authorized and not modes.production_qualified
    with (tmp_path/'curved-native-modes.json').open('xb') as stream: stream.write(canonical(modes))


def test_two_element_shared_trace_modes_survive_large_translation(tmp_path):
    outputs = []
    for shift in (0., 2.**40):
        model = problem(count=2, shift=shift)
        result, states, force = accepted(model); saved = canonical(states)
        packet, modes = solve(model, result, states, force)
        assert modes.full_modes.shape == (42, 6)
        np.testing.assert_allclose(modes.full_modes.T@packet.mass@modes.full_modes, np.eye(6), atol=1e-11)
        outputs.append(canonical(dict(eigenvalues=modes.eigenvalues, modes=modes.full_modes,
            spectral_residual=modes.spectral_residual, production_qualified=False)))
        assert canonical(states) == saved
    assert outputs[0] == outputs[1]
    with (tmp_path/'translated-native-modes.json').open('xb') as stream: stream.write(outputs[0])


@pytest.mark.parametrize('mutation', ['force', 'moment', 'state', 'parameter'])
def test_invalid_state_or_external_work_fails_before_signed_kernel(loaded, monkeypatch, mutation):
    model, result, states, force = loaded; states = deepcopy(states); force = force.copy(); p = result.parameter
    if mutation == 'force': force[12] += .1
    if mutation == 'moment': force[15] = .1
    if mutation == 'state': states = {}
    if mutation == 'parameter': p = .5
    monkeypatch.setattr(adapter, 'solve_signed_factor_modes', lambda *a, **k: pytest.fail('invalid signed kernel entry'))
    with pytest.raises(ValueError):
        adapter.solve_signed_loaded_modes(model, states, result.displacements, {1: section_mass()}, force,
            load_parameter=p, bounds=(-10000., 1000000.))


def test_model_mutation_inside_kernel_is_detected_after_return(loaded, monkeypatch):
    model, result, states, force = loaded
    real = adapter.solve_signed_factor_modes; dofs = model.mesh.dof_manager.total_dofs
    def mutate(*a, **k):
        output = real(*a, **k)
        model.mesh.dof_manager._total_dofs = dofs+1
        return output
    monkeypatch.setattr(adapter, 'solve_signed_factor_modes', mutate)
    try:
        with pytest.raises(ValueError, match='changed'): solve(*loaded)
    finally:
        model.mesh.dof_manager._total_dofs = dofs


def test_cancel_before_preparation(loaded, monkeypatch):
    token = CancellationToken(); token.cancel('stop')
    monkeypatch.setattr(adapter, 'prepare', lambda *a, **k: pytest.fail('prepare after cancel'))
    with pytest.raises(SolveCancelled): solve(*loaded, cancellation_token=token)


def test_plastic_increment_is_rejected_but_elastic_unloading_keeps_history():
    model = problem(plastic=True)
    result, states, force = accepted(model); saved = canonical(states)
    with pytest.raises(ValueError, match='plastic or yield-boundary'): solve(model, result, states, force)
    assert canonical(states) == saved
    model = problem(plastic=True)
    result, states, force = accepted(model, targets=(.5, 1., 0.)); saved = canonical(states)
    packet, modes = solve(model, result, states, force)
    assert all(op.elastic_interior for _, op in packet.operators)
    assert np.all(modes.eigenvalues > 0.) and canonical(states) == saved
    element = model.mesh.elements[1]
    decoded = element.validate_model_bound_nonlinear_state(model.mesh, element.core.section, states[1], 1)
    assert any(h.accumulated > 0. for h in decoded['material_state']['histories'])


def test_curved_mode_shapes_covary_under_proper_frame_rotation():
    from anyfesolver.fe_core import FEModel
    from anyfesolver._ge_beam3_centered_reference import CenteredCurvedBeam3ReferenceGeometry
    from anyfesolver._ge_beam3_p5_loads import NativeP5BeamElement
    old = problem(); rotation = np.array([[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]])
    new = FEModel('signed-rotated-loaded-native')
    for i, node in old.mesh.nodes.items(): new.add_node(i, *(rotation@node.coords()))
    for i, previous in old.mesh.elements.items():
        ref = previous.core.reference
        moved = CenteredCurvedBeam3ReferenceGeometry(ref.coordinates@rotation.T, rotation@ref.nodal_triads)
        e = NativeP5BeamElement(i, previous.node_ids, moved, previous.core.section,
            line_force=rotation@previous.core.line_force)
        new.add_element(i, e); new.materials[e.material_name] = e.core.section
    for boundary in old.boundary_conditions: new.add_boundary_condition(deepcopy(boundary))
    outputs = []
    for model in (old, new):
        result, states, force = accepted(model)
        outputs.append(solve(model, result, states, force))
    transform = np.kron(np.eye(8), rotation)
    packet, modes = outputs[1]; original = outputs[0][1]
    np.testing.assert_allclose(modes.eigenvalues, original.eigenvalues, rtol=1e-11, atol=1e-11)
    correlation = (transform@original.full_modes).T@packet.mass@modes.full_modes
    np.testing.assert_allclose(np.abs(correlation), np.eye(6), rtol=1e-11, atol=1e-11)


def test_free_curved_beam_keeps_complete_analytical_six_rigid_mode_space():
    from scipy import linalg
    model = problem(); model.boundary_conditions.clear()
    e = model.mesh.elements[1]; total = np.zeros(18)
    states = {1: e.init_model_bound_nonlinear_state(model.mesh, e.core.section, 1)}
    saved = canonical(states)
    packet, modes = adapter.solve_signed_loaded_modes(model, states, total, {1: section_mass()},
        np.zeros(18), load_parameter=0., bounds=(-10000., 1000000.), num_modes=7)
    assert np.max(np.abs(modes.eigenvalues[:6])) <= 1e-11
    assert modes.eigenvalues[6] > 1e-6
    rigid = np.zeros((24, 6))
    for n, node in enumerate(model.mesh.nodes.values()):
        x, y, z = node.coords()
        rigid[6*n:6*n+3, :3] = np.eye(3)
        rigid[6*n:6*n+3, 3:] = -np.array([[0., -z, y], [z, 0., -x], [-y, x, 0.]])
        rigid[6*n+3:6*n+6, 3:] = np.eye(3)
    rigid[18:21, 3:] = rigid[21:24, 3:] = np.eye(3)
    mass_r = linalg.cholesky(rigid.T@packet.mass@rigid)
    orthogonal = linalg.solve_triangular(mass_r.T, rigid.T, lower=True).T
    correlation = orthogonal.T@packet.mass@modes.full_modes[:, :6]
    np.testing.assert_allclose(correlation.T@correlation, np.eye(6), atol=1e-11, rtol=1e-11)
    assert canonical(states) == saved
