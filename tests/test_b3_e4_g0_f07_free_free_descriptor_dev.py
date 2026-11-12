"""Bounded physical free-free rest-mode check on captured native operators."""

import numpy as np
import pytest
from scipy import linalg

from anyfesolver._ge_beam3_coupled_modes import capture
from anyfesolver._ge_beam3_shell_joint_state import decode
from anyfesolver._ge_beam3_variational_shell import response as shell_response
from test_ge_beam3_coupled_modes import arguments, fixture


def _rigid_nodal_basis(points):
    center = np.mean(points, axis=0)
    basis = np.zeros((6*len(points), 6))
    for index, point in enumerate(points):
        lever = point-center
        skew = np.array(((0., -lever[2], lever[1]),
                         (lever[2], 0., -lever[0]),
                         (-lever[1], lever[0], 0.)))
        basis[6*index:6*index+3, :3] = np.eye(3)
        basis[6*index:6*index+3, 3:] = -skew
        basis[6*index+3:6*index+6, 3:] = np.eye(3)
    return basis


@pytest.mark.parametrize("topology", ("Q4", "S3-V2D"))
def test_separately_assembled_shell_and_retained_beam_blocks_annihilate_rigid_fields(topology):
    owner = fixture(topology, curved=False)
    solved = owner.solve(stop_after=0)
    assert solved.status == "paused", solved.failure
    state = solved.state
    assembly = owner.assembly
    shell_residual, shell_tangent, _ = shell_response(
        assembly.model, assembly.element, state.shell_u,
        decode(state.shell_history), assembly.layers)
    shell_rigid = _rigid_nodal_basis(assembly.coordinates)
    assert np.linalg.norm(shell_residual) <= 1e-11
    assert np.linalg.norm(shell_tangent@shell_rigid)/(
        np.linalg.norm(shell_tangent)*np.linalg.norm(shell_rigid)) <= 1e-9

    beam = assembly.beam
    beam_residual, beam_tangent, _, _, _ = beam.assemble(
        state.mechanical, 0., state.beam_histories)
    beam_rigid = np.zeros((beam.count, 6))
    beam_rigid[:beam.nodal_count] = _rigid_nodal_basis(beam.reference_positions)
    for element in range(len(beam.elements)):
        first = beam.nodal_count+24*element
        beam_rigid[first:first+3, 3:] = np.eye(3)
        beam_rigid[first+3:first+6, 3:] = np.eye(3)
    assert np.linalg.norm(beam_residual/beam.scale) <= 1e-11
    assert np.linalg.norm(beam_tangent@beam_rigid)/(
        np.linalg.norm(beam_tangent)*np.linalg.norm(beam_rigid)) <= 1e-9


@pytest.mark.parametrize("topology", ("Q4", "S3-V2D"))
def test_native_point_candidate_has_six_free_free_physical_modes(topology):
    owner = fixture(topology, curved=False)
    solved = owner.solve(stop_after=0)
    assert solved.status == "paused", solved.failure
    packet = capture(owner, solved.state, **arguments(owner))
    state = solved.state
    _trial, full_residual, _candidate, _meta = owner._evaluate(
        state, state.mechanical, state.shell_u,
        state.multipliers, state.cursor)
    assert np.linalg.norm(full_residual/owner.scale) <= 1e-11
    count = len(packet.full_kinematic_dofs)
    rows = packet.constraints.shape[0]
    k = np.block([[packet.full_stiffness, packet.constraints.T],
                  [packet.constraints, np.zeros((rows, rows))]])
    m = linalg.block_diag(packet.full_mass, np.zeros((rows, rows)))
    np.testing.assert_array_equal(m[count:, :], 0.)
    np.testing.assert_array_equal(m[:, count:], 0.)
    values, vectors = linalg.eig(k, m)
    assert not np.isnan(values).any()
    constrained = linalg.null_space(packet.constraints)
    physical_mass = constrained.T@packet.full_mass@constrained
    finite_expected = np.linalg.matrix_rank(physical_mass, tol=1e-10)
    finite_values = values[np.isfinite(values)]
    assert len(finite_values) == finite_expected
    assert np.max(np.abs(finite_values.imag)) <= 1e-8*max(
        1., float(np.max(np.abs(finite_values.real))))
    finite = [(float(value.real), vectors[:, index])
              for index, value in enumerate(values)
              if np.isfinite(value) and
              abs(value.imag) <= 1e-8*max(1., abs(value.real))]
    near = [(value, vector) for value, vector in finite if abs(value) <= 1e-7]
    assert len(near) == 6, sorted(value for value, _ in finite)[:12]
    rigid_modes = np.column_stack([vector[:count] for _, vector in near])
    rigid_mass = rigid_modes.conj().T@packet.full_mass@rigid_modes
    mass_eigenvalues = np.linalg.eigvalsh(rigid_mass)
    assert mass_eigenvalues[0] > 1e-12*mass_eigenvalues[-1]
    assert min(value for value, _ in finite if value > 1e-7) > 1e-4

    assembly = owner.assembly
    points = np.vstack((assembly.coordinates,
                        assembly.beam.reference_positions))
    nodal_count = assembly.shell_count+assembly.beam.nodal_count
    rigid = _rigid_nodal_basis(points)
    analytic = np.zeros((count, 6))
    analytic[:nodal_count] = rigid
    for retained in range(packet.internal_cell_coordinates//3):
        first = nodal_count+3*retained
        analytic[first:first+3, 3:] = np.eye(3)
    assert np.linalg.norm(packet.full_stiffness@analytic)/(
        np.linalg.norm(packet.full_stiffness)*np.linalg.norm(analytic)) <= 1e-9
    assert np.linalg.norm(packet.constraints@analytic)/(
        np.linalg.norm(packet.constraints)*np.linalg.norm(analytic)) <= 1e-9
    represented = rigid_modes@np.linalg.lstsq(rigid_modes, analytic,
                                             rcond=None)[0]
    assert np.linalg.norm(analytic-represented)/np.linalg.norm(analytic) <= 1e-8
    rigid_coefficients = []
    for value, vector in near:
        displacement = vector[:count]
        nodal = displacement[:nodal_count]
        coefficients = np.linalg.lstsq(rigid, nodal, rcond=None)[0]
        fitted = rigid@coefficients
        assert np.linalg.norm(nodal-fitted)/np.linalg.norm(nodal) <= 1e-8
        rigid_coefficients.append(coefficients/np.linalg.norm(coefficients))
        assert np.linalg.norm(packet.constraints@displacement) <= 1e-8
        residual = k@vector-value*m@vector
        scale = np.linalg.norm(k@vector)+abs(value)*np.linalg.norm(m@vector)
        assert np.linalg.norm(residual) <= 1e-8*max(scale, 1.)
    assert np.linalg.matrix_rank(np.column_stack(rigid_coefficients), tol=1e-8) == 6
