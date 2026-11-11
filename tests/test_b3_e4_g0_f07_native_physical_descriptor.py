"""F07 development check of physical coupled inertia and algebraic coordinates.

The supported straight Q4 owner also supplies full reference operators for a
free-free descriptor oracle. Neither check admits a coupled modal range or
changes the G0/G1 qualification status.
"""

import numpy as np
import pytest
from scipy import linalg

from anyfesolver._ge_beam3_coupled_modes import capture, modes
from test_ge_beam3_coupled_modes import arguments, fixture


def test_native_beam_shell_mass_survives_exact_joint_and_descriptor():
    owner = fixture("Q4", curved=False)
    solved = owner.solve(stop_after=0)
    assert solved.status == "paused", solved.failure
    state = solved.state
    inertias = arguments(owner)
    packet = capture(owner, state, **inertias)
    assembly = owner.assembly

    # An independent rigid-translation kinetic-work oracle: unit shell area,
    # thickness 0.2, density 2; beam length 2, sectional mass/length 2.
    rigid_x = np.zeros(len(packet.full_kinematic_dofs))
    for node in range(assembly.shell_count // 6):
        rigid_x[6 * node] = 1.
    for node in range(assembly.beam.nodal_count // 6):
        rigid_x[assembly.shell_count + 6 * node] = 1.
    np.testing.assert_allclose(packet.constraints @ rigid_x, 0., atol=1e-13)
    np.testing.assert_allclose(rigid_x @ packet.full_mass @ rigid_x, 4.4,
                               rtol=1e-12, atol=1e-12)
    assert packet.internal_cell_coordinates == 6
    assert packet.shell_mass_rank == 24

    # The retained physical cell has inertia, but geometric nodal rotations
    # and the six exact-joint multipliers have none.  No mass is patched into
    # either algebraic block to make a generalized eigenproblem nonsingular.
    cell = np.arange(assembly.shell_count + assembly.beam.nodal_count,
                     len(rigid_x))
    assert np.linalg.norm(packet.full_mass[np.ix_(cell, cell)]) > 0.
    assert packet.algebraic_dofs
    reduced_mass = packet.lift.T @ packet.full_mass @ packet.lift
    np.testing.assert_allclose(packet.kinetic.T @ packet.kinetic,
                               reduced_mass, rtol=1e-11, atol=1e-11)
    np.testing.assert_array_equal(packet.kinetic[:, packet.algebraic_dofs], 0.)
    np.testing.assert_allclose(reduced_mass[:, packet.algebraic_dofs], 0.,
                               atol=1e-13)

    free = [i for i, dof in enumerate(packet.full_kinematic_dofs)
            if dof in owner.free]
    k = packet.full_stiffness[np.ix_(free, free)]
    m = packet.full_mass[np.ix_(free, free)]
    c = packet.constraints[:, free]
    saddle = np.block([[k, c.T], [c, np.zeros((6, 6))]])
    descriptor_mass = linalg.block_diag(m, np.zeros((6, 6)))
    np.testing.assert_array_equal(descriptor_mass[-6:, :], 0.)
    np.testing.assert_array_equal(descriptor_mass[:, -6:], 0.)
    eigenvalues = linalg.eigvals(saddle, descriptor_mass)
    finite = np.sort(eigenvalues[np.isfinite(eigenvalues)].real)
    assert np.all(np.abs(eigenvalues[np.isfinite(eigenvalues)].imag) < 1e-8)
    # For this supported fixture, 24 physical finite directions remain;
    # six algebraic retained directions plus six exact constraints and six
    # multiplier rows account for the 18 infinite descriptor roots.
    assert len(finite) == 24
    assert np.count_nonzero(~np.isfinite(eigenvalues)) == 18

    _, actual = modes(owner, state, **inertias, bounds=(-100., 10000.),
                      num_modes=3)
    np.testing.assert_allclose(actual.eigenvalues, finite[:3],
                               rtol=1e-10, atol=1e-10)


@pytest.mark.parametrize("topology", ("Q4", "S3-V2D"))
def test_connected_free_free_reference_has_six_physical_rigid_modes(topology):
    # The bounded legacy owner rightly refuses an unsupported static solve.
    # Inspect the *full* rest operators before applying its support mask.
    owner = fixture(topology, curved=False)
    assembly = owner.assembly
    solved = owner.solve(stop_after=0)
    assert solved.status == "paused", solved.failure
    packet = capture(owner, solved.state, **arguments(owner))
    sh = assembly.shell_count
    bn = assembly.beam.nodal_count
    rigid = np.zeros((len(packet.full_kinematic_dofs), 6))
    shell_xyz = assembly.coordinates
    beam_xyz = solved.state.mechanical.positions
    for first, points in ((0, shell_xyz), (sh, beam_xyz)):
        for index, xyz in enumerate(points):
            rows = first + 6 * index
            rigid[rows:rows+3, :3] = np.eye(3)
            rigid[rows:rows+3, 3:] = -np.array(
                ((0., -xyz[2], xyz[1]),
                 (xyz[2], 0., -xyz[0]),
                 (-xyz[1], xyz[0], 0.)))
            rigid[rows+3:rows+6, 3:] = np.eye(3)
    for start in range(sh+bn, len(rigid), 6):
        rigid[start:start+3, 3:] = np.eye(3)
        rigid[start+3:start+6, 3:] = np.eye(3)

    np.testing.assert_allclose(packet.constraints @ rigid, 0., atol=1e-11)
    mass = packet.full_mass
    stiffness = packet.full_stiffness
    assert np.linalg.matrix_rank(rigid.T @ mass @ rigid, tol=1e-9) == 6
    scale = max(1., np.linalg.norm(stiffness) * np.linalg.norm(rigid))
    assert np.linalg.norm(stiffness @ rigid) / scale < 1e-10

    saddle = np.block([[stiffness, packet.constraints.T],
                       [packet.constraints, np.zeros((6, 6))]])
    descriptor_mass = linalg.block_diag(mass, np.zeros((6, 6)))
    values = linalg.eigvals(saddle, descriptor_mass)
    finite = values[np.isfinite(values)]
    assert np.max(np.abs(finite.imag)) < 1e-7
    if topology == "Q4":
        assert len(finite) == 36
        assert len(values) == 54
        assert np.count_nonzero(~np.isfinite(values)) == 18
    else:
        assert len(finite) == 23
        assert len(values) == 48
        assert np.count_nonzero(~np.isfinite(values)) == 25
    zero = np.sort(np.abs(finite.real))[:7]
    assert np.max(zero[:6]) < 1e-7
    assert zero[6] > 1e-5
