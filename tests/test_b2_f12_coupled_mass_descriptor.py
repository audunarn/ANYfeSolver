"""Bounded B2/Q4 point-mass and massless-multiplier descriptor checks."""

from dataclasses import replace

import numpy as np
import pytest
from scipy import linalg

from anyfesolver._b2_physical_linear_definition import MassMetadata
from anyfesolver._stiffened_shell_physical_q4_mass import physical_mass
from test_b2_f10_f11_point_attachment import VERTICES, specimen


def _section_mass():
    # Four physical cross-section particles, 2 kg/m each, referred to an
    # offset velocity origin. The beam axis is y=z=0.
    positions = np.array([(0., .25+y, -.125+z)
                          for y in (-.3, .3) for z in (-.4, .4)])
    origin = np.array((0., .1, .05))
    inertia = np.zeros((6, 6))
    for p in positions-origin:
        x, y, z = p
        velocity_map = np.array([[1.,0.,0.,0.,z,-y],
                                 [0.,1.,0.,-z,0.,x],
                                 [0.,0.,1.,y,-x,0.]])
        inertia += 2.*velocity_map.T@velocity_map
    return MassMetadata("four-particles", tuple(origin), inertia), positions


def _shell_mass():
    # The old fixture reused the inherited mass with drilling inertia. This
    # candidate uses the independently kinetic-checked director plate law.
    return physical_mass(VERTICES, density_kg_m3=2., thickness_m=.15)


def test_candidate_b2_mass_reproduces_independent_rigid_particle_work():
    member, _, _, _ = specimen()
    metadata, positions = _section_mass()
    member = replace(member, mass=metadata)
    matrix = member.local_candidate_mass_matrix()
    assert matrix.shape == (12, 12)
    np.testing.assert_allclose(matrix, matrix.T, rtol=2e-14, atol=2e-14)
    v = np.array((.13, -.21, .31))
    omega = np.array((-.17, .29, .11))
    length = member.length_m
    q = np.r_[v, omega, v+np.cross(omega, [length, 0., 0.]), omega]
    # Independent physical particles integrated with a two-point rule;
    # rigid velocity is affine along the axis, so the energy is exact.
    line = (.5-.5/np.sqrt(3.), .5+.5/np.sqrt(3.))
    direct = sum(length*.5*2.*np.linalg.norm(
        v+np.cross(omega, [s*length, *p[1:]]))**2
        for s in line for p in positions)
    assert q@matrix@q == pytest.approx(direct, rel=3e-12)
    matrix[0, 0] = 0.
    assert member.local_candidate_mass_matrix()[0, 0] > 0.
    with pytest.raises(ValueError, match="mass metadata"):
        replace(member, mass=None).local_candidate_mass_matrix()


def test_oblique_member_mass_preserves_particle_work_and_reversal():
    member, _, _, _ = specimen()
    p0 = np.array((.4, -.2, .5))
    p1 = np.array((1.6, .7, 1.3))
    axis = (p1-p0)/np.linalg.norm(p1-p0)
    local_y = np.array((0., 0., 1.))-axis[2]*axis
    local_y /= np.linalg.norm(local_y)
    metadata, particles = _section_mass()
    forward = replace(member, reference_points_m=(tuple(p0), tuple(p1)),
                      local_y_global=tuple(local_y), mass=metadata)
    forward_frame = forward.local_frame_rows()
    forward_map = forward.local_dof_map()
    forward_mass = forward_map.T@forward.local_candidate_mass_matrix()@forward_map

    velocity = np.array((.13, -.21, .31))
    angular = np.array((-.17, .29, .11))
    nodal = np.r_[velocity, angular,
                  velocity+np.cross(angular, p1-p0), angular]
    stations = (.5-.5/np.sqrt(3.), .5+.5/np.sqrt(3.))
    direct = sum(np.linalg.norm(velocity+np.cross(angular,
                 station*(p1-p0)+forward_frame.T@particle))**2
                 for station in stations for particle in particles)
    direct *= forward.length_m
    assert nodal@forward_mass@nodal == pytest.approx(direct, rel=3e-12)

    reverse = replace(forward, reference_points_m=(tuple(p1), tuple(p0)))
    reverse_frame = reverse.local_frame_rows()
    change = reverse_frame@forward_frame.T
    reverse_origin = change@np.asarray(metadata.origin_local_m)
    reverse_particles = particles@change.T
    reverse_inertia = np.zeros((6, 6))
    for particle in reverse_particles-reverse_origin:
        x, y, z = particle
        velocity_map = np.array(((1.,0.,0.,0.,z,-y),
                                 (0.,1.,0.,-z,0.,x),
                                 (0.,0.,1.,y,-x,0.)))
        reverse_inertia += 2.*velocity_map.T@velocity_map
    reverse = replace(reverse, mass=MassMetadata(
        "four-particles/reversed", tuple(reverse_origin), reverse_inertia))
    reverse_map = reverse.local_dof_map()
    reverse_mass = reverse_map.T@reverse.local_candidate_mass_matrix()@reverse_map
    swap = np.kron(np.array(((0., 1.), (1., 0.))), np.eye(6))
    np.testing.assert_allclose(reverse_mass, swap@forward_mass@swap,
                               rtol=3e-12, atol=3e-12)


def test_physical_b2_q4_point_descriptor_matches_independent_elimination():
    member, law, shell_k, beam_k = specimen()
    metadata, _ = _section_mass()
    member = replace(member, mass=metadata)
    transform = member.local_dof_map()
    beam_mass = transform.T@member.local_candidate_mass_matrix()@transform
    stiffness = linalg.block_diag(shell_k, beam_k)
    physical_mass = linalg.block_diag(_shell_mass(), beam_mass)
    fixed = set(range(6)) | set(range(18, 24)) | set(range(24, 30))
    fixed.update(6*i+j for i in (1, 2) for j in (3, 4, 5))
    free = np.array([i for i in range(36) if i not in fixed])
    k = stiffness[np.ix_(free, free)]
    m = physical_mass[np.ix_(free, free)]
    c = law.constraint[:, free]
    assert np.linalg.matrix_rank(c) == 6
    saddle = np.block([[k, c.T], [c, np.zeros((6, 6))]])
    descriptor = linalg.block_diag(m, np.zeros((6, 6)))
    np.testing.assert_array_equal(descriptor[-6:, :], 0.)
    np.testing.assert_array_equal(descriptor[:, -6:], 0.)
    eigenvalues, vectors = linalg.eig(saddle, descriptor)
    finite = np.isfinite(eigenvalues)
    assert finite.sum() == len(free)-6
    assert (~finite).sum() == 12
    assert max(abs(eigenvalues[finite].imag)) < 1e-8
    for lam, vector in zip(eigenvalues[finite], vectors[:, finite].T):
        residual = np.linalg.norm(saddle@vector-lam*descriptor@vector)
        scale = max(np.linalg.norm(saddle@vector),
                    abs(lam)*np.linalg.norm(descriptor@vector))
        assert residual/scale < 2e-10
        assert np.linalg.norm(c@vector[:len(free)]) < 2e-10
    null = linalg.null_space(c)
    reduced = linalg.eigh(null.T@k@null, null.T@m@null,
                          eigvals_only=True)
    np.testing.assert_allclose(np.sort(eigenvalues[finite].real),
                               reduced, rtol=2e-9, atol=2e-9)


def test_physical_b2_q4_point_free_free_rest_has_six_rigid_modes():
    member, law, shell_k, beam_k = specimen()
    metadata, _ = _section_mass()
    member = replace(member, mass=metadata)
    transform = member.local_dof_map()
    stiffness = linalg.block_diag(shell_k, beam_k)
    physical_mass = linalg.block_diag(
        _shell_mass(),
        transform.T@member.local_candidate_mass_matrix()@transform)
    constraint = law.constraint
    assert np.linalg.matrix_rank(constraint) == 6

    points = np.vstack((VERTICES, member.reference_points_m))
    center = points.mean(axis=0)
    rigid = np.zeros((36, 6))
    for i, point in enumerate(points):
        lever = point-center
        for j in range(3):
            axis = np.eye(3)[j]
            rigid[6*i:6*i+3, j] = axis
            rigid[6*i:6*i+3, j+3] = np.cross(axis, lever)
            rigid[6*i+3:6*i+6, j+3] = axis
    assert np.linalg.norm(stiffness@rigid)/(
        np.linalg.norm(stiffness)*np.linalg.norm(rigid)) <= 1e-9
    assert np.linalg.norm(constraint@rigid)/(
        np.linalg.norm(constraint)*np.linalg.norm(rigid)) <= 1e-9
    rigid_mass = rigid.T@physical_mass@rigid
    assert np.linalg.eigvalsh(rigid_mass)[0] > 1e-10

    saddle = np.block([[stiffness, constraint.T],
                       [constraint, np.zeros((6, 6))]])
    descriptor = linalg.block_diag(physical_mass, np.zeros((6, 6)))
    np.testing.assert_array_equal(descriptor[-6:], 0.)
    eigenvalues, eigenvectors = linalg.eig(saddle, descriptor)
    assert not np.isnan(eigenvalues).any()
    null = linalg.null_space(constraint)
    expected_finite = np.linalg.matrix_rank(
        null.T@physical_mass@null, tol=1e-10)
    finite = np.isfinite(eigenvalues)
    assert finite.sum() == expected_finite
    near = np.flatnonzero(finite & (abs(eigenvalues) <= 1e-7))
    assert len(near) == 6
    near_primal = eigenvectors[:36, near]
    represented = near_primal@np.linalg.lstsq(near_primal, rigid,
                                             rcond=None)[0]
    assert np.linalg.norm(rigid-represented)/np.linalg.norm(rigid) <= 1e-8
    for index in near:
        lam = eigenvalues[index]
        vector = eigenvectors[:, index]
        residual = saddle@vector-lam*descriptor@vector
        scale = np.linalg.norm(saddle@vector)+abs(lam)*np.linalg.norm(descriptor@vector)
        assert np.linalg.norm(residual)/max(scale, 1.) <= 1e-8
        assert np.linalg.norm(constraint@vector[:36]) <= 1e-8


def test_b2_q4_point_prestress_descriptor_keeps_mass_and_curvature_distinct():
    member, law, shell_k, beam_k = specimen()
    metadata, _ = _section_mass()
    member = replace(member, mass=metadata)
    frame = member.local_dof_map()
    physical_mass = linalg.block_diag(
        _shell_mass(), frame.T@member.local_candidate_mass_matrix()@frame)
    axial = member.compile_local().global_candidate_constant_axial_geometric_stiffness(-.05)
    multiplier = np.array((.04, -.03, .02, .005, -.004, .006))
    connection = law.candidate_prestress_curvature(multiplier)
    geometric = np.zeros((36, 36))
    geometric[24:, 24:] = axial
    stiffness = linalg.block_diag(shell_k, beam_k)+geometric+connection

    fixed = set(range(6)) | set(range(18, 24)) | set(range(24, 30))
    fixed.update(6*i+j for i in (1, 2) for j in (3, 4, 5))
    free = np.array([i for i in range(36) if i not in fixed])
    c = law.constraint[:, free]
    null = linalg.null_space(c)
    assert np.linalg.matrix_rank(c) == 6
    assert np.linalg.norm(null.T@geometric[np.ix_(free, free)]@null) > 1e-8
    assert np.linalg.norm(null.T@connection[np.ix_(free, free)]@null) > 1e-8

    k = stiffness[np.ix_(free, free)]
    m = physical_mass[np.ix_(free, free)]
    saddle = np.block([[k, c.T], [c, np.zeros((6, 6))]])
    descriptor = linalg.block_diag(m, np.zeros((6, 6)))
    np.testing.assert_array_equal(descriptor[-6:, :], 0.)
    np.testing.assert_array_equal(descriptor[:, -6:], 0.)
    eigenvalues, eigenvectors = linalg.eig(saddle, descriptor)
    assert not np.isnan(eigenvalues).any()
    finite = np.isfinite(eigenvalues)
    projected_mass = null.T@m@null
    assert np.linalg.eigvalsh(projected_mass)[0] > 0.
    assert finite.sum() == len(free)-6
    assert max(abs(eigenvalues[finite].imag)) <= 1e-8
    for value, vector in zip(eigenvalues[finite], eigenvectors[:, finite].T):
        residual = saddle@vector-value*descriptor@vector
        scale = max(np.linalg.norm(saddle@vector),
                    abs(value)*np.linalg.norm(descriptor@vector), 1.)
        assert np.linalg.norm(residual)/scale <= 1e-8
        assert np.linalg.norm(c@vector[:len(free)]) <= 1e-8
    projected = linalg.eigh(null.T@k@null, projected_mass,
                            eigvals_only=True)
    np.testing.assert_allclose(np.sort(eigenvalues[finite].real), projected,
                               rtol=2e-8, atol=2e-8)
