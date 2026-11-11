"""Preparatory physical B2 graph covariance checks, not F09 acceptance."""

from fractions import Fraction

import numpy as np
import pytest
from scipy import sparse
from scipy.sparse.linalg import splu

from anyfesolver._b2_physical_flexibility_linear import LinearSection
from anyfesolver._b2_physical_linear_definition import PhysicalB2Member


def _members(rotation=np.eye(3), origin=np.zeros(3)):
    section = np.diag([22., 13., 17., 9., 7., 11.])
    section[0, 4] = section[4, 0] = 1.5
    section[1, 5] = section[5, 1] = -.7
    law = LinearSection.generalized(section)
    points = np.array([[0., 0., 0.], [2., 0., 0.], [2., 1.5, 0.]])
    points = points @ rotation.T + origin
    axes = np.array([[0., 1., 0.], [-1., 0., 0.]]) @ rotation.T
    return tuple(PhysicalB2Member(
        f"member-{i}", f"source-{i}", (i, i + 1),
        (points[i], points[i + 1]), axes[i], "generalized-A", law,
    ) for i in range(2)), points


def _solve(members, load):
    rows, cols, values = [], [], []
    for beam in members:
        transform = beam.local_dof_map()
        stiffness = transform.T @ beam.local_operator().evaluate(np.zeros(12)).tangent @ transform
        dofs = np.r_[np.arange(6 * beam.node_ids[0], 6 * beam.node_ids[0] + 6),
                     np.arange(6 * beam.node_ids[1], 6 * beam.node_ids[1] + 6)]
        for i, row in enumerate(dofs):
            for j, col in enumerate(dofs):
                rows.append(int(row))
                cols.append(int(col))
                values.append(stiffness[i, j])
    assembled = sparse.coo_matrix((values, (rows, cols)), shape=(18, 18)).tocsc()
    free = np.arange(6, 18)
    displacement = np.zeros(18)
    displacement[free] = splu(assembled[free, :][:, free]).solve(load[free])
    reaction = assembled @ displacement - load
    return displacement, reaction, assembled


def _exact_linear_solve(matrix, rhs):
    """Independent rational Gauss elimination for a sectional compliance."""
    n = len(rhs)
    augmented = [[Fraction(matrix[i][j]) for j in range(n)] + [Fraction(rhs[i])]
                 for i in range(n)]
    for k in range(n):
        pivot_row = next(i for i in range(k, n) if augmented[i][k])
        augmented[k], augmented[pivot_row] = augmented[pivot_row], augmented[k]
        pivot = augmented[k][k]
        augmented[k] = [v / pivot for v in augmented[k]]
        for i in range(n):
            if i != k:
                multiplier = augmented[i][k]
                augmented[i] = [a - multiplier*b for a, b in zip(augmented[i], augmented[k])]
    return [row[-1] for row in augmented]


def _force_method_energy(members, applied):
    """Exact two-segment cantilever energy from equilibrium section wrenches.

    A cut at station x carries the terminal force and its moment arm. The
    local section compliance is integrated independently of B/H and KKT.
    Simpson integration is exact because each wrench is affine in x.
    """
    vector = lambda data: tuple(Fraction(float(v)) for v in data)
    tip = vector(members[-1].reference_points_m[1])
    force, couple = vector(applied[:3]), vector(applied[3:])
    def cross(a, b):
        return (a[1]*b[2]-a[2]*b[1], a[2]*b[0]-a[0]*b[2],
                a[0]*b[1]-a[1]*b[0])
    total = Fraction(0)
    # These are the exact reference frames of the two prescribed members;
    # they are not obtained from local_frame_rows or its transform method.
    frames = (((1,0,0),(0,1,0),(0,0,1)),
              ((0,1,0),(-1,0,0),(0,0,1)))
    for beam, frame in zip(members, frames):
        start, end = (vector(p) for p in beam.reference_points_m)
        length = Fraction(float(beam.length_m))
        station_energies = []
        for t in (Fraction(0), Fraction(1,2), Fraction(1)):
            point = tuple(a + t*(b-a) for a, b in zip(start, end))
            lever = tuple(a-b for a, b in zip(tip, point))
            cross_moment = cross(lever, force)
            moment = tuple(a+b for a,b in zip(couple, cross_moment))
            wrench = tuple(sum(Fraction(axis[k])*v[k] for k in range(3))
                           for v in (force, moment) for axis in frame)
            strain = _exact_linear_solve(beam.section.stiffness, wrench)
            station_energies.append(sum(a*b for a,b in zip(wrench, strain)))
        total += length*(station_energies[0] + 4*station_energies[1]
                         + station_energies[2])/12
    return total


def test_oriented_two_member_graph_covariance_work_and_balance():
    members, points = _members()
    load = np.zeros(18)
    load[12:18] = [0.4, -0.9, 0.7, -0.2, 0.3, 0.1]
    displacement, reaction, stiffness = _solve(members, load)
    assert np.linalg.norm(reaction[6:]) < 1e-13
    np.testing.assert_allclose(reaction[:3], -load[12:15], rtol=0, atol=1e-13)
    np.testing.assert_allclose(
        reaction[3:6] + load[15:18] + np.cross(points[2] - points[0], load[12:15]),
        0, rtol=0, atol=1e-13,
    )
    energy = .5 * displacement @ (stiffness @ displacement)
    assert energy == pytest.approx(.5 * displacement @ load, rel=2e-14)
    assert energy > 0

    # Rigid rotation and translation of the reference graph must transform
    # both forces and moments without changing the elastic work.
    exact_axes = np.array([[0., 0., 1.], [1., 0., 0.], [0., 1., 0.]])
    axis = np.array([1., 2., 3.]) / np.sqrt(14.)
    cross = np.array([[0., -axis[2], axis[1]],
                      [axis[2], 0., -axis[0]],
                      [-axis[1], axis[0], 0.]])
    oblique = np.eye(3) + np.sin(.37) * cross + (1. - np.cos(.37)) * cross @ cross
    for rotation in (exact_axes, oblique):
        moved, _ = _members(rotation, np.array([4., -2., 3.]))
        block = np.kron(np.eye(6), rotation)
        moved_displacement, moved_reaction, moved_stiffness = _solve(moved, block @ load)
        np.testing.assert_allclose(moved_displacement, block @ displacement,
                                   rtol=2e-13, atol=2e-13)
        np.testing.assert_allclose(moved_reaction, block @ reaction,
                                   rtol=2e-13, atol=2e-13)
        np.testing.assert_allclose(moved_stiffness.toarray(),
                                   block @ stiffness.toarray() @ block.T,
                                   rtol=2e-13, atol=2e-13)
        assert .5 * moved_displacement @ (moved_stiffness @ moved_displacement) == pytest.approx(
            energy, rel=2e-13)


def test_generalized_elbow_energy_against_independent_exact_force_method():
    members, _ = _members()
    terminal_cases = [np.eye(6)[i] for i in range(6)]
    terminal_cases.append(np.array([.5, -1., .75, -.25, .375, .125]))
    for terminal in terminal_cases:
        load = np.zeros(18)
        load[12:18] = terminal
        displacement, _, stiffness = _solve(members, load)
        numerical = .5 * displacement @ (stiffness @ displacement)
        reference = _force_method_energy(members, terminal)
        assert numerical == pytest.approx(float(reference), rel=3e-14)
