"""Independent affine corner-gap control for Q4 pressure against two S3 faces."""

import numpy as np
import pytest

from anyfesolver.general_contact.overlap import PlanarFace
from anyfesolver.general_contact.patch_mortar import MovingPatchMortar, PatchFace


_PENETRATION = 1.0e-4  # m
_SLOPE = 1.0e-3  # m/m in each in-plane direction
_STIFFNESS = 1.0e6  # Pa/m
_POINTWISE_FORCE = _STIFFNESS * _PENETRATION**3 / (6.0 * _SLOPE**2)


def _square_q4_vs_two_s3(divisions_or_coordinates):
    coordinates = (np.linspace(0.0, 1.0, divisions_or_coordinates + 1)
                   if isinstance(divisions_or_coordinates, int)
                   else np.asarray(divisions_or_coordinates, dtype=float))
    divisions = len(coordinates) - 1
    positions = {}

    def node(i, j):
        return 1 + j*(divisions + 1) + i

    for j in range(divisions + 1):
        for i in range(divisions + 1):
            positions[node(i, j)] = (float(coordinates[i]),
                                     float(coordinates[j]))
    source = []
    for j in range(divisions):
        for i in range(divisions):
            ids = (node(i, j), node(i + 1, j),
                   node(i + 1, j + 1), node(i, j + 1))
            corners = np.array([(*positions[index], 0.0) for index in ids])
            source.append(PatchFace(
                PlanarFace((1, len(source) + 1), corners), ids,
            ))
    target = (
        PatchFace(PlanarFace((2, 1), np.array((
            (0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (1.0, 1.0, 0.0),
        ))), (10_001, 10_002, 10_003)),
        PatchFace(PlanarFace((2, 2), np.array((
            (0.0, 0.0, 0.0), (1.0, 1.0, 0.0), (0.0, 1.0, 0.0),
        ))), (10_001, 10_003, 10_004)),
    )
    kernel = MovingPatchMortar(
        tuple(source), target, np.array((0.0, 0.0, 1.0)),
        _STIFFNESS, maximum_normal_dofs=500,
    )
    heights = np.array([
        -_PENETRATION + _SLOPE*sum(positions[index])
        for index in kernel.source_node_ids
    ])
    state = kernel.evaluate(
        np.zeros(3), np.zeros(3), heights,
        np.zeros(len(kernel.target_node_ids)),
    )
    return state, kernel, positions, kernel.source_index[node(0, 0)]


def test_q4_pressure_refinement_recovers_a_weak_open_s3_corner():
    results = {}
    for divisions in (1, 4, 8, 12, 16):
        state, _kernel, _positions, corner = _square_q4_vs_two_s3(divisions)
        expected_corner_gap = (-_PENETRATION
                               + 2.0*_SLOPE/(3.0*divisions))
        assert state.normalized_gap[corner] == pytest.approx(
            expected_corner_gap, abs=1.0e-15,
        )
        assert state.pressure_rank == (divisions + 1)**2
        assert state.overlap_area == pytest.approx(1.0, abs=1.0e-12)
        force = -float(np.sum(
            state.normal_residual[:len(state.source_node_ids)],
        ))
        results[divisions] = (force, int(np.count_nonzero(state.normalized_gap < 0)))

    # The physical corner gap is -0.1 mm at every resolution, above the
    # prospective 0.05 mm gap budget. Full pressure-gradient rank alone does
    # not make the coarse weak-open state mechanically adequate.
    assert results[1] == (0.0, 0)
    assert results[4] == (0.0, 0)
    assert results[8][1] > 0
    assert 0.0 < results[8][0] < results[12][0] < results[16][0]
    assert abs(results[16][0] - _POINTWISE_FORCE) < abs(
        results[8][0] - _POINTWISE_FORCE,
    )


def test_graded_q4_pressure_matches_independent_gap_integrals_and_wrench():
    coordinates = np.r_[np.linspace(0.0, 0.1, 17), 0.2, 1.0]
    state, kernel, positions, _corner = _square_q4_vs_two_s3(coordinates)
    assert len(kernel.source_node_ids) == 361
    assert state.pressure_rank == 361
    assert state.pressure_gradient_condition is not None

    # Independent tensor-Q1 cell integrals, without production overlap or
    # quadrature routines. A local corner shape has support area/4 and
    # weighted means x0+h/3 or x0+2h/3 in each coordinate.
    support = np.zeros(len(kernel.source_node_ids))
    weighted_gap = np.zeros_like(support)
    gram = np.zeros((len(support), len(support)))
    local_side = ((0, 0), (1, 0), (1, 1), (0, 1))
    for face in kernel.source_faces:
        x0, y0 = positions[face.node_ids[0]]
        x1, y1 = positions[face.node_ids[2]]
        area_over_four = (x1 - x0)*(y1 - y0)/4.0
        for node, (right, upper) in zip(face.node_ids, local_side):
            index = kernel.source_index[node]
            x_mean = x0 + (2.0 if right else 1.0)*(x1 - x0)/3.0
            y_mean = y0 + (2.0 if upper else 1.0)*(y1 - y0)/3.0
            support[index] += area_over_four
            weighted_gap[index] += area_over_four*(
                -_PENETRATION + _SLOPE*(x_mean + y_mean)
            )
        for first, (x_first, y_first) in zip(face.node_ids, local_side):
            row = kernel.source_index[first]
            for second, (x_second, y_second) in zip(
                face.node_ids, local_side,
            ):
                column = kernel.source_index[second]
                x_mass = 1.0/3.0 if x_first == x_second else 1.0/6.0
                y_mass = 1.0/3.0 if y_first == y_second else 1.0/6.0
                gram[row, column] += (x1 - x0)*(y1 - y0)*x_mass*y_mass
    np.testing.assert_allclose(state.support, support, rtol=1.0e-10,
                               atol=1.0e-16)
    np.testing.assert_allclose(state.weighted_gap, weighted_gap,
                               rtol=1.0e-10, atol=1.0e-16)

    reactions = -state.normal_residual[:len(kernel.source_node_ids)]
    independent_pressure = _STIFFNESS*np.maximum(-weighted_gap/support, 0.0)
    np.testing.assert_allclose(reactions, gram @ independent_pressure,
                               rtol=1.0e-10, atol=1.0e-12)
    assert float(np.linalg.eigvalsh(gram)[0]) > 0.0
    force = float(np.sum(reactions))
    moment_x = sum(positions[node][1]*reaction for node, reaction in
                   zip(kernel.source_node_ids, reactions))
    moment_y = -sum(positions[node][0]*reaction for node, reaction in
                    zip(kernel.source_node_ids, reactions))
    pointwise_moment = (_STIFFNESS*_PENETRATION**4
                        / (24.0*_SLOPE**3))
    assert abs(force - _POINTWISE_FORCE) / _POINTWISE_FORCE <= 0.02
    assert abs(moment_x - pointwise_moment) / pointwise_moment <= 0.02
    assert abs(moment_y + pointwise_moment) / pointwise_moment <= 0.02
