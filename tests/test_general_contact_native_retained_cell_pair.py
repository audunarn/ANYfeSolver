"""Distributed selected-cell native pair work and fail-closed ownership."""

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from anyfesolver._ge_beam3_centered_reference import (
    CenteredCurvedBeam3ReferenceGeometry,
)
from anyfesolver.general_contact.native_retained_cell_pair import (
    NativeRetainedSelectedCellPair,
)


def _references():
    a = np.array(((0.0, 0.0, 0.0),
                  (1.0, 0.1, 0.0),
                  (2.0, 0.0, 0.0)))
    b = np.array(((0.5, -1.5, 0.08),
                  (0.5, -0.5, 0.08),
                  (0.5, 0.5, 0.08)))
    frames_a = []
    for slope in (0.2, 0.0, -0.2):
        tangent = np.array((1.0, slope, 0.0))
        tangent /= np.linalg.norm(tangent)
        normal = np.array((0.0, 0.0, 1.0))
        frames_a.append(np.column_stack((tangent, normal,
                                         np.cross(tangent, normal))))
    tangent_b = np.array((0.0, 1.0, 0.0))
    normal_b = np.array((0.0, 0.0, 1.0))
    frames_b = np.tile(np.column_stack((tangent_b, normal_b,
                                        np.cross(tangent_b, normal_b))),
                       (3, 1, 1))
    return (CenteredCurvedBeam3ReferenceGeometry(a, np.array(frames_a)),
            CenteredCurvedBeam3ReferenceGeometry(b, frames_b))


def _kernel(*, maximum_gap_cells=10000):
    a, b = _references()
    return NativeRetainedSelectedCellPair(
        11, a, 0, 0.05, 12, b, 1, 0.05,
        line_stiffness_n_per_m2=100.0,
        integration_relative_tolerance=1e-5,
        maximum_subdivisions=10,
        gap_accuracy_m=1e-4,
        maximum_gap_cells=maximum_gap_cells,
        maximum_gap_depth=28,
    ), a, b


def _pose(reference):
    return (reference.coordinates.copy(), np.zeros((3, 3)),
            np.tile(np.eye(3), (2, 1, 1)))


def test_native_selected_pair_has_retained_work_cross_blocks_and_wrench():
    kernel, a, b = _kernel()
    pa, la, ra = _pose(a)
    pb, lb, rb = _pose(b)
    result = kernel.evaluate(pa, la, ra, pb, lb, rb)
    assert result.potential_j > 0.0
    assert result.conservative_lower_gap_m < 0.0
    assert result.integration_points > 0
    assert np.linalg.norm(result.residual[18:21]) > 0.0
    assert np.linalg.norm(result.tangent[:42, 42:]) > 0.0
    np.testing.assert_allclose(result.beam_a_force_n+result.beam_b_force_n,
                               0.0, atol=1e-8)
    np.testing.assert_allclose(result.beam_a_moment_nm+result.beam_b_moment_nm,
                               0.0, atol=1e-8)
    np.testing.assert_array_equal(result.residual[24:42], 0.0)
    np.testing.assert_array_equal(result.residual[66:84], 0.0)


def test_native_pair_energy_virtual_work_and_second_directional_derivative():
    kernel, a, b = _kernel()
    pa, la, ra = _pose(a)
    pb, lb, rb = _pose(b)
    result = kernel.evaluate(pa, la, ra, pb, lb, rb)
    direction = np.zeros(84)
    direction[0:3] = (0.003, -0.002, 0.001)
    direction[6:9] = (-0.004, 0.002, -0.003)
    direction[18:21] = (0.02, -0.01, 0.015)
    direction[42:45] = (-0.002, 0.001, 0.003)
    direction[42+12:42+15] = (0.002, -0.002, -0.001)

    def energy(fraction):
        moved_a, moved_b = pa.copy(), pb.copy()
        for node in range(3):
            moved_a[node] += fraction*direction[6*node:6*node+3]
            moved_b[node] += fraction*direction[42+6*node:42+6*node+3]
        spun_a = ra.copy()
        spun_a[0] = (Rotation.from_rotvec(fraction*direction[18:21])
                     .as_matrix() @ ra[0])
        return kernel.evaluate(moved_a, la, spun_a,
                               moved_b, lb, rb).potential_j

    step = 2e-5
    first = (energy(step)-energy(-step))/(2*step)
    second = (energy(step)-2*energy(0.0)+energy(-step))/step**2
    assert first == pytest.approx(result.residual@direction,
                                  rel=1e-5, abs=1e-7)
    assert second == pytest.approx(direction@result.tangent@direction,
                                   rel=1e-5, abs=1e-6)


def test_native_selected_pair_is_rigidly_covariant():
    kernel, a, b = _kernel()
    pa, la, ra = _pose(a)
    pb, lb, rb = _pose(b)
    base = kernel.evaluate(pa, la, ra, pb, lb, rb)
    turn = Rotation.from_rotvec((0.2, -0.1, 0.15)).as_matrix()
    shift = np.array((0.7, -0.4, 0.2))
    rotated = np.tile(turn, (2, 1, 1))
    moved = kernel.evaluate(pa@turn.T+shift, la@turn.T, rotated,
                            pb@turn.T+shift, lb@turn.T, rotated)
    assert moved.potential_j == pytest.approx(base.potential_j, rel=1e-11)
    np.testing.assert_allclose(moved.beam_a_force_n,
                               turn@base.beam_a_force_n, atol=1e-8)
    np.testing.assert_allclose(moved.beam_b_force_n,
                               turn@base.beam_b_force_n, atol=1e-8)


def test_undeclared_cell_pair_and_search_budget_fail_closed():
    kernel, a, b = _kernel()
    pa, la, ra = _pose(a)
    pb, lb, rb = _pose(b)
    moved_b = pb+np.array((1.0, 0.0, 0.0))
    with pytest.raises(ValueError, match="undeclared native half-cell pair"):
        kernel.evaluate(pa, la, ra, moved_b, lb, rb)
    bounded, _, _ = _kernel(maximum_gap_cells=1)
    with pytest.raises(MemoryError, match="gap cell budget"):
        bounded.evaluate(pa, la, ra, pb, lb, rb)
    kernel.law.stiffness *= 2.0
    with pytest.raises(ValueError, match="definition changed"):
        kernel.evaluate(pa, la, ra, pb, lb, rb)
