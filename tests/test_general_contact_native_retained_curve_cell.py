"""Exact retained B3-GE half-cell reconstruction and energy pullback."""

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from anyfesolver.general_contact.native_retained_curve_cell import (
    NativeRetainedCurveCell,
)
from test_general_contact_native_retained_station import _reference


def _pose():
    reference = _reference(True)
    high = reference.coordinates.copy()
    high[1] += (0.02, -0.01, 0.005)
    low = np.zeros((3, 3))
    rotations = np.stack((
        Rotation.from_rotvec((0.13, -0.04, 0.07)).as_matrix(),
        Rotation.from_rotvec((-0.09, 0.06, -0.03)).as_matrix(),
    ))
    return reference, high, low, rotations


@pytest.mark.parametrize("cell_index", (0, 1))
def test_quadratic_cell_reconstructs_native_centerline(cell_index):
    reference, high, low, rotations = _pose()
    cell = NativeRetainedCurveCell(7, reference, cell_index)
    state = cell.evaluate(high, low, rotations)
    assert state.positions.shape == (3, 3)
    assert state.geometric_jacobian.shape == (9, 42)
    for t in (0.0, 0.13, 0.37, 0.5, 0.81, 1.0):
        shape = np.array((2*t*t-3*t+1, 4*t-4*t*t, 2*t*t-t))
        actual = shape @ state.positions
        expected = ((1-t)*(high[cell_index]+low[cell_index])
                    + t*(high[cell_index+1]+low[cell_index+1])
                    + rotations[cell_index]
                    @ reference.half_cell_lift(cell_index, t))
        np.testing.assert_allclose(actual, expected, atol=3e-15)
    moved = rotations.copy()
    moved[cell_index] = Rotation.from_rotvec((0.03, -0.04, 0.02)).as_matrix() @ moved[cell_index]
    changed = cell.evaluate(high, low, moved)
    assert np.linalg.norm(changed.positions[1]-state.positions[1]) > 1e-5
    np.testing.assert_array_equal(state.geometric_jacobian[:, 24:], 0.0)
    for node in range(3):
        np.testing.assert_array_equal(
            state.geometric_jacobian[:, 6*node+3:6*node+6], 0.0,
        )


def test_curve_position_energy_pullback_matches_virtual_work_and_curvature():
    reference, high, low, rotations = _pose()
    cell = NativeRetainedCurveCell(7, reference, 0)
    state = cell.evaluate(high, low, rotations)
    hessian = np.diag(np.linspace(1.0, 2.0, 9))
    hessian[3:6, 0:3] = 0.1*np.eye(3)
    hessian[0:3, 3:6] = 0.1*np.eye(3)
    linear = np.linspace(-0.2, 0.3, 9)
    points = state.positions.reshape(9)
    force = hessian@points+linear
    gradient, tangent = state.pullback(force, hessian)
    direction = np.zeros(42)
    direction[0:3] = (0.13, -0.08, 0.04)
    direction[6:9] = (-0.03, 0.12, -0.07)
    direction[18:21] = (0.1, -0.12, 0.09)

    def energy(fraction):
        moved = high.copy()
        spun = rotations.copy()
        for node in range(3):
            moved[node] += fraction*direction[6*node:6*node+3]
        spun[0] = (Rotation.from_rotvec(fraction*direction[18:21])
                   .as_matrix() @ rotations[0])
        value = cell.evaluate(moved, low, spun).positions.reshape(9)
        return 0.5*value@hessian@value+linear@value

    step = 1e-4
    first = (energy(step)-energy(-step))/(2*step)
    second = (energy(step)-2*energy(0.0)+energy(-step))/step**2
    assert abs(first-gradient@direction) < 2e-8
    assert abs(second-direction@tangent@direction) < 3e-6
    assert np.linalg.norm(gradient[18:21]) > 0.0
    np.testing.assert_array_equal(gradient[24:], 0.0)
    np.testing.assert_array_equal(tangent[24:], 0.0)
    np.testing.assert_array_equal(tangent[:, 24:], 0.0)


def test_curve_cell_rigid_covariance_and_foreign_reference_rejection():
    reference, high, low, rotations = _pose()
    cell = NativeRetainedCurveCell(7, reference, 1)
    original = cell.evaluate(high, low, rotations)
    turn = Rotation.from_rotvec((0.2, 0.1, -0.14)).as_matrix()
    shift = np.array((0.4, -0.3, 0.2))
    moved = cell.evaluate(high@turn.T+shift, low@turn.T,
                          np.einsum("ij,cjk->cik", turn, rotations))
    np.testing.assert_allclose(moved.positions,
                               original.positions@turn.T+shift, atol=2e-15)
    cell.reference = _reference(False)
    with pytest.raises(ValueError, match="reference identity changed"):
        cell.evaluate(high, low, rotations)
