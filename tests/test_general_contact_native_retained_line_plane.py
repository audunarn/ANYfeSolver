"""Native retained-coordinate frictionless line work against a plane."""

import numpy as np
import pytest
from scipy.integrate import quad
from scipy.spatial.transform import Rotation

from anyfesolver.general_contact.native_retained_line_plane import (
    NativeRetainedRoundLinePlane,
)
from test_general_contact_native_retained_station import _reference


def _kernel(reference, *, origin=(0.0, 0.0, 0.0), normal=(0.0, 0.0, 1.0),
            stiffness=100.0, energy_tolerance=1e-10,
            force_tolerance=1e-9, tangent_tolerance=1e-8,
            maximum_intervals=64):
    return NativeRetainedRoundLinePlane(
        3, reference, radius_m=0.1,
        plane_origin_m=np.asarray(origin), plane_normal=np.asarray(normal),
        line_stiffness_n_per_m2=stiffness,
        energy_tolerance_j=energy_tolerance,
        force_tolerance_n=force_tolerance,
        tangent_tolerance_n_per_m=tangent_tolerance,
        maximum_intervals=maximum_intervals,
    )


def test_native_line_plane_straight_full_and_partial_analytic_response():
    reference = _reference(False)
    kernel = _kernel(reference)
    positions = reference.coordinates.copy()
    low = np.zeros((3, 3))
    rotations = np.tile(np.eye(3), (2, 1, 1))
    full = kernel.evaluate(positions, low, rotations)
    assert full.potential_j == pytest.approx(1.0, abs=1e-12)
    np.testing.assert_allclose(full.beam_force_n, (0.0, 0.0, 20.0), atol=1e-11)
    np.testing.assert_allclose(full.beam_force_n+full.plane_force_n,
                               0.0, atol=1e-12)
    np.testing.assert_allclose(full.beam_moment_nm, 0.0, atol=1e-11)
    np.testing.assert_allclose(full.beam_moment_nm+full.plane_moment_nm,
                               0.0, atol=1e-12)
    assert full.minimum_gap_m == pytest.approx(-0.1, abs=1e-14)
    np.testing.assert_array_equal(full.residual[24:], 0.0)
    np.testing.assert_array_equal(full.tangent[24:, :], 0.0)
    np.testing.assert_array_equal(full.tangent[:, 24:], 0.0)

    positions[:, 2] = (-0.2, 0.0, 0.2)
    partial = kernel.evaluate(positions, low, rotations)
    analytic_energy = quad(lambda x: 50.0*(0.2*x-0.1)**2,
                           -1.0, 0.5, epsabs=1e-13)[0]
    analytic_force = quad(lambda x: -100.0*(0.2*x-0.1),
                          -1.0, 0.5, epsabs=1e-13)[0]
    assert partial.potential_j == pytest.approx(analytic_energy, abs=1e-11)
    assert partial.beam_force_n[2] == pytest.approx(analytic_force, abs=1e-11)
    assert partial.minimum_gap_m == pytest.approx(-0.3, abs=1e-13)
    assert partial.integration_intervals >= 1
    np.testing.assert_allclose(partial.beam_moment_nm+partial.plane_moment_nm,
                               0.0, atol=1e-12)

    opened = positions.copy()
    opened[:, 2] += 1.0
    free = kernel.evaluate(opened, low, rotations)
    assert free.minimum_gap_m > 0.0
    assert free.integration_intervals == 0
    assert free.potential_j == 0.0
    np.testing.assert_array_equal(free.residual, 0.0)


def test_native_line_plane_curved_internal_work_and_smooth_tangent():
    reference = _reference(True)
    kernel = _kernel(reference, origin=(0.0, 0.25, 0.0),
                     normal=(0.0, 1.0, 0.0))
    positions = reference.coordinates.copy()
    low = np.zeros((3, 3))
    rotations = np.stack((Rotation.from_rotvec((0.02, 0.01, 0.05)).as_matrix(),
                          Rotation.from_rotvec((-0.01, 0.03, -0.02)).as_matrix()))
    base = kernel.evaluate(positions, low, rotations)
    assert np.linalg.norm(base.residual[18:24]) > 1e-5
    assert np.linalg.norm(base.tangent[18:24, :24]) > 1e-5
    direction = np.zeros(42)
    direction[0:3] = (0.02, -0.01, 0.0)
    direction[6:9] = (-0.01, 0.02, 0.01)
    direction[12:15] = (0.01, -0.005, 0.0)
    direction[18:21] = (0.04, -0.02, 0.03)
    direction[21:24] = (-0.03, 0.01, 0.02)

    def energy(sign, step):
        moved = positions.copy()
        cells = rotations.copy()
        for node in range(3):
            moved[node] += sign*step*direction[6*node:6*node+3]
        for cell in (0, 1):
            cells[cell] = (Rotation.from_rotvec(
                sign*step*direction[18+3*cell:21+3*cell],
            ).as_matrix() @ cells[cell])
        return kernel.evaluate(moved, low, cells).potential_j

    for step in (2e-4, 1e-4, 5e-5):
        first = (energy(1, step)-energy(-1, step))/(2*step)
        second = (energy(1, step)-2*base.potential_j+energy(-1, step))/step**2
        np.testing.assert_allclose(first, base.residual @ direction,
                                   rtol=1e-6, atol=1e-9)
        np.testing.assert_allclose(second, direction @ base.tangent @ direction,
                                   rtol=2e-5, atol=1e-7)
    np.testing.assert_allclose(base.beam_force_n+base.plane_force_n,
                               0.0, atol=1e-12)


def test_native_line_plane_rigid_covariance_and_budget_failure():
    reference = _reference(True)
    kernel = _kernel(reference, origin=(0.0, 0.25, 0.0),
                     normal=(0.0, 1.0, 0.0))
    positions = reference.coordinates
    low = np.zeros((3, 3))
    rotations = np.tile(np.eye(3), (2, 1, 1))
    base = kernel.evaluate(positions, low, rotations)
    rigid = Rotation.from_rotvec((0.2, 0.1, -0.3)).as_matrix()
    translation = np.array((0.4, -0.2, 0.5))
    transformed_kernel = _kernel(
        reference, origin=rigid@kernel.origin+translation,
        normal=rigid@kernel.normal,
    )
    transformed = transformed_kernel.evaluate(
        positions@rigid.T+translation, low, rigid@rotations,
    )
    assert transformed.potential_j == pytest.approx(base.potential_j,
                                                    rel=1e-12)
    np.testing.assert_allclose(transformed.beam_force_n,
                               rigid@base.beam_force_n, atol=1e-10)
    np.testing.assert_allclose(transformed.beam_moment_nm,
                               rigid@base.beam_moment_nm
                               + np.cross(translation, rigid@base.beam_force_n),
                               atol=1e-10)

    exhausted = _kernel(reference, origin=(0.0, 0.25, 0.0),
                        normal=(0.0, 1.0, 0.0), energy_tolerance=1e-30,
                        force_tolerance=1e-30, tangent_tolerance=1e-30,
                        maximum_intervals=1)
    with pytest.raises(MemoryError, match="interval budget"):
        exhausted.evaluate(positions, low, rotations)
