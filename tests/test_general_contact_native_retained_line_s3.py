"""Local native B3-GE line contact with a moving physical S3 interior."""

import numpy as np
import pytest
from scipy.integrate import quad
from scipy.spatial.transform import Rotation

from anyfesolver._ge_beam3_centered_reference import (
    CenteredCurvedBeam3ReferenceGeometry,
)
from anyfesolver.general_contact.native_retained_line_s3 import (
    NativeRetainedRoundLineS3Interior,
)
from anyfesolver.general_contact.overlap import PlanarFace


def _case(*, curved=False, maximum_integration_intervals=100):
    beam = np.array(((0.0, 0.0, 0.08),
                     (1.05 if curved else 1.0, 0.0, 0.08),
                     (2.0, 0.0, 0.08)))
    reference = CenteredCurvedBeam3ReferenceGeometry(
        beam, np.tile(np.eye(3), (3, 1, 1)),
    )
    shell = np.array(((-5.0, -5.0, 0.0),
                      (5.0, -5.0, 0.0),
                      (0.0, 10.0, 0.0)))
    face = PlanarFace((1, 21), shell)
    kernel = NativeRetainedRoundLineS3Interior(
        11, reference, (1, 2, 3), 21, face, (4, 5, 6),
        radius_m=0.1, line_stiffness_n_per_m2=100.0,
        rim_clearance_m=0.1, geometry_tolerance_m=1e-10,
        energy_tolerance_j=1e-10, force_tolerance_n=1e-9,
        tangent_tolerance_n_per_m=1e-7,
        maximum_integration_intervals=maximum_integration_intervals,
    )
    return kernel, beam, np.zeros((3, 3)), np.tile(np.eye(3), (2, 1, 1)), shell


def test_parallel_plane_energy_force_and_wrench():
    kernel, beam, low, rotations, shell = _case()
    result = kernel.evaluate(beam, low, rotations, shell)
    assert result.potential_j == pytest.approx(0.04, abs=1e-12)
    assert result.minimum_gap_m == pytest.approx(-0.02, abs=1e-12)
    np.testing.assert_allclose(result.beam_force_n, (0, 0, 4), atol=1e-10)
    np.testing.assert_allclose(result.shell_force_n, (0, 0, -4), atol=1e-10)
    np.testing.assert_allclose(result.beam_force_n+result.shell_force_n, 0, atol=1e-10)
    np.testing.assert_allclose(result.beam_moment_nm+result.shell_moment_nm,
                               0, atol=1e-10)
    assert result.residual.shape == (51,)
    assert result.tangent.shape == (51, 51)
    assert result.residual.flags.writeable is False
    assert result.active_intervals == 2


def test_shell_vertex_work_and_tangent():
    kernel, beam, low, rotations, shell = _case()
    center = kernel.evaluate(beam, low, rotations, shell)
    step = 1e-5
    plus, minus = shell.copy(), shell.copy()
    plus[1, 2] += step
    minus[1, 2] -= step
    forward = kernel.evaluate(beam, low, rotations, plus)
    backward = kernel.evaluate(beam, low, rotations, minus)
    index = 42+3+2
    assert (forward.potential_j-backward.potential_j)/(2*step) == pytest.approx(
        center.residual[index], rel=2e-7, abs=2e-8)
    np.testing.assert_allclose((forward.residual-backward.residual)/(2*step),
                               center.tangent[:, index], rtol=2e-5, atol=2e-5)


def test_retained_rotation_work_curvature_and_independent_integral():
    kernel, beam, low, rotations, shell = _case(curved=True)
    rotations[0] = Rotation.from_rotvec((0.0, 0.12, 0.0)).as_matrix()
    center = kernel.evaluate(beam, low, rotations, shell)

    def rotated(delta):
        changed = rotations.copy()
        changed[0] = Rotation.from_rotvec((0.0, delta, 0.0)).as_matrix() @ rotations[0]
        return kernel.evaluate(beam, low, changed, shell)

    step = 1e-4
    forward, backward = rotated(step), rotated(-step)
    angular = 19
    assert (forward.potential_j-backward.potential_j)/(2*step) == pytest.approx(
        center.residual[angular], rel=1e-6, abs=1e-8)
    assert (forward.potential_j-2*center.potential_j+backward.potential_j)/step**2 == pytest.approx(
        center.tangent[angular, angular], rel=2e-4, abs=2e-5)
    assert abs(center.residual[angular]) > 1e-6

    # Scalar reference: independently interpolate the current Q2 position and
    # the undeformed reference measure, then integrate the line penalty.
    total = 0.0
    for cell in (0, 1):
        endpoints = beam[cell:cell+2]
        lift = kernel.beam_reference.half_cell_lift(cell, 0.5)
        current = np.vstack((endpoints[0], 0.5*np.sum(endpoints, axis=0)
                             +rotations[cell]@lift, endpoints[1]))
        original = np.vstack((endpoints[0], 0.5*np.sum(endpoints, axis=0)
                              +lift, endpoints[1]))

        def density(t):
            shape = np.array((2*(t-0.5)*(t-1), 4*t*(1-t), 2*t*(t-0.5)))
            derivative = np.array((4*t-3, 4-8*t, 4*t-1))
            gap = shape@current[:, 2]-0.1
            return 50.0*min(0.0, gap)**2*np.linalg.norm(derivative@original)

        total += quad(density, 0, 1, epsabs=1e-12, epsrel=1e-12)[0]
    assert center.potential_j == pytest.approx(total, rel=1e-9, abs=1e-11)


def test_definition_and_physical_rim_rejections():
    kernel, beam, low, rotations, shell = _case()
    degenerate = shell.copy()
    degenerate[2] = degenerate[1]
    with pytest.raises(ValueError, match="degenerate geometry"):
        kernel.evaluate(beam, low, rotations, degenerate)
    near_rim = shell.copy()
    near_rim[:, 1] += 100.0
    with pytest.raises(ValueError, match="rim"):
        kernel.evaluate(beam, low, rotations, near_rim)
    kernel.shell_reference.corners.setflags(write=True)
    kernel.shell_reference.corners[0, 0] -= 1.0
    with pytest.raises(ValueError, match="definition changed"):
        kernel.evaluate(beam, low, rotations, shell)


def test_common_rigid_motion_and_opening():
    kernel, beam, low, rotations, shell = _case()
    baseline = kernel.evaluate(beam, low, rotations, shell)
    rigid = Rotation.from_euler("x", 170, degrees=True).as_matrix()
    shift = np.array((0.7, -1.1, 2.3))
    moved = kernel.evaluate(beam@rigid.T+shift, low,
                            np.tile(rigid, (2, 1, 1)), shell@rigid.T+shift)
    assert moved.potential_j == pytest.approx(baseline.potential_j, abs=1e-12)
    np.testing.assert_allclose(moved.beam_force_n, rigid@baseline.beam_force_n,
                               atol=1e-10)
    np.testing.assert_allclose(moved.shell_force_n, rigid@baseline.shell_force_n,
                               atol=1e-10)
    opened = kernel.evaluate(beam+np.array((0, 0, 0.05)), low, rotations, shell)
    assert opened.potential_j == 0.0
    assert opened.active_intervals == 0
    np.testing.assert_array_equal(opened.residual, 0.0)
    np.testing.assert_array_equal(opened.tangent, 0.0)


def test_partial_activation_and_bounded_integration():
    kernel, beam, low, rotations, shell = _case()
    beam[0, 2] = beam[2, 2] = 0.12
    result = kernel.evaluate(beam, low, rotations, shell)
    assert result.active_intervals == 2
    assert result.minimum_gap_m == pytest.approx(-0.02)
    assert result.potential_j == pytest.approx(0.0066666666666667, rel=1e-9)
    limited, *_ = _case(maximum_integration_intervals=1)
    with pytest.raises(MemoryError, match="interval budget exhausted"):
        limited.evaluate(beam, low, rotations, shell)


def test_native_and_shell_identity_namespaces_are_separate():
    kernel, beam, low, rotations, shell = _case()
    same_numbers = NativeRetainedRoundLineS3Interior(
        21, kernel.beam_reference, (4, 5, 6),
        21, kernel.shell_reference, (4, 5, 6),
        radius_m=0.1, line_stiffness_n_per_m2=100.0,
        rim_clearance_m=0.1, geometry_tolerance_m=1e-10,
        energy_tolerance_j=1e-10, force_tolerance_n=1e-9,
        tangent_tolerance_n_per_m=1e-7,
        maximum_integration_intervals=100,
    )
    assert same_numbers.evaluate(beam, low, rotations, shell).potential_j == pytest.approx(
        kernel.evaluate(beam, low, rotations, shell).potential_j)
