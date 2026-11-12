"""Development checks for a finite-support line-indexed shell pose."""

import numpy as np
import pytest
from scipy.spatial.transform import Rotation
from scipy import linalg

from anyfesolver._ge_beam3_pose_joint import RigidPoseJoint
from scripts._b3_e4_f03_native_fixture import build
from scripts._b3_e4_f03_native_spatial_port import NativeHalfCellSpatialPort
from scripts.b3_e4_f03_finite_patch_pose import FlatQ4FiniteSupportPose
from scripts.b3_e4_f03_plate_patch_work import raw_e4_tangent


def _patch(segments: int, *, station=1.):
    x = np.r_[(0., .25), np.linspace(.5, 1., segments+1)]
    y = np.array((-.1, 0., .1))
    reference = np.array([[(xi, yi, 0.) for yi in y] for xi in x])
    patch = FlatQ4FiniteSupportPose.at_reference(
        reference, x, y, station=station, line_y=0.,
        longitudinal_width=.2, transverse_bounds=(-.05, .05),
        thickness=.035)
    return patch, x, y


@pytest.mark.parametrize("station", (.5, .83, 1.))
def test_finite_patch_reference_and_rigid_motion_are_objective(station):
    patch, x, _ = _patch(8, station=station)
    q = np.zeros((*patch.reference.shape[:2], 6))
    point, frame = patch.pose(q)
    np.testing.assert_allclose(point, (station, 0., .0175),
                               rtol=0., atol=2e-14)
    np.testing.assert_allclose(frame, np.eye(3), rtol=0., atol=2e-14)
    assert patch.x_window[1]-patch.x_window[0] >= .1-1e-14
    assert patch.covariance > 0.
    np.testing.assert_allclose(patch.weights.sum(), 1., atol=2e-14)
    np.testing.assert_allclose(patch.moments.sum(), 0., atol=2e-14)

    turn = np.array((.24, -.17, .31))
    rotation = Rotation.from_rotvec(turn).as_matrix()
    shift = np.array((.4, -.2, .7))
    q[:, :, :3] = np.einsum(
        "ab,ijb->ija", rotation, patch.reference)+shift-patch.reference
    q[:, :, 3:] = turn
    moved, moved_frame = patch.pose(q)
    np.testing.assert_allclose(moved, rotation@point+shift,
                               rtol=0., atol=2e-13)
    np.testing.assert_allclose(moved_frame, rotation@frame,
                               rtol=0., atol=2e-13)

    x[-1] = 20.
    np.testing.assert_allclose(patch.pose(np.zeros_like(q))[0], point,
                               rtol=0., atol=2e-14)


def test_finite_patch_endpoint_yaw_filters_physical_core_witness():
    ratios = []
    energies = []
    for segments in (8, 16):
        patch, x, y = _patch(segments)
        q = np.zeros((*patch.reference.shape[:2], 6))
        q[-1, :, 1] = (1., -1., 1.)
        expected_yaw = float(np.dot(
            patch.moments[-1], (1., -1., 1.))/patch.covariance)
        step = 1e-6
        _, plus = patch.pose(step*q)
        _, minus = patch.pose(-step*q)
        measured_yaw = Rotation.from_matrix(
            minus.T@plus).as_rotvec()[2]/(2*step)
        assert measured_yaw == pytest.approx(
            expected_yaw, rel=2e-10, abs=2e-10)
        assert abs(measured_yaw) <= (.5/segments)/(8*patch.covariance)

        _, physical = raw_e4_tangent(x, y, thickness=.035)
        full = np.zeros(len(physical))
        for j in range(len(y)):
            full[6*(j*len(x)+len(x)-1)+1] = q[-1, j, 1]
        energy = float(full@physical@full)
        assert energy > 0.
        energies.append(energy)
        ratios.append(abs(measured_yaw)/np.sqrt(energy))
    assert energies[0]/energies[1] == pytest.approx(2., rel=2e-12)
    assert ratios[1] < ratios[0]


def test_finite_patch_yaw_transfers_physical_couple_and_preserves_shear():
    patch, x, y = _patch(8, station=.95)
    zero = np.zeros((*patch.reference.shape[:2], 6))
    shell_force_y = -patch.moments/patch.covariance
    np.testing.assert_allclose(shell_force_y.sum(), 0.,
                               rtol=0., atol=2e-13)
    assert np.sum(patch.reference[:, :, 0]*shell_force_y) == pytest.approx(
        -1., abs=2e-13)
    _, physical = raw_e4_tangent(x, y, thickness=.035)
    load = np.zeros(len(physical))
    for i in range(len(x)):
        for j in range(len(y)):
            load[6*(j*len(x)+i)+1] = shell_force_y[i, j]
    np.testing.assert_array_equal(load[5::6], 0.)
    free = np.array([6*(j*len(x)+i)+axis
                     for j in range(len(y)) for i in range(1, len(x))
                     for axis in range(5)])
    factor = linalg.cho_factor(physical[np.ix_(free, free)], lower=True)
    compliance = float(load[free]@linalg.cho_solve(factor, load[free]))
    assert compliance > 0.

    shell_position, shell_frame = patch.pose(zero)
    joint = RigidPoseJoint(
        np.array((shell_position, shell_position)),
        np.array((shell_frame, shell_frame)))
    direction = np.zeros_like(zero)
    direction[:, :, 1] = np.arange(direction.shape[0]*direction.shape[1]
                                   ).reshape(direction.shape[:2])/13.
    _, frame, position_step, frame_step = patch.directional(zero, direction)
    skew = frame_step@frame.T
    spin = np.array((skew[2, 1], skew[0, 2], skew[1, 0]))
    pure_yaw = np.array((0., 0., 0., 0., 0., 1.))
    result = joint.evaluate(
        np.array((shell_position, shell_position)),
        np.array((shell_frame, shell_frame)), pure_yaw)
    predicted = float(
        pure_yaw@result.constraint_jacobian@
        np.r_[position_step, spin, np.zeros(6)])
    assert predicted == pytest.approx(
        np.sum(shell_force_y*direction[:, :, 1]),
        rel=2e-13, abs=2e-13)

    shear = zero.copy()
    shear[:, :, 2] = .03*(patch.reference[:, :, 0]-.95)
    sheared_position, sheared_frame = patch.pose(shear)
    np.testing.assert_allclose(sheared_frame, np.eye(3),
                               rtol=0., atol=2e-14)
    assert abs(shear[-1, 1, 2]-shear[0, 1, 2]) > 1e-3
    pure_vertical = np.array((0., 0., 1., 0., 0., 0.))
    vertical_work = joint.evaluate(
        np.array((sheared_position, shell_position)),
        np.array((sheared_frame, shell_frame)), pure_vertical).potential
    assert vertical_work == pytest.approx(
        -(sheared_position[2]-shell_position[2]), abs=2e-14)
    assert abs(vertical_work) > 1e-5


@pytest.mark.parametrize("multiplier", (
    (1., -2., 3., .5, -1., 2.),
    (0., 0., 0., 0., 0., 1.),
))
def test_finite_patch_six_channel_work_follows_actual_shell_and_native_update(
        multiplier):
    patch, _, _ = _patch(8, station=.95)
    beam = build(4, z_offset=.0675, assemble=False)
    element, cell, weight, _ = beam._locate(.95)
    port = NativeHalfCellSpatialPort(beam.context, element, cell, weight)
    initial = beam.context.initial.mechanical
    beam_offset = np.array((0., 0., -.05))
    reference_shell, reference_frame = patch.pose(
        np.zeros((*patch.reference.shape[:2], 6)))
    high, low, beam_frame = port.pose_pair(initial)
    reference_beam = high+low+beam_frame@beam_offset
    joint = RigidPoseJoint(
        np.array((reference_shell, reference_beam)),
        np.array((reference_frame, beam_frame)))
    rng = np.random.default_rng(805)
    q = rng.normal(scale=.006, size=(*patch.reference.shape[:2], 6))
    dq = rng.normal(scale=.02, size=q.shape)
    db = rng.normal(scale=.02, size=beam.context.count)
    db[list(beam.context.fixed)] = 0.
    lam = np.asarray(multiplier)

    def evaluation(shell_total, beam_state):
        shell_position, shell_frame = patch.pose(shell_total)
        high, low, native_frame = port.pose_pair(beam_state)
        beam_position = high+low+native_frame@beam_offset
        return joint.evaluate(
            np.array((shell_position, beam_position)),
            np.array((shell_frame, native_frame)), lam)

    shell_position, shell_frame, shell_step, frame_step = patch.directional(
        q, dq)
    high, low, native_frame = port.pose_pair(initial)
    arm = native_frame@beam_offset
    spatial = port.tangent(initial)@db
    native_step = spatial[:3]+np.cross(spatial[3:], arm)
    frame_spin = frame_step@shell_frame.T
    shell_spin = np.array((frame_spin[2, 1], frame_spin[0, 2],
                           frame_spin[1, 0]))
    pose_step = np.r_[shell_step, shell_spin, native_step, spatial[3:]]
    result = evaluation(q, initial)
    predicted = float(lam@result.constraint_jacobian@pose_step)
    assert predicted == pytest.approx(result.residual[:12]@pose_step,
                                      rel=2e-13, abs=2e-13)
    step = 2e-6
    plus = evaluation(
        q+step*dq, beam.context.advance(initial, step*db))
    minus = evaluation(
        q-step*dq, beam.context.advance(initial, -step*db))
    observed = (plus.potential-minus.potential)/(2*step)
    assert observed == pytest.approx(predicted, rel=2e-6, abs=2e-8)
