"""Nonzero-multiplier objective work through native and shell station ports."""

import numpy as np
import pytest

from anyfesolver._ge_beam3_pose_joint import RigidPoseJoint

from scripts._b3_e4_f03_coupled_point_work_candidate import directional_point_work
from scripts._b3_e4_f03_native_fixture import build
from scripts._b3_e4_f03_native_spatial_port import NativeHalfCellSpatialPort
from scripts._b3_e4_f03_shell_finite_pose_candidate import pose_from_e4_total
from scripts._b3_e4_f03_shell_chord_pose_candidate import pose_from_e4_total_chord
from scripts._b3_e4_f03_shell_trace import make_shell


@pytest.mark.parametrize("topology", ("Q4", "S3-V2D"))
@pytest.mark.parametrize("eccentricity", (0., .11))
@pytest.mark.parametrize("shell_port", ("rail", "line_chord_020",
                                        "line_chord_confined_020"))
def test_composed_point_work_first_and_mixed_second_variations(
        topology, eccentricity, shell_port):
    shell = make_shell(2, topology)
    beam = build(4)
    station = .37
    element, cell, weight, _ = beam._locate(station)
    port = NativeHalfCellSpatialPort(beam.context, element, cell, weight)
    reference_shell = np.zeros((shell.node_count, 6))
    reference_shell[:, 2] = -eccentricity
    pose_map = {"rail": pose_from_e4_total,
                "line_chord_020": pose_from_e4_total_chord,
                "line_chord_confined_020": lambda p, s, u: (
                    pose_from_e4_total_chord(
                        p, s, u, confined=True))}[shell_port]
    shell_ref = pose_map(shell, station, reference_shell.ravel())
    beam_high, beam_low, beam_frame = port.pose_pair(
        beam.context.initial.mechanical)
    joint = RigidPoseJoint(
        np.array((shell_ref[0], beam_high+beam_low)),
        np.array((shell_ref[1], beam_frame)))
    np.testing.assert_allclose(beam_high+beam_low-shell_ref[0],
                               (0., 0., eccentricity), rtol=0, atol=2e-14)
    virgin = joint.evaluate(
        np.array((shell_ref[0], beam_high)),
        np.array((shell_ref[1], beam_frame)), np.zeros(6),
        position_low=np.array((np.zeros(3), beam_low)))
    np.testing.assert_allclose(virgin.constraints, 0., rtol=0, atol=2e-14)

    rng = np.random.default_rng(721)
    shell_base = reference_shell+rng.normal(size=(shell.node_count, 6))*.006
    shell_first = rng.normal(size=shell_base.shape)*.04
    shell_second = rng.normal(size=shell_base.shape)*.04
    beam_base = np.zeros(beam.context.count)
    beam_first = np.zeros_like(beam_base)
    beam_second = np.zeros_like(beam_base)
    beam_base[6:9] = (.001, -.002, .003)
    beam_base[beam.context.nodal_count+3:beam.context.nodal_count+6] = (
        .018, -.012, .009)
    beam_first[6:12] = (.03, -.02, .01, .01, -.02, .015)
    beam_first[beam.context.nodal_count+3:beam.context.nodal_count+6] = (
        .06, -.04, .03)
    beam_second[12:15] = (-.02, .01, .04)
    beam_second[beam.context.nodal_count+3:beam.context.nodal_count+6] = (
        -.03, .05, -.02)
    base_state = beam.context.advance(
        beam.context.initial.mechanical, beam_base)
    force = np.array((18., -23., 31., 7., -11., 13.))
    analytic = directional_point_work(
        joint, shell, station, shell_base.ravel(), shell_first,
        shell_second, port, base_state, beam_first, beam_second, force,
        shell_port=shell_port)

    def potential(a, b):
        current_shell = shell_base+a*shell_first+b*shell_second
        shell_position, shell_frame = pose_map(
            shell, station, current_shell.ravel())
        current_beam = beam.context.advance(
            base_state, a*beam_first+b*beam_second)
        high, low, frame = port.pose_pair(current_beam)
        return joint.evaluate(
            np.array((shell_position, high)),
            np.array((shell_frame, frame)), force,
            position_low=np.array((np.zeros(3), low))).potential

    assert analytic.potential == pytest.approx(potential(0., 0.), abs=2e-13)
    h = 1e-5
    np.testing.assert_allclose((analytic.first, analytic.second),
        ((potential(h, 0.)-potential(-h, 0.))/(2*h),
         (potential(0., h)-potential(0., -h))/(2*h)),
        rtol=2e-7, atol=2e-9)
    h = 2e-3
    observed = (potential(h, h)-potential(h, -h)-
                potential(-h, h)+potential(-h, -h))/(4*h*h)
    np.testing.assert_allclose(analytic.mixed, observed,
                               rtol=2e-5, atol=2e-7)
