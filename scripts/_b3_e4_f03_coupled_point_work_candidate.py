"""Bounded F03 shell/native-B3 point work and full pose-pullback curvature.

This joins independently checked kinematic ports to the existing objective
pose joint at one material station. It proves neither a weak line multiplier
space nor a coupled element/material owner.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from anyfesolver._ge_beam3_pose_joint import RigidPoseJoint
from anyfesolver._ge_beam3_retained_generalized_state import Mechanical

from scripts._b3_e4_f03_native_spatial_port import (
    NativeHalfCellSpatialPort, NativeNodalSpatialPort)
from scripts._b3_e4_f03_shell_finite_pose_candidate import (
    directional_pose_from_e4_total, pose_from_e4_total)
from scripts._b3_e4_f03_shell_local_translation_candidate import (
    directional_pose_from_e4_total_local_translation,
    directional_pose_from_e4_total_local_director,
    directional_pose_from_e4_total_local_frame,
    pose_from_e4_total_local_translation,
    pose_from_e4_total_local_director,
    pose_from_e4_total_local_frame)
from scripts._b3_e4_f03_shell_strip_pose_candidate import (
    LINE_ORIGIN_PORT, STRIP_PORTS, directional_pose_from_e4_total_strip,
    pose_from_e4_total_strip)
from scripts._b3_e4_f03_shell_chord_pose_candidate import (
    TRANSVERSE_TOP_CHORD_PORT, directional_pose_from_e4_total_chord,
    pose_from_e4_total_chord)
from scripts._b3_e4_f03_shell_trace import ShellPanel


def _skew(vector: np.ndarray) -> np.ndarray:
    x, y, z = vector
    return np.array(((0., -z, y), (z, 0., -x), (-y, x, 0.)))


def _axial(matrix: np.ndarray) -> np.ndarray:
    return np.array((matrix[2, 1]-matrix[1, 2],
                     matrix[0, 2]-matrix[2, 0],
                     matrix[1, 0]-matrix[0, 1]))/2.


def _angular(first: np.ndarray, frame: np.ndarray) -> np.ndarray:
    return _axial(first@frame.T)


def _angular_mixed(mixed: np.ndarray, frame: np.ndarray,
                   first: np.ndarray, second: np.ndarray) -> np.ndarray:
    a, b = _skew(first), _skew(second)
    correction = .5*(a@b+b@a)
    return _axial(mixed@frame.T-correction)


@dataclass(frozen=True)
class DirectionalJointWork:
    """Owned scalar work, physical constraints, and directional derivatives."""

    potential: float
    constraints: np.ndarray
    constraints_first: np.ndarray
    constraints_second: np.ndarray
    first: float
    second: float
    mixed: float


def directional_point_work(
        joint: RigidPoseJoint,
        shell: ShellPanel, station: float, shell_total: np.ndarray,
        shell_first: np.ndarray, shell_second: np.ndarray,
        beam_port: NativeHalfCellSpatialPort | NativeNodalSpatialPort,
        beam_state: Mechanical,
        beam_first: np.ndarray, beam_second: np.ndarray,
        multipliers: np.ndarray, *,
        shell_port: str = "rail") -> DirectionalJointWork:
    """Pull the joint's chart Hessian through both actual station maps.

    Shell directions are additive total E4 coordinates. Beam directions are
    ``Context.advance`` retained increments. Joint residual rows are spatial
    physical port wrenches; the joint Hessian is its scalar Exp-chart Hessian.
    The shell/beam port curvature is added exactly once in that chart.
    """
    if not isinstance(joint, RigidPoseJoint) or not isinstance(
            beam_port, (NativeHalfCellSpatialPort, NativeNodalSpatialPort)):
        raise TypeError("objective joint and native station port required")
    shell_u = np.asarray(shell_first, dtype=float)
    shell_v = np.asarray(shell_second, dtype=float)
    beam_u = np.asarray(beam_first, dtype=float)
    beam_v = np.asarray(beam_second, dtype=float)
    if (shell_u.shape != (shell.node_count, 6) or
            shell_v.shape != (shell.node_count, 6) or
            beam_u.shape != (beam_port.context.count,) or
            beam_v.shape != (beam_port.context.count,) or
            not all(np.isfinite(a).all() for a in
                    (shell_u, shell_v, beam_u, beam_v))):
        raise ValueError("finite complete shell and beam directions required")
    if shell_port == "rail":
        pose_map = pose_from_e4_total
        derivative_map = directional_pose_from_e4_total
    elif shell_port == "local_translation":
        pose_map = pose_from_e4_total_local_translation
        derivative_map = directional_pose_from_e4_total_local_translation
    elif shell_port == "local_director":
        pose_map = pose_from_e4_total_local_director
        derivative_map = directional_pose_from_e4_total_local_director
    elif shell_port == "local_frame":
        pose_map = pose_from_e4_total_local_frame
        derivative_map = directional_pose_from_e4_total_local_frame
    elif shell_port in STRIP_PORTS:
        half_length,half_width=STRIP_PORTS[shell_port]
        pose_map = lambda p, s, u: pose_from_e4_total_strip(
            p, s, u, half_length=half_length, half_width=half_width,
            line_origin=shell_port==LINE_ORIGIN_PORT)
        derivative_map = lambda p, s, u, a, b: (
            directional_pose_from_e4_total_strip(
                p, s, u, a, b, half_length=half_length,
                half_width=half_width,
                line_origin=shell_port==LINE_ORIGIN_PORT))
    elif shell_port == "line_chord_020":
        pose_map = pose_from_e4_total_chord
        derivative_map = directional_pose_from_e4_total_chord
    elif shell_port == TRANSVERSE_TOP_CHORD_PORT:
        pose_map = lambda p, s, u: pose_from_e4_total_chord(
            p, s, u, half_chord=.05, axis="transverse_top")
        derivative_map = lambda p, s, u, a, b: (
            directional_pose_from_e4_total_chord(
                p, s, u, a, b, half_chord=.05,
                axis="transverse_top"))
    elif shell_port == "line_chord_confined_020":
        pose_map = lambda p, s, u: pose_from_e4_total_chord(
            p, s, u, confined=True)
        derivative_map = lambda p, s, u, a, b: (
            directional_pose_from_e4_total_chord(
                p, s, u, a, b, confined=True))
    else:
        raise ValueError("unknown bounded shell pose port")
    shell_position, shell_frame = pose_map(shell, station, shell_total)
    shell_d = derivative_map(
        shell, station, shell_total, shell_u, shell_v)
    beam_high, beam_low, beam_frame = beam_port.pose_pair(beam_state)
    beam_jacobian = beam_port.tangent(beam_state)
    beam_position_mixed, beam_frame_mixed = beam_port.second_variation(
        beam_state, beam_u, beam_v)
    w = np.asarray(multipliers, dtype=float)
    if w.shape != (6,) or not np.isfinite(w).all():
        raise ValueError("finite six-component joint multipliers required")
    evaluation = joint.evaluate(
        np.array((shell_position, beam_high)),
        np.array((shell_frame, beam_frame)), w,
        position_low=np.array((np.zeros(3), beam_low)))
    shell_angular_u = _angular(shell_d.frame_first, shell_frame)
    shell_angular_v = _angular(shell_d.frame_second, shell_frame)
    beam_direction_u = beam_jacobian@beam_u
    beam_direction_v = beam_jacobian@beam_v
    du = np.r_[shell_d.position_first, shell_angular_u,
               beam_direction_u]
    dv = np.r_[shell_d.position_second, shell_angular_v,
               beam_direction_v]
    curved = np.r_[
        shell_d.position_mixed,
        _angular_mixed(shell_d.frame_mixed, shell_frame,
                       shell_angular_u, shell_angular_v),
        beam_position_mixed,
        _angular_mixed(beam_frame_mixed, beam_frame,
                       beam_direction_u[3:], beam_direction_v[3:])]
    gradient = evaluation.residual[:12]
    mixed = (du@evaluation.chart_hessian[:12, :12]@dv+
             gradient@curved)
    return DirectionalJointWork(
        float(evaluation.potential), evaluation.constraints.copy(),
        evaluation.constraint_jacobian@du,
        evaluation.constraint_jacobian@dv,
        float(gradient@du), float(gradient@dv), float(mixed))
