"""Development-only six-channel line work from fixed finite shell patches.

This is a first-variation discriminator for F03. It is not a selected bonded
line, a mortar space, or a production attachment. Second variations and the
full mixed-system contract remain separate work.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from anyfesolver._ge_beam3_pose_joint import RigidPoseJoint
from scripts._b3_e4_f03_native_spatial_port import NativeHalfCellSpatialPort
from scripts.b3_e4_f03_finite_patch_pose import FlatQ4FiniteSupportPose
from scripts.b3_e4_f03_l1_physical_rail import _owned


@dataclass(frozen=True)
class _Station:
    patch: FlatQ4FiniteSupportPose
    port: NativeHalfCellSpatialPort
    joint: RigidPoseJoint
    segment: int
    measure: float
    beam_offset: np.ndarray


@dataclass(frozen=True)
class FinitePatchLineFirstVariation:
    """Two reference Gauss stations per segment, one fixed patch per station."""

    stations: tuple[_Station, ...]
    shell_shape: tuple[int, int, int]
    beam_size: int
    segment_count: int

    @classmethod
    def at_reference(
            cls, reference, x_cuts, y_cuts, port_at, beam_state, *,
            longitudinal_width, transverse_bounds, thickness,
            line_y=0., beam_offset=(0., 0., -.05),
            expected_reference_offset_shell=(0., 0., 0.)):
        x = np.asarray(x_cuts, dtype=float)
        y = np.asarray(y_cuts, dtype=float)
        ref = _owned(reference, (len(x), len(y), 3))
        offset = _owned(beam_offset, (3,))
        expected = _owned(expected_reference_offset_shell, (3,))
        abscissae = (.5-.5/np.sqrt(3.), .5+.5/np.sqrt(3.))
        stations = []
        for segment, (left, right) in enumerate(zip(x[:-1], x[1:])):
            measure = .5*(right-left)
            for alpha in abscissae:
                station = float(left+(right-left)*alpha)
                patch = FlatQ4FiniteSupportPose.at_reference(
                    ref, x, y, station=station, line_y=line_y,
                    longitudinal_width=longitudinal_width,
                    transverse_bounds=transverse_bounds, thickness=thickness)
                port = port_at(station)
                if not isinstance(port, NativeHalfCellSpatialPort):
                    raise TypeError("native half-cell port required")
                if beam_state is not port.context.initial.mechanical:
                    raise ValueError(
                        "finite patch reference requires virgin native state")
                shell, shell_frame = patch.pose(np.zeros((len(x), len(y), 6)))
                high, low, beam_frame = port.pose_pair(beam_state)
                beam = high+low+beam_frame@offset
                actual = shell_frame.T@(beam-shell)
                scale = max(right-left, longitudinal_width,
                            transverse_bounds[1]-transverse_bounds[0],
                            thickness, np.linalg.norm(offset), 1e-3)
                if np.linalg.norm(actual-expected) > 1e-10*scale:
                    raise ValueError("finite patch/native reference offset differs")
                joint = RigidPoseJoint(
                    np.array((shell, beam)),
                    np.array((shell_frame, beam_frame)))
                if np.max(np.abs(joint.evaluate(
                        np.array((shell, beam)),
                        np.array((shell_frame, beam_frame)),
                        np.zeros(6)).constraints)) > 1e-10:
                    raise ValueError("finite patch reference is not stress free")
                stations.append(_Station(
                    patch, port, joint, segment, float(measure), offset))
        context = stations[0].port.context
        if any(item.port.context is not context for item in stations):
            raise ValueError("one native retained context required")
        return cls(tuple(stations), (len(x), len(y), 6), context.count,
                   len(x)-1)

    def evaluate(self, shell_total, beam_state, shell_direction,
                 beam_direction, density):
        """Return sampled gaps and the exact first variation of weak work."""
        q = _owned(shell_total, self.shell_shape)
        dq = _owned(shell_direction, self.shell_shape)
        db = _owned(beam_direction, (self.beam_size,))
        lam = _owned(density, (self.segment_count, 6))
        gaps = np.empty((len(self.stations), 6))
        work_step = 0.
        for index, item in enumerate(self.stations):
            shell, shell_frame, shell_step, frame_step = item.patch.directional(
                q, dq)
            high, low, beam_frame = item.port.pose_pair(beam_state)
            arm = beam_frame@item.beam_offset
            beam = high+low+arm
            response = item.joint.evaluate(
                np.array((shell, beam)),
                np.array((shell_frame, beam_frame)), np.zeros(6))
            spatial = item.port.tangent(beam_state)@db
            beam_step = spatial[:3]+np.cross(spatial[3:], arm)
            frame_spin = frame_step@shell_frame.T
            shell_spin = np.array((frame_spin[2, 1], frame_spin[0, 2],
                                   frame_spin[1, 0]))
            pose_step = np.r_[shell_step, shell_spin,
                              beam_step, spatial[3:]]
            gaps[index] = response.constraints
            work_step += (item.measure *
                          lam[item.segment]@response.constraint_jacobian@
                          pose_step)
        return _owned(gaps, (len(self.stations), 6)), float(work_step)
