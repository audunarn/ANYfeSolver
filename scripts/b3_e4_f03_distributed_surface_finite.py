"""Bounded finite-chart proofs for physical terminal-strip candidates.

This is a G0B mechanics experiment, not a production attachment or a line
mortar. The six rows are the mean material-frame surface-position gap and its
reference-arm first moment over one flat terminal footprint.
The optional hybrid replaces one position-moment row with a shell-director
pitch row. Weighted hybrid rows remain exploratory line operators.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from anyfesolver import _ge_beam3_mixed_ad as ad
from anyfesolver._ge_beam3_mixed_ad import rotation_exponential
from anyfesolver.corotational import rotation_matrix_from_vector
from scripts._b3_e4_f03_native_spatial_port import NativeHalfCellSpatialPort
from scripts.b3_e4_f03_local_footprint_work import (
    bond_x_segments, distributed_surface_patch_trace, skew,
    structured_q4_point_shape, structured_c1_mixed_point_shape,
)


@dataclass(frozen=True)
class _Point:
    x: float
    y: float
    weight: float
    nodes: np.ndarray
    shape: np.ndarray
    port: NativeHalfCellSpatialPort


class FiniteSurfaceMomentPatch:
    """Actual E4 director and retained B3 half-cell chart at a bounded patch."""

    def __init__(self, xs: np.ndarray, ys: np.ndarray, beam, *,
                 thickness: float, quadrature_order: int = 3,
                 mixed_s3: bool = False,
                 mixed_s3_diagonal: str = "bl_tr"):
        if quadrature_order not in (3, 5):
            raise ValueError("bounded surface quadrature order must be 3 or 5")
        if (mixed_s3_diagonal not in ("bl_tr", "br_tl") or
                not mixed_s3 and mixed_s3_diagonal != "bl_tr"):
            raise ValueError("mixed C1 S3 diagonal requires bounded mixed case")
        self.xs = np.asarray(xs, dtype=float)
        self.ys = np.asarray(ys, dtype=float)
        self.beam = beam
        self.s3_diagonal = mixed_s3_diagonal
        if mixed_s3:
            from scripts.b3_e4_f03_plate_patch_work import c1_mixed_s3_cells
            self.s3_cells = c1_mixed_s3_cells(self.xs, self.ys)
        else:
            self.s3_cells = frozenset()
        self.top = thickness / 2
        self.moment_center_x = .95
        self.shell_count = len(self.xs) * len(self.ys)
        self.shell_size = 6 * self.shell_count
        self.reference_nodes = np.array(
            [(x, y, 0.) for y in self.ys for x in self.xs], dtype=float)
        self.points = []
        roots, weights = np.polynomial.legendre.leggauss(quadrature_order)
        inertia = np.zeros((3, 3))
        area = 0.
        for i, (x0, x1) in enumerate(zip(self.xs[:-1], self.xs[1:])):
            lx, hx = max(float(x0), .9), min(float(x1), 1.)
            if hx <= lx:
                continue
            for sx0, sx1 in bond_x_segments(lx, hx, beam.abscissae):
                for j, (y0, y1) in enumerate(zip(self.ys[:-1], self.ys[1:])):
                    ly, hy = max(float(y0), -.05), min(float(y1), .05)
                    if hy <= ly:
                        continue
                    for xi, wx in zip(roots, weights):
                        x = (sx0+sx1)/2 + float(xi)*(sx1-sx0)/2
                        element, cell, fraction, _ = beam._locate(x)
                        port = NativeHalfCellSpatialPort(
                            beam.context, element, cell, fraction)
                        if (i,j) in self.s3_cells:
                            xi_cell = (x-x0)/(x1-x0)
                            diagonal = float(y0+(
                                xi_cell if self.s3_diagonal == "bl_tr" else
                                1.-xi_cell)*(y1-y0))
                            y_ranges = ((ly,min(hy,diagonal)),
                                        (max(ly,diagonal),hy))
                        else:
                            y_ranges = ((ly,hy),)
                        for ylo, yhi in y_ranges:
                            if yhi <= ylo:
                                continue
                            jac = (sx1-sx0)*(yhi-ylo)/4
                            for eta, wy in zip(roots, weights):
                                y = (ylo+yhi)/2 + float(eta)*(yhi-ylo)/2
                                nodes, shape = self._point_shape(x, y)
                                w = float(wx*wy*jac)/.01
                                self.points.append(_Point(x, y, w, nodes, shape, port))
                                arm = np.array((x-self.moment_center_x, y, 0.))
                                inertia += w*((arm@arm)*np.eye(3)-np.outer(arm, arm))
                                area += float(wx*wy*jac)
        if abs(area-.01) > 1e-12 or len(self.points) == 0:
            raise ValueError("complete bounded 0.1 by 0.1 m patch required")
        self.inertia = inertia
        if np.linalg.eigvalsh(inertia)[0] <= 1e-8:
            raise ValueError("physical footprint inertia is degenerate")
        element, cell, fraction, _ = beam._locate(.95)
        self.center = NativeHalfCellSpatialPort(
            beam.context, element, cell, fraction)

    def _point_shape(self, x: float, y: float
                     ) -> tuple[np.ndarray, np.ndarray]:
        return (structured_c1_mixed_point_shape(
            self.xs, self.ys, x, y, self.s3_cells,
            diagonal=self.s3_diagonal) if self.s3_cells else
            structured_q4_point_shape(self.xs, self.ys, x, y))

    def _shell(self, coordinates: np.ndarray, frames: np.ndarray,
               point: _Point):
        position = np.zeros(3)
        offset = np.array((0., 0., self.top))
        for node, shape in zip(point.nodes, point.shape):
            node = int(node)
            directed = frames[node] @ offset
            position += shape*(coordinates[node]+directed)
        return position

    def linearized_gap_statistics(self, shell_step, beam_step) -> dict:
        """Recover sampled local slip separately from the six weak moments."""
        shell = np.asarray(shell_step, dtype=float).reshape(self.shell_count, 6)
        beam = np.asarray(beam_step, dtype=float)
        if beam.shape != (self.beam.context.count,) or not (
                np.isfinite(shell).all() and np.isfinite(beam).all()):
            raise ValueError("finite complete linearized surface state required")
        base = self.beam.context.initial.mechanical
        _, _, center_frame = self.center.pose_pair(base)
        material = center_frame.T
        mean = np.zeros(3)
        moment = np.zeros(3)
        square = 0.
        maximum = 0.
        for point in self.points:
            shell_displacement = np.zeros(3)
            for node, shape in zip(point.nodes, point.shape):
                step = shell[int(node)]
                shell_displacement += shape*(
                    step[:3]+np.cross(step[3:], (0., 0., self.top)))
            pose = point.port.tangent(base) @ beam
            beam_displacement = pose[:3]+np.cross(
                pose[3:], (0., point.y, -.05))
            gap = material @ (shell_displacement-beam_displacement)
            arm = np.array((point.x-self.moment_center_x, point.y, 0.))
            mean += point.weight*gap
            moment += point.weight*np.cross(arm, gap)
            square += point.weight*float(gap @ gap)
            maximum = max(maximum, float(np.linalg.norm(gap)))
        weak = np.r_[mean, np.linalg.solve(self.inertia, moment)]
        return {
            "quadrature_rms_position_gap_m": float(np.sqrt(square)),
            "quadrature_max_position_gap_m": maximum,
            "weak_six_gap_m_rad": weak.tolist(),
            "quadrature_points": len(self.points),
            "meaning": "sampled linearized shell-top minus native beam-bottom position; six weak moments may vanish while local slip remains",
        }

    def evaluate(self, shell_coordinates: np.ndarray, shell_frames: np.ndarray,
                 beam_state, first=None, second=None, *,
                 _shell_frame_variations=None):
        """Return six gaps and, optionally, exact first and mixed variations.

        Both directions are applied with spatial Exp updates from the same
        predecessor. The result includes curvature of the moving material
        frame, E4 director, and retained B3 cell frame.
        """
        coordinates = np.asarray(shell_coordinates, dtype=float)
        frames = np.asarray(shell_frames, dtype=float)
        if coordinates.shape != (self.shell_count, 3) or frames.shape != (self.shell_count, 3, 3):
            raise ValueError("complete shell physical state required")
        directional = first is not None or second is not None
        if directional:
            if first is None or second is None:
                raise ValueError("two directions required")
            su, bu = first
            sv, bv = second
            su, sv = np.asarray(su).reshape(self.shell_count, 6), np.asarray(sv).reshape(self.shell_count, 6)
            bu, bv = np.asarray(bu), np.asarray(bv)
            if _shell_frame_variations is not None:
                frame_directions = tuple(np.asarray(a, dtype=float)
                                         for a in _shell_frame_variations)
                if (len(frame_directions) != 3 or any(
                        a.shape != frames.shape or not np.isfinite(a).all()
                        for a in frame_directions)):
                    raise ValueError("complete E4 frame variations required")
        elif _shell_frame_variations is not None:
            raise ValueError("E4 frame variations require two directions")
        _, _, center_frame = self.center.pose_pair(beam_state)
        qc = center_frame.T
        if directional:
            center_u = self.center.tangent(beam_state) @ bu
            center_v = self.center.tangent(beam_state) @ bv
            qc_u = (skew(center_u[3:]) @ center_frame).T
            qc_v = (skew(center_v[3:]) @ center_frame).T
            _, center_uv = self.center.second_variation(beam_state, bu, bv)
            qc_uv = center_uv.T
        rows = np.zeros((3, 4 if directional else 1))
        moments = np.zeros_like(rows)
        for point in self.points:
            shell = self._shell(coordinates, frames, point)
            high, low, beam_frame = point.port.pose_pair(beam_state)
            offset = np.array((0., point.y, -.05))
            beam_position = high+low+beam_frame @ offset
            gap = shell-beam_position
            values = [qc @ gap]
            if directional:
                shell_u = np.zeros(3)
                shell_v = np.zeros(3)
                shell_uv = np.zeros(3)
                directed_offset = np.array((0., 0., self.top))
                for node, shape in zip(point.nodes, point.shape):
                    node = int(node)
                    directed = frames[node] @ directed_offset
                    if _shell_frame_variations is None:
                        a, b = skew(su[node, 3:]), skew(sv[node, 3:])
                        directed_u, directed_v = a@directed, b@directed
                        directed_uv = .5*(a@b+b@a)@directed
                    else:
                        fu, fv, fuv = frame_directions
                        directed_u = fu[node]@directed_offset
                        directed_v = fv[node]@directed_offset
                        directed_uv = fuv[node]@directed_offset
                    shell_u += shape*(su[node, :3]+directed_u)
                    shell_v += shape*(sv[node, :3]+directed_v)
                    shell_uv += shape*directed_uv
                port = point.port.tangent(beam_state)
                pu, pv = port@bu, port@bv
                beam_u = pu[:3]+np.cross(pu[3:], beam_frame @ offset)
                beam_v = pv[:3]+np.cross(pv[3:], beam_frame @ offset)
                position_uv, frame_uv = point.port.second_variation(beam_state, bu, bv)
                beam_uv = position_uv+frame_uv@offset
                gap_u, gap_v = shell_u-beam_u, shell_v-beam_v
                values.extend((qc_u@gap+qc@gap_u,
                               qc_v@gap+qc@gap_v,
                               qc_uv@gap+qc_u@gap_v+qc_v@gap_u+qc@(shell_uv-beam_uv)))
            arm = np.array((point.x-self.moment_center_x, point.y, 0.))
            for column, value in enumerate(values):
                rows[:, column] += point.weight*value
                moments[:, column] += point.weight*np.cross(arm, value)
        return np.vstack((rows, np.linalg.solve(self.inertia, moments)))

    def advance(self, coordinates, frames, beam_state, shell_step, beam_step):
        shell_step = np.asarray(shell_step).reshape(self.shell_count, 6)
        made_coordinates = np.asarray(coordinates)+shell_step[:, :3]
        made_frames = np.array([
            rotation_exponential(step[3:]) @ frame
            for step, frame in zip(shell_step, frames)])
        return (made_coordinates, made_frames,
                self.beam.context.advance(beam_state, np.asarray(beam_step)))


class FiniteSurfaceLineHybrid(FiniteSurfaceMomentPatch):
    """Bounded six-channel reduced strip-pose line connection.

    Mean shell-top/beam-bottom position supplies three force channels.
    Transverse position moments supply Mx/Mz; an explicit E4 director-pitch
    tie supplies My. The latter is a declared reduced-section assumption.
    Higher transverse slip shapes can remain while all six continuous
    section gaps vanish: this is not a pointwise welded surface bond.
    Material-frame duals and actual-chart variations remain separate from
    display-ready spatial wrenches.
    """

    def __init__(self, xs: np.ndarray, ys: np.ndarray, beam, *,
                 thickness: float,
                 x_interval: tuple[float, float] = (.9, 1.),
                 x_weight: Callable[[float], float] | None = None,
                 quadrature_order: int = 3,
                 mixed_s3: bool = False,
                 mixed_s3_diagonal: str = "bl_tr"):
        super().__init__(xs, ys, beam, thickness=thickness,
                         quadrature_order=quadrature_order,
                         mixed_s3=mixed_s3,
                         mixed_s3_diagonal=mixed_s3_diagonal)
        low, high = x_interval
        if (not (.9 <= low < high <= 1.) or
                min(abs(self.xs-low)) > 1e-12 or
                min(abs(self.xs-high)) > 1e-12):
            raise ValueError("C1 line interval requires shell-host endpoints")
        selected = [p for p in self.points if low < p.x < high]
        profile = [1. if x_weight is None else float(x_weight(p.x))
                   for p in selected]
        if not all(np.isfinite(profile)):
            raise ValueError("finite line multiplier profile required")
        mass = float(sum(p.weight*value for p,value in zip(selected,profile)))
        if mass <= 1e-12:
            raise ValueError("positive line multiplier basis integral required")
        self.basis_integral_m = .1*mass
        self.points = [
            _Point(p.x, p.y, p.weight*value/mass,
                   p.nodes, p.shape, p.port)
            for p,value in zip(selected,profile) if value != 0.]
        if not self.points or abs(sum(p.weight for p in self.points)-1.) > 1e-12:
            raise ValueError("C1 line interval lacks exact host quadrature")
        self.x_interval = x_interval
        self.transverse_inertia = float(sum(
            point.weight*point.y**2 for point in self.points))
        if self.transverse_inertia <= 1e-8:
            raise ValueError("C1 line has degenerate transverse arm")

    def _director(self, frames: np.ndarray, x: float):
        nodes, shape = self._point_shape(x, 0.)
        return nodes, shape, sum((weight*frames[int(node), :, 2]
                                  for node, weight in zip(nodes, shape)),
                                 np.zeros(3))

    @staticmethod
    def _pitch(frame: np.ndarray, director: np.ndarray,
               frame_u=None, frame_v=None, frame_uv=None,
               director_u=None, director_v=None, director_uv=None):
        x_axis, z_axis = frame[:, 0], frame[:, 2]
        a, c = float(x_axis@director), float(z_axis@director)
        if (not np.isfinite((a, c)).all() or c <= .2 or
                np.linalg.norm(director) <= .5):
            raise ValueError("C1 line director pitch left its guarded chart")
        angle = float(np.arctan2(a, c))
        if frame_u is None:
            return (angle,)
        a_u = float(frame_u[:, 0]@director+x_axis@director_u)
        c_u = float(frame_u[:, 2]@director+z_axis@director_u)
        a_v = float(frame_v[:, 0]@director+x_axis@director_v)
        c_v = float(frame_v[:, 2]@director+z_axis@director_v)
        a_uv = float(frame_uv[:, 0]@director+
                     frame_u[:, 0]@director_v+
                     frame_v[:, 0]@director_u+x_axis@director_uv)
        c_uv = float(frame_uv[:, 2]@director+
                     frame_u[:, 2]@director_v+
                     frame_v[:, 2]@director_u+z_axis@director_uv)
        denom = a*a+c*c
        n_u, n_v = c*a_u-a*c_u, c*a_v-a*c_v
        mixed = ((c_v*a_u+c*a_uv-a_v*c_u-a*c_uv)/denom-
                 n_u*(2*a*a_v+2*c*c_v)/denom**2)
        return angle, n_u/denom, n_v/denom, mixed

    def linearized_continuous_gap_statistics(self, shell_step, beam_step) -> dict:
        """Sample the unprojected line pitch alongside the positional gap."""
        report = self.linearized_gap_statistics(shell_step, beam_step)
        shell = np.asarray(shell_step, dtype=float).reshape(self.shell_count, 6)
        beam = np.asarray(beam_step, dtype=float)
        base = self.beam.context.initial.mechanical
        frames = np.broadcast_to(np.eye(3),
                                 (self.shell_count, 3, 3))
        pitches = []
        for point in self.points:
            _, _, frame = point.port.pose_pair(base)
            nodes, shape, director = self._director(frames, point.x)
            director_step = sum((weight*np.cross(shell[int(node), 3:],
                                                  frames[int(node), :, 2])
                                 for node, weight in zip(nodes, shape)),
                                np.zeros(3))
            spin = (point.port.tangent(base)@beam)[3:]
            x_axis, z_axis = frame[:, 0], frame[:, 2]
            a, c = float(x_axis@director), float(z_axis@director)
            a_step = float(np.cross(spin, x_axis)@director+
                           x_axis@director_step)
            c_step = float(np.cross(spin, z_axis)@director+
                           z_axis@director_step)
            pitches.append((c*a_step-a*c_step)/(a*a+c*c))
        weights = np.array([point.weight for point in self.points])
        values = np.asarray(pitches)
        report.update({
            "quadrature_rms_director_pitch_gap_rad": float(
                np.sqrt(weights@(values*values))),
            "quadrature_max_abs_director_pitch_gap_rad": float(
                np.max(np.abs(values))),
            "quadrature_mean_director_pitch_gap_rad": float(weights@values),
            "meaning": ("reference-linearized unprojected shell-top minus "
                        "native beam-bottom position and E4 director minus "
                        "B3 section pitch; not an accepted finite-state gap"),
        })
        return report

    def evaluate(self, shell_coordinates: np.ndarray, shell_frames: np.ndarray,
                 beam_state, first=None, second=None, *,
                 _shell_frame_variations=None) -> np.ndarray:
        """Six gaps and optional exact spatial-left chart variations."""
        coordinates = np.asarray(shell_coordinates, dtype=float)
        frames = np.asarray(shell_frames, dtype=float)
        if (coordinates.shape != (self.shell_count, 3) or
                frames.shape != (self.shell_count, 3, 3) or
                not (np.isfinite(coordinates).all() and np.isfinite(frames).all())):
            raise ValueError("complete finite shell physical state required")
        directional = first is not None or second is not None
        if directional:
            if first is None or second is None:
                raise ValueError("two complete chart directions required")
            su, bu = first
            sv, bv = second
            su = np.asarray(su, dtype=float).reshape(self.shell_count, 6)
            sv = np.asarray(sv, dtype=float).reshape(self.shell_count, 6)
            bu, bv = np.asarray(bu, dtype=float), np.asarray(bv, dtype=float)
            if (bu.shape != (self.beam.context.count,) or
                    bv.shape != bu.shape or
                    not all(np.isfinite(a).all() for a in (su, sv, bu, bv))):
                raise ValueError("complete finite C1 line directions required")
            if _shell_frame_variations is not None:
                frame_directions = tuple(np.asarray(a, dtype=float)
                                         for a in _shell_frame_variations)
                if (len(frame_directions) != 3 or any(
                        a.shape != frames.shape or not np.isfinite(a).all()
                        for a in frame_directions)):
                    raise ValueError("complete E4 frame variations required")
        elif _shell_frame_variations is not None:
            raise ValueError("E4 frame variations require two directions")
        rows = np.zeros((6, 4 if directional else 1))
        offset_shell = np.array((0., 0., self.top))
        for point in self.points:
            high, low, beam_frame = point.port.pose_pair(beam_state)
            offset_beam = np.array((0., point.y, -.05))
            beam_position = high+low+beam_frame@offset_beam
            shell_position = self._shell(coordinates, frames, point)
            gap = shell_position-beam_position
            qc = beam_frame.T
            values = [qc@gap]
            center_nodes, center_shape, director = self._director(frames, point.x)
            if directional:
                port = point.port.tangent(beam_state)
                pu, pv = port@bu, port@bv
                beam_position_uv, frame_uv = point.port.second_variation(
                    beam_state, bu, bv)
                frame_u = skew(pu[3:])@beam_frame
                frame_v = skew(pv[3:])@beam_frame
                qc_u, qc_v, qc_uv = frame_u.T, frame_v.T, frame_uv.T
                shell_u = np.zeros(3)
                shell_v = np.zeros(3)
                shell_uv = np.zeros(3)
                for node, shape in zip(point.nodes, point.shape):
                    node = int(node)
                    directed = frames[node]@offset_shell
                    if _shell_frame_variations is None:
                        a, b = skew(su[node, 3:]), skew(sv[node, 3:])
                        position_u, position_v = a@directed, b@directed
                        shell_position_uv = .5*(a@b+b@a)@directed
                    else:
                        fu, fv, fuv = frame_directions
                        position_u = fu[node]@offset_shell
                        position_v = fv[node]@offset_shell
                        shell_position_uv = fuv[node]@offset_shell
                    shell_u += shape*(su[node, :3]+position_u)
                    shell_v += shape*(sv[node, :3]+position_v)
                    shell_uv += shape*shell_position_uv
                beam_u = pu[:3]+frame_u@offset_beam
                beam_v = pv[:3]+frame_v@offset_beam
                beam_uv = beam_position_uv+frame_uv@offset_beam
                gap_u, gap_v = shell_u-beam_u, shell_v-beam_v
                values.extend((qc_u@gap+qc@gap_u,
                               qc_v@gap+qc@gap_v,
                               qc_uv@gap+qc_u@gap_v+qc_v@gap_u+
                               qc@(shell_uv-beam_uv)))
                director_u = np.zeros(3)
                director_v = np.zeros(3)
                director_uv = np.zeros(3)
                for node, shape in zip(center_nodes, center_shape):
                    node = int(node)
                    directed = frames[node, :, 2]
                    if _shell_frame_variations is None:
                        a, b = skew(su[node, 3:]), skew(sv[node, 3:])
                        change_u, change_v = a@directed, b@directed
                        change_uv = .5*(a@b+b@a)@directed
                    else:
                        fu, fv, fuv = frame_directions
                        change_u = fu[node, :, 2]
                        change_v = fv[node, :, 2]
                        change_uv = fuv[node, :, 2]
                    director_u += shape*change_u
                    director_v += shape*change_v
                    director_uv += shape*change_uv
                pitch = self._pitch(
                    beam_frame, director, frame_u, frame_v, frame_uv,
                    director_u, director_v, director_uv)
            else:
                pitch = self._pitch(beam_frame, director)
            for column, value in enumerate(values):
                rows[:3, column] += point.weight*value
                rows[3, column] += (
                    point.weight*point.y*value[2]/self.transverse_inertia)
                rows[4, column] += point.weight*pitch[column]
                rows[5, column] -= (
                    point.weight*point.y*value[0]/self.transverse_inertia)
        return rows

    def evaluate_e4_total(self, total_u: np.ndarray, beam_state,
                          first=None, second=None) -> np.ndarray:
        """Pull the line law through E4's additive total rotation chart."""
        values = np.asarray(total_u, dtype=float)
        if values.shape != (self.shell_size,) or not np.isfinite(values).all():
            raise ValueError("complete finite E4 total coordinates required")
        nodal = values.reshape(self.shell_count, 6)
        coordinates = self.reference_nodes+nodal[:, :3]
        frames = np.array([rotation_matrix_from_vector(theta)
                           for theta in nodal[:, 3:]])
        if first is None and second is None:
            return self.evaluate(coordinates, frames, beam_state)
        if first is None or second is None:
            raise ValueError("two E4 total-coordinate directions required")
        if (not isinstance(first, (tuple, list)) or len(first) != 2 or
                not isinstance(second, (tuple, list)) or len(second) != 2):
            raise ValueError("E4 and B3 direction pairs required")
        su = np.asarray(first[0], dtype=float).reshape(self.shell_count, 6)
        sv = np.asarray(second[0], dtype=float).reshape(self.shell_count, 6)
        if not np.isfinite(su).all() or not np.isfinite(sv).all():
            raise ValueError("finite E4 directions required")
        epsilon = ad.Jet2.variable(0., 0, 2)
        eta = ad.Jet2.variable(0., 1, 2)
        frame_u = np.zeros_like(frames)
        frame_v = np.zeros_like(frames)
        frame_uv = np.zeros_like(frames)
        for node in range(self.shell_count):
            spin = [ad.Jet2.constant(nodal[node, 3+i], 2)+
                    epsilon*su[node, 3+i]+eta*sv[node, 3+i]
                    for i in range(3)]
            jets = ad.so3_exp(spin)
            for i in range(3):
                for j in range(3):
                    entry = jets[i][j]
                    if abs(entry.value-frames[node, i, j]) > 1e-11:
                        raise AssertionError("E4 total rotation charts disagree")
                    frame_u[node, i, j] = entry.gradient[0]
                    frame_v[node, i, j] = entry.gradient[1]
                    frame_uv[node, i, j] = entry.hessian[0, 1]
        return self.evaluate(
            coordinates, frames, beam_state, first, second,
            _shell_frame_variations=(frame_u, frame_v, frame_uv))

    def reference_trace(self) -> np.ndarray:
        """Analytic six-row Jacobian at the physical C1 reference state."""
        shell_size = self.shell_size
        total = shell_size+self.beam.context.count
        trace = np.zeros((6, total))
        frames = np.broadcast_to(np.eye(3),
                                 (self.shell_count, 3, 3))
        state = self.beam.context.initial.mechanical
        station_ports = {}
        station_directors = {}
        for point in self.points:
            key = (point.port.element_index, point.port.cell,
                   point.port.weight)
            if key not in station_ports:
                _, _, frame = point.port.pose_pair(state)
                station_ports[key] = (frame, point.port.tangent(state))
            frame, port = station_ports[key]
            qc = frame.T
            shell_map = np.zeros((3, total))
            for node, shape in zip(point.nodes, point.shape):
                start = 6*int(node)
                shell_map[:, start:start+3] += shape*np.eye(3)
                shell_map[:, start+3:start+6] -= shape*skew(
                    np.array((0., 0., self.top)))
            beam_map = np.column_stack((
                np.eye(3), -skew(frame@np.array((0., point.y, -.05)))))@port
            shell_map[:, shell_size:] -= beam_map
            gap_map = qc@shell_map
            trace[:3] += point.weight*gap_map
            trace[3] += point.weight*point.y*gap_map[2]/self.transverse_inertia
            trace[5] -= point.weight*point.y*gap_map[0]/self.transverse_inertia

            if point.x not in station_directors:
                station_directors[point.x] = self._director(frames, point.x)
            nodes, shape, director = station_directors[point.x]
            a, c = float(frame[:, 0]@director), float(frame[:, 2]@director)
            denom = a*a+c*c
            if c <= .2:
                raise ValueError("C1 reference director pitch left its chart")
            for node, weight in zip(nodes, shape):
                start = 6*int(node)+3
                dd = -skew(frames[int(node), :, 2])
                da = frame[:, 0]@dd
                dc = frame[:, 2]@dd
                trace[4, start:start+3] += (
                    point.weight*weight*(c*da-a*dc)/denom)
            beam_pitch = (c*np.cross(frame[:, 0], director)-
                          a*np.cross(frame[:, 2], director))/denom
            trace[4, shell_size:] += point.weight*beam_pitch@port[3:]
        return trace


class FiniteSurfaceTopFaceLine(FiniteSurfaceLineHybrid):
    """Segmented top-face positional work, distinct from the pitch weld.

    This bounded candidate keeps the parent patch's six position-moment
    functionals over each selected longitudinal interval. Its pitch action
    comes from the longitudinal arm of top-face position, never from an
    independent shell-director tie.
    """

    def __init__(self, xs: np.ndarray, ys: np.ndarray, beam, **kwargs):
        super().__init__(xs, ys, beam, **kwargs)
        self.moment_center_x = float(sum(p.weight*p.x for p in self.points))
        self.inertia = np.zeros((3, 3))
        for point in self.points:
            arm = np.array((point.x-self.moment_center_x, point.y, 0.))
            self.inertia += point.weight*((arm@arm)*np.eye(3)-np.outer(arm, arm))
        if np.linalg.eigvalsh(self.inertia)[0] <= 1e-8:
            raise ValueError("top-face line interval has degenerate footprint inertia")
        element, cell, fraction, _ = beam._locate(self.moment_center_x)
        self.center = NativeHalfCellSpatialPort(
            beam.context, element, cell, fraction)

    def evaluate(self, shell_coordinates: np.ndarray, shell_frames: np.ndarray,
                 beam_state, first=None, second=None, *,
                 _shell_frame_variations=None):
        return FiniteSurfaceMomentPatch.evaluate(
            self, shell_coordinates, shell_frames, beam_state, first, second,
            _shell_frame_variations=_shell_frame_variations)

    def reference_trace(self) -> np.ndarray:
        """Six positional mean/moment rows in the centered material frame."""
        shell_size = self.shell_size
        total = shell_size+self.beam.context.count
        mean = np.zeros((3, total))
        moment = np.zeros((3, total))
        state = self.beam.context.initial.mechanical
        _, _, center_frame = self.center.pose_pair(state)
        qc = center_frame.T
        for point in self.points:
            _, _, frame = point.port.pose_pair(state)
            port = point.port.tangent(state)
            gap_map = np.zeros((3, total))
            for node, shape in zip(point.nodes, point.shape):
                start = 6*int(node)
                gap_map[:, start:start+3] += shape*np.eye(3)
                gap_map[:, start+3:start+6] -= shape*skew(
                    np.array((0., 0., self.top)))
            beam_offset = frame@np.array((0., point.y, -.05))
            gap_map[:, shell_size:] -= np.column_stack((
                np.eye(3), -skew(beam_offset)))@port
            material_gap = qc@gap_map
            arm = np.array((point.x-self.moment_center_x, point.y, 0.))
            mean += point.weight*material_gap
            moment += point.weight*skew(arm)@material_gap
        return np.vstack((mean, np.linalg.solve(self.inertia, moment)))


def c1_hybrid_line_space(xs: np.ndarray, ys: np.ndarray, beam, *,
                         kind: str,
                         intervals: int = 2,
                         thickness: float = .035,
                         quadrature_order: int = 3,
                         mixed_s3: bool = False,
                         mixed_s3_diagonal: str = "bl_tr",
                         ) -> tuple[np.ndarray, np.ndarray, list[FiniteSurfaceLineHybrid]]:
    """Bounded P0/P1/dual rows of one six-channel C1 reference work law.

    Each row is a basis-weighted *mean* gap. Its physical P0/P1/dual
    density coefficient is the integrated row multiplier divided by the
    returned basis integral in metres. Multiplier knots coincide with shell
    cuts; beam cuts are independently segmented by the patch constructor.
    """
    from scripts._b3_e4_f03_weak_line_candidate import LineMultiplierSpace

    if (thickness != .035 or beam.n_half_cells not in (12, 14, 18, 24, 26) or
            not beam.model.name.endswith("-c1_tangent_2um")):
        raise ValueError("weighted line spaces require the bounded 35 mm C1 reference")
    if type(intervals) is not int or intervals not in (2, 4, 8, 16):
        raise ValueError("bounded 2/4/8/16 longitudinal multiplier intervals required")
    knots = np.linspace(0., 1., intervals+1)
    if any(not np.any(np.isclose(xs, .9+.1*value, rtol=0., atol=1e-12))
           for value in knots):
        raise ValueError("multiplier knots must coincide with shell host cuts")
    space = LineMultiplierSpace(kind, tuple(knots))
    patches = []
    for index in range(space.count):
        def basis(x: float, row: int = index) -> float:
            return float(space.values((x-.9)/.1)[row])

        patches.append(FiniteSurfaceLineHybrid(
            xs, ys, beam, thickness=thickness, x_weight=basis,
            quadrature_order=quadrature_order, mixed_s3=mixed_s3,
            mixed_s3_diagonal=mixed_s3_diagonal))
    rows = np.vstack([patch.reference_trace() for patch in patches])
    masses = np.array([patch.basis_integral_m for patch in patches])
    if not (np.isfinite(rows).all() and np.isfinite(masses).all() and
            np.all(masses > 0.)):
        raise ArithmeticError("C1 line multiplier work is not finite")
    return rows, masses, patches


def finite_chart_screen(reference_profile: str = "straight", *,
                        hybrid: bool = False,
                        x_interval: tuple[float, float] = (.9, 1.)) -> dict:
    """One deterministic, bounded nonzero-gap and nonzero-multiplier proof."""
    from scripts._b3_e4_f03_native_fixture import build
    from scripts.b3_e4_f03_square_section_control import square_torsion_constant

    if reference_profile not in ("straight", "c1_tangent_2um"):
        raise ValueError("bounded straight or C1 chart reference required")
    beam = build(4 if reference_profile == "straight" else 12,
                 z_offset=.0675,
                 torsion_constant=square_torsion_constant(),
                 reference_profile=reference_profile)
    patch_class = FiniteSurfaceLineHybrid if hybrid else FiniteSurfaceMomentPatch
    if hybrid:
        from scripts.b3_e4_f03_plate_patch_work import local_grid
        xs, ys = local_grid(2)
    else:
        xs, ys = np.linspace(-.5, 1.5, 5), np.linspace(-.5, .5, 3)
    patch = patch_class(
        xs, ys, beam,
        thickness=.035, **({"x_interval": x_interval} if hybrid else {}))
    count = patch.shell_count
    shell0 = patch.reference_nodes.copy()
    frame0 = np.broadcast_to(np.eye(3), (count, 3, 3)).copy()
    beam0 = beam.context.initial.mechanical
    reference = patch.evaluate(shell0, frame0, beam0)[:, 0]
    rng = np.random.default_rng(1729)
    shell_base = rng.normal(size=(count, 6)) * np.array((.001, .001, .001, .025, .025, .025))
    beam_base = np.zeros(beam.context.count)
    beam_base[6:beam.context.nodal_count] = rng.normal(
        size=beam.context.nodal_count-6) * .001
    for element in range(len(beam.context.elements)):
        begin = beam.context.nodal_count+24*element
        beam_base[begin:begin+6] = rng.normal(size=6)*.02
    coordinates, frames, state = patch.advance(
        shell0, frame0, beam0, shell_base, beam_base)
    shell_u = rng.normal(size=(count, 6))*np.array((.002, .002, .002, .04, .04, .04))
    shell_v = rng.normal(size=(count, 6))*np.array((.002, .002, .002, .04, .04, .04))
    beam_u = rng.normal(size=beam.context.count)*.001
    beam_v = rng.normal(size=beam.context.count)*.001
    beam_u[:6] = 0.
    beam_v[:6] = 0.
    for element in range(len(beam.context.elements)):
        begin = beam.context.nodal_count+24*element
        beam_u[begin:begin+6] *= 30
        beam_v[begin:begin+6] *= 30
    u, v = (shell_u, beam_u), (shell_v, beam_v)
    derivatives = patch.evaluate(coordinates, frames, state, u, v)
    multipliers = np.array((18., -23., 31., 7., -100., 13.))

    def work(a, b):
        c, f, s = patch.advance(
            coordinates, frames, state, a*shell_u+b*shell_v,
            a*beam_u+b*beam_v)
        return float(multipliers @ patch.evaluate(c, f, s)[:, 0])

    h1 = 1e-5
    h2 = 2e-3
    first_error = max(
        abs(float(multipliers@derivatives[:, 1])-(work(h1, 0)-work(-h1, 0))/(2*h1)),
        abs(float(multipliers@derivatives[:, 2])-(work(0, h1)-work(0, -h1))/(2*h1)))
    mixed = float(multipliers@derivatives[:, 3])
    mixed_fd = (work(h2, h2)-work(h2, -h2)-work(-h2, h2)+work(-h2, -h2))/(4*h2*h2)
    mixed_error = abs(mixed-mixed_fd)

    # The reference Jacobian must be the screened six-row linear operator.
    if hybrid:
        trace = patch.reference_trace()
    else:
        trace, _ = distributed_surface_patch_trace(
            patch.xs, patch.ys, beam, plate_thickness=.035)
    reference_direction = np.r_[shell_u.ravel(), beam_u]
    reference_first = patch.evaluate(shell0, frame0, beam0, u, v)[:, 1]
    reference_error = float(np.max(np.abs(reference_first-trace@reference_direction)))

    # Full finite common rigid motion, including retained cell frames.
    omega = np.array((.07, -.05, .04))
    translation = np.array((.003, -.002, .004))
    rigid = rotation_exponential(omega)
    rigid_shell = coordinates @ rigid.T + translation
    rigid_frames = rigid @ frames
    rigid_step = np.zeros(beam.context.count)
    for node in range(beam.context.nodal_count//6):
        start = 6*node
        position = state.positions[node]+state.position_low[node]
        rigid_step[start:start+3] = rigid@position+translation-position
        rigid_step[start+3:start+6] = omega
    for element in range(len(beam.context.elements)):
        begin = beam.context.nodal_count+24*element
        rigid_step[begin:begin+6] = np.tile(omega, 2)
    # Only terminal half-cell nodes enter this patch. Keep the fixture's
    # fixed root untouched while transforming every participating pose.
    rigid_step[:6] = 0.
    rigid_state = beam.context.advance(state, rigid_step)
    rigid_error = float(np.max(np.abs(
        patch.evaluate(rigid_shell, rigid_frames, rigid_state)[:, 0]-derivatives[:, 0])))
    rigid_shell_direction = np.zeros((count, 6))
    rigid_beam_direction = np.zeros(beam.context.count)
    for node in range(count):
        rigid_shell_direction[node, :3] = (
            translation+np.cross(omega, coordinates[node]))
        rigid_shell_direction[node, 3:] = omega
    for node in range(1, beam.context.nodal_count//6):
        start = 6*node
        position = state.positions[node]+state.position_low[node]
        rigid_beam_direction[start:start+3] = (
            translation+np.cross(omega, position))
        rigid_beam_direction[start+3:start+6] = omega
    for element in range(len(beam.context.elements)):
        begin = beam.context.nodal_count+24*element
        rigid_beam_direction[begin:begin+6] = np.tile(omega, 2)
    rigid_direction = (rigid_shell_direction, rigid_beam_direction)
    rigid_virtual_work = float(multipliers @ patch.evaluate(
        coordinates, frames, state, rigid_direction, rigid_direction)[:, 1])
    return {
        "reference_profile": reference_profile,
        "hybrid": hybrid,
        "x_interval": list(x_interval) if hybrid else None,
        "reference_gap_max_m_rad": float(np.max(np.abs(reference))),
        "reference_jacobian_direction_error": reference_error,
        "nonzero_gap_norm_m_rad": float(np.linalg.norm(derivatives[:, 0])),
        "nonzero_multiplier_mixed_work_j": mixed,
        "first_direction_error_j": first_error,
        "mixed_direction_error_j": mixed_error,
        "finite_common_rigid_gap_error": rigid_error,
        "common_rigid_virtual_work_j": rigid_virtual_work,
        "quadrature_points": len(patch.points),
        "scope": ("spatial-left E4 director and retained B3 half-cell charts; "
                  "bounded single-interval six-channel hybrid" if hybrid else
                  "spatial-left E4 director and retained B3 half-cell charts; "
                  "bounded finite six-resultant surface-position law"),
    }


def bounded_physical_compliance_screen(*, hybrid: bool = False,
                                       segmented: bool = False,
                                       c1_beam_half_cells: int = 12,
                                       c1_characteristic_length_m: float = 1.,
                                       line_space_kind: str | None = None,
                                       sparse_action: bool = False,
                                       include_shell_eight: bool = False,
                                       axial_shell_cells: tuple[int, ...] | None = None,
                                       reciprocal_shell_x_cells: int | None = None,
                                       mixed_s3: bool = False,
                                       mixed_s3_diagonal: str = "bl_tr",
                                       ) -> list[dict]:
    """Compare six physical interface actions on several bounded local grids.

    This is a finite-dimensional, support-conditioned compliance diagnostic,
    not an inf-sup constant or a choice of a true-line multiplier space.
    The positive-energy Gram is formed from actual retained B3 and E4
    operators without symmetrizing either operator to repair its tangent.
    """
    from scipy import linalg, sparse

    from scripts._b3_e4_f03_native_fixture import build
    from scripts.b3_e4_f03_plate_patch_work import local_grid, raw_e4_tangent
    from scripts.b3_e4_f03_square_section_control import square_torsion_constant

    results = []
    if segmented and not hybrid:
        raise ValueError("segmented diagnostic requires the hybrid law")
    if line_space_kind is not None and (
            not hybrid or not segmented or line_space_kind not in (
                "piecewise_constant", "continuous_linear", "dual_linear")):
        raise ValueError("weighted C1 line space requires a segmented hybrid law")
    if include_shell_eight and not (hybrid and segmented and sparse_action and
                                    line_space_kind is not None):
        raise ValueError("eight-cell C1 screen requires sparse weighted line action")
    if axial_shell_cells is not None and (
            axial_shell_cells != (4, 8, 16) or include_shell_eight or
            not (hybrid and segmented and sparse_action and
                 line_space_kind is not None)):
        raise ValueError("axial-only C1 screen requires bounded sparse weighted cells")
    if reciprocal_shell_x_cells is not None and (
            reciprocal_shell_x_cells != 2 or axial_shell_cells is not None or
            include_shell_eight or not (hybrid and segmented and sparse_action
                                        and line_space_kind is not None)):
        raise ValueError("reciprocal C1 screen requires bounded sparse weighted cells")
    if mixed_s3 and not (hybrid and segmented and sparse_action and
                         line_space_kind is not None and
                         c1_beam_half_cells == 12 and
                         not include_shell_eight and
                         axial_shell_cells is None and
                         reciprocal_shell_x_cells is None):
        raise ValueError("mixed C1 Q4/S3 requires bounded sparse weighted cells")
    if (mixed_s3_diagonal not in ("bl_tr", "br_tl") or
            not mixed_s3 and mixed_s3_diagonal != "bl_tr"):
        raise ValueError("mixed C1 S3 diagonal requires bounded mixed case")
    if c1_beam_half_cells not in (12, 14, 18, 24, 26) or (
            not hybrid and c1_beam_half_cells != 12):
        raise ValueError("bounded C1 beam resolution requires hybrid 12/14/18/24/26")
    if c1_beam_half_cells in (14, 18, 26) and reciprocal_shell_x_cells is None:
        raise ValueError("fine C1 beam resolution requires reciprocal screen")
    if (c1_characteristic_length_m not in (.1, 1.) or
            not hybrid and c1_characteristic_length_m != 1.):
        raise ValueError("bounded C1 moment scale must be 0.1 or 1 m")
    shell_operators = {}
    cases = ((2, c1_beam_half_cells), (4, c1_beam_half_cells)) if hybrid else (
        (2, 4), (2, 8), (4, 4), (4, 8), (4, 16))
    if include_shell_eight:
        cases += ((8, c1_beam_half_cells),)
    if axial_shell_cells is not None:
        cases = tuple((cells, c1_beam_half_cells)
                      for cells in axial_shell_cells)
    if reciprocal_shell_x_cells is not None:
        cases = ((reciprocal_shell_x_cells, c1_beam_half_cells),)
    if mixed_s3:
        cases = ((2, c1_beam_half_cells),)
    for patch_cells, half_cells in cases:
        ys_cells = (4 if axial_shell_cells is not None or
                    reciprocal_shell_x_cells is not None or mixed_s3
                    else patch_cells)
        xs, ys = local_grid(patch_cells, transverse_cells=ys_cells)
        beam = build(half_cells, z_offset=.0675,
                     torsion_constant=square_torsion_constant(),
                     reference_profile=("c1_tangent_2um" if hybrid
                                        else "straight"))
        if patch_cells not in shell_operators:
            from scripts.b3_e4_f03_plate_patch_work import (
                raw_e4_mixed_c1_tangent,
            )
            shell_operators[patch_cells] = (
                raw_e4_mixed_c1_tangent(
                    xs,ys,thickness=.035,diagonal=mixed_s3_diagonal)
                if mixed_s3 else raw_e4_tangent(xs,ys,thickness=.035))
        shell_k, shell_physical = shell_operators[patch_cells]
        if hybrid:
            if line_space_kind is None:
                intervals = (((.9, .95), (.95, 1.)) if segmented else
                             ((.9, 1.),))
                patches = [FiniteSurfaceLineHybrid(
                    xs, ys, beam, thickness=.035, x_interval=interval)
                    for interval in intervals]
                rows = np.vstack([patch.reference_trace() for patch in patches])
                basis_integrals = np.full(len(patches), .1/len(patches))
            else:
                rows, basis_integrals, patches = c1_hybrid_line_space(
                    xs, ys, beam, kind=line_space_kind,
                    mixed_s3=mixed_s3,
                    mixed_s3_diagonal=mixed_s3_diagonal)
            reference_gap = max(float(np.max(np.abs(patch.evaluate(
                patch.reference_nodes,
                np.broadcast_to(np.eye(3),
                                (patch.shell_count, 3, 3)),
                beam.context.initial.mechanical)[:, 0])))
                                for patch in patches)
        else:
            rows, reference = distributed_surface_patch_trace(
                xs, ys, beam, plate_thickness=.035)
            reference_gap = reference["reference_gap_m"]
        actions = np.diag(np.tile(
            (1000., 1000., 1000., 100., 100., 100.), len(rows)//6))
        shell_size = shell_k.shape[0]
        total = shell_size+beam.context.count
        if (max(shell_size, beam.context.count) > 1900 or
                total > 1900 and not sparse_action):
            raise ValueError("bounded local compliance matrix cap exceeded")
        fixed_shell = {6*j*len(xs)+axis for j in range(len(ys))
                       for axis in range(6)}
        fixed_beam = {shell_size+i for i in beam.context.fixed}
        free = np.array([i for i in range(total)
                         if i not in fixed_shell and i not in fixed_beam])
        shell_free = free[free < shell_size]
        beam_free = free[free >= shell_size]-shell_size
        shell_rows = rows[:, shell_free]
        beam_rows = rows[:, shell_size+beam_free]
        shell_block = shell_k[np.ix_(shell_free, shell_free)]
        beam_block = beam.tangent[np.ix_(beam_free, beam_free)].toarray()
        shell_response = (sparse.linalg.splu(
            sparse.csc_matrix(shell_block)).solve(shell_rows.T)
            if sparse_action else linalg.solve(
                shell_block, shell_rows.T, assume_a="gen", check_finite=True))
        beam_response = linalg.solve(
            beam_block, beam_rows.T, assume_a="gen", check_finite=True)
        gram_shell = shell_rows@ shell_response
        gram_beam = beam_rows@ beam_response
        gram = actions@(gram_shell+gram_beam)@actions
        skew_relative = float(np.linalg.norm(gram-gram.T)/max(
            np.linalg.norm(gram), 1e-30))
        if skew_relative > 1e-8:
            raise ArithmeticError("reference physical compliance is nonsymmetric")
        diagonal = np.diag(gram)
        if np.min(diagonal) <= 0.:
            raise ArithmeticError("nonpositive physical interface action")
        energy_scale = np.sqrt(diagonal)
        normalized = gram/np.outer(energy_scale, energy_scale)
        eigenvalues, eigenvectors = np.linalg.eigh(normalized)
        mode = eigenvectors[:, 0]
        shell_action = float(mode@(
            actions@gram_shell@actions/np.outer(energy_scale, energy_scale))@mode)
        beam_action = float(mode@(
            actions@gram_beam@actions/np.outer(energy_scale, energy_scale))@mode)
        action_dual = actions@(mode/energy_scale)
        shell_mode = shell_response@action_dual
        physical_shell_block = shell_physical[np.ix_(shell_free, shell_free)]
        physical_shell_action = float(shell_mode@physical_shell_block@shell_mode)
        numerical_shell_action = shell_action-physical_shell_action
        if (eigenvalues[0] <= 0. or shell_action <= 0. or
                beam_action <= 0. or physical_shell_action <= 0. or
                abs(shell_action+beam_action-eigenvalues[0]) > 1e-8):
            raise ArithmeticError("physical interface mode action failed")
        line_norm = None
        if segmented:
            from scripts._b3_e4_f03_p0_dual_norm import p0_dual_gram
            from scripts._b3_e4_f03_p1_dual_norm import p1_dual_gram
            from scripts._b3_e4_f03_dual_linear_norm import dual_linear_gram

            # Rows are basis-normalized gaps (signed for dual P1); their
            # duals are integrated N/Nm. Convert density coefficients by
            # each basis integral. Scale moments by the declared length.
            bond_length_m = .1
            characteristic_length_m = c1_characteristic_length_m
            density_to_integrated = np.tile(
                np.r_[np.ones(3), np.full(3, characteristic_length_m)],
                len(patches))*np.repeat(basis_integrals, 6)
            gram_builder = {
                None: p0_dual_gram,
                "piecewise_constant": p0_dual_gram,
                "continuous_linear": p1_dual_gram,
                "dual_linear": dual_linear_gram,
            }[line_space_kind]
            lower, upper = gram_builder(2 if line_space_kind else
                                        len(patches), bond_length_m, 4095)
            dual_lower = np.kron(lower, np.eye(6))
            dual_upper = np.kron(upper, np.eye(6))
            stiffness_n_per_m = 210e9*.035
            density_compliance = stiffness_n_per_m*(
                density_to_integrated[:, None]*
                (gram_shell+gram_beam)*density_to_integrated[None, :])
            # The signed diagnostic is a quadratic action. eigh otherwise
            # reads one triangle of the roundoff-skew Gram, while the split
            # work below contracts the full Gram. This does not alter an
            # element tangent, KKT solve, or physical response.
            density_quadratic = .5*(density_compliance+
                                    density_compliance.T)
            signed_upper, vectors = linalg.eigh(
                density_quadratic, dual_upper, check_finite=True)
            signed_lower = linalg.eigh(
                density_quadratic, dual_lower,
                eigvals_only=True, check_finite=True)
            density_mode = vectors[:, 0]
            integrated_mode = density_to_integrated*density_mode
            shell_mode_work = float(stiffness_n_per_m*(
                integrated_mode@gram_shell@integrated_mode))
            beam_mode_work = float(stiffness_n_per_m*(
                integrated_mode@gram_beam@integrated_mode))
            nonaxial = np.array([i for i in range(len(density_to_integrated))
                                 if i % 6 != 0], dtype=int)
            transverse_upper, transverse_vectors = linalg.eigh(
                density_quadratic[np.ix_(nonaxial, nonaxial)],
                dual_upper[np.ix_(nonaxial, nonaxial)], check_finite=True)
            transverse_lower = linalg.eigh(
                density_quadratic[np.ix_(nonaxial, nonaxial)],
                dual_lower[np.ix_(nonaxial, nonaxial)],
                eigvals_only=True, check_finite=True)
            transverse_density_mode = np.zeros(len(density_to_integrated))
            transverse_density_mode[nonaxial] = transverse_vectors[:, 0]
            transverse_integrated = (
                density_to_integrated*transverse_density_mode)
            transverse_shell_action = float(stiffness_n_per_m*(
                transverse_integrated@gram_shell@transverse_integrated))
            transverse_beam_action = float(stiffness_n_per_m*(
                transverse_integrated@gram_beam@transverse_integrated))
            if abs(shell_mode_work+beam_mode_work-signed_upper[0]) > 1e-8*max(
                    1., abs(signed_upper[0])) or abs(
                    transverse_shell_action+transverse_beam_action-
                    transverse_upper[0]) > 1e-8*max(
                        1., abs(transverse_upper[0])):
                raise ArithmeticError("line density action split inconsistent")
            line_norm = {
                "status": "FINITE_GRID_SIGNED_COMPLIANCE_NOT_INF_SUP",
                "bond_length_m": bond_length_m,
                "basis_integrals_m": basis_integrals.tolist(),
                **({"interval_length_m": bond_length_m/len(patches)}
                   if line_space_kind is None else {}),
                "characteristic_length_m": characteristic_length_m,
                "reference_stiffness_n_per_m": stiffness_n_per_m,
                "density_units": "N/m for forces; Nm/m for moments",
                "sobolev_dual": ("analytic Neumann H^-1/2 "
                                 f"{line_space_kind or 'P0'} density Gram"),
                "smallest_signed_quotient_upper": float(signed_upper[0]),
                "smallest_signed_quotient_lower": float(signed_lower[0]),
                "smallest_mode_density_equivalent_n_per_m": density_mode.tolist(),
                "smallest_mode_shell_action": shell_mode_work,
                "smallest_mode_beam_action": beam_mode_work,
                "nonaxial_channels": ["Fy", "Fz", "Mx", "My", "Mz"],
                "smallest_nonaxial_signed_quotient_upper": float(
                    transverse_upper[0]),
                "smallest_nonaxial_signed_quotient_lower": float(
                    transverse_lower[0]),
                "smallest_nonaxial_mode_density_equivalent_n_per_m": (
                    transverse_density_mode.tolist()),
                "smallest_nonaxial_mode_shell_action": transverse_shell_action,
                "smallest_nonaxial_mode_beam_action": transverse_beam_action,
                "maximum_gram_bracket_ratio": float(np.max(linalg.eigvalsh(
                    dual_upper, dual_lower, check_finite=True))),
            }
        results.append({
            "patch_cells_per_side": (patch_cells if axial_shell_cells is None and
                                      reciprocal_shell_x_cells is None and
                                      not mixed_s3
                                      else None),
            "bond_x_cells": patch_cells,
            "bond_y_cells": ys_cells,
            "beam_half_cells": half_cells,
            "shell_nodes": len(xs)*len(ys),
            "total_primal_rows": total,
            "constraint_rank_on_free": int(np.linalg.matrix_rank(rows[:, free])),
            "applied_action_scales_N_Nm": np.diag(actions).tolist(),
            "unit_action_twice_compliances_j": diagonal.tolist(),
            "normalized_mode_minimum": float(eigenvalues[0]),
            "normalized_mode_maximum": float(eigenvalues[-1]),
            "weakest_mode_components": mode.tolist(),
            "weakest_mode_shell_action": shell_action,
            "weakest_mode_physical_shell_action": physical_shell_action,
            "weakest_mode_numerical_shell_action": numerical_shell_action,
            "weakest_mode_beam_action": beam_action,
            "compliance_relative_skew": skew_relative,
            "reference_gap_m": reference_gap,
            "operator": (("C1 two-interval line hybrid" if segmented else
                          "C1 single-interval line hybrid") if hybrid else
                         "finite surface-moment footprint"),
            **({"shell_topology": "Q4/S3-V2D"} if mixed_s3 else {}),
            **({"s3_diagonal": mixed_s3_diagonal} if mixed_s3 else {}),
            "interpretation": "support-conditioned finite-grid physical action; not continuum inf-sup or line-space qualification",
            **({("p0_density_norm" if line_space_kind is None else
                 "line_density_norm"): line_norm} if line_norm is not None else {}),
            **({"line_space_kind": line_space_kind} if line_space_kind else {}),
        })
    return results
