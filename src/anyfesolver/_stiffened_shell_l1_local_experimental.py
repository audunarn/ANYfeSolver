"""Local L1 beam/shell work for an experimental, unqualified line candidate.

Only the 12 shell trace coordinates, two native endpoint translations, and
one retained half-cell rotation enter the local work.  The returned slots
refer to a native retained context; callers supply their global gather map.
This module selects neither a multiplier space nor a production connection.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from . import _ge_beam3_mixed_ad as ad
from ._ge_beam3_retained_generalized_state import Mechanical
from ._ge_beam3_p5.compensated_coordinates import split_sum
from .corotational import rotation_matrix_from_vector


def _owned(value, shape):
    array = np.array(value, dtype=float, copy=True)
    if array.shape != shape or not np.isfinite(array).all():
        raise ValueError("finite L1 local data with declared shape required")
    return np.frombuffer(array.tobytes(), dtype=float).reshape(shape)


def _owned_slots(value, shape):
    array = np.array(value, dtype=np.intp, copy=True)
    if array.shape != shape or np.any(array < 0):
        raise ValueError("nonnegative native L1 gather slots required")
    return np.frombuffer(array.tobytes(), dtype=np.intp).reshape(shape)


def _unit(values, label):
    size = float(np.linalg.norm(values))
    if not np.isfinite(size) or size <= 1e-10:
        raise ValueError(f"degenerate {label}")
    return values/size


def _jet_unit(values, label):
    squared = ad.dot(values, values)
    if not np.isfinite(squared.value) or squared.value <= 1e-20:
        raise ValueError(f"degenerate {label}")
    length = ad.sqrt(squared)
    return [value/length for value in values]


def _jet_cross(left, right):
    return [left[1]*right[2]-left[2]*right[1],
            left[2]*right[0]-left[0]*right[2],
            left[0]*right[1]-left[1]*right[0]]


@dataclass(frozen=True)
class NativeHalfCellPort:
    """Physical one-sided station of a native B3 retained half-cell."""

    nodes: tuple[int, int]
    element_index: int
    cell: int
    weight: float
    nodal_count: int
    retained_count: int
    lift: np.ndarray
    frame: np.ndarray

    @classmethod
    def from_reference(cls, reference, element_nodes, element_index, cell,
                       weight, nodal_count, retained_count):
        if (type(element_nodes) not in (tuple, list) or len(element_nodes) != 3 or
                type(cell) is not int or cell not in (0, 1) or
                isinstance(weight, (bool, str)) or
                not isinstance(weight, (int, float, np.floating)) or
                not np.isfinite(weight) or not 0. <= weight <= 1.):
            raise ValueError("three native element nodes and half-cell station required")
        lift = reference.half_cell_lift(cell, weight)
        frame = reference.frame(cell-1+weight)
        return cls(tuple(element_nodes[cell:cell+2]), element_index, cell,
                   float(weight), nodal_count, retained_count,
                   _owned(lift, (3,)), _owned(frame, (3, 3)))

    def __post_init__(self):
        if (type(self.nodes) is not tuple or len(self.nodes) != 2 or
                any(type(v) is not int or v < 0 for v in self.nodes) or
                self.nodes[0] == self.nodes[1] or
                type(self.element_index) is not int or self.element_index < 0 or
                type(self.cell) is not int or self.cell not in (0, 1) or
                isinstance(self.weight, (bool, str)) or
                not isinstance(self.weight, (int, float, np.floating)) or
                not math.isfinite(self.weight) or not 0. <= self.weight <= 1. or
                type(self.nodal_count) is not int or self.nodal_count % 6 or
                max(self.nodes) >= self.nodal_count//6 or
                type(self.retained_count) is not int or
                (self.retained_count-self.nodal_count) % 24 or
                self.retained_count < self.nodal_count+24*(self.element_index+1)):
            raise ValueError("valid native half-cell station required")

        lift = _owned(self.lift, (3,))
        frame = _owned(self.frame, (3, 3))
        if (np.linalg.norm(frame.T@frame-np.eye(3)) > 1e-10 or
                np.linalg.det(frame) < 0.):
            raise ValueError("physical native reference station frame required")
        object.__setattr__(self, "lift", lift)
        object.__setattr__(self, "frame", frame)

    def parts(self, state):
        if not isinstance(state, Mechanical):
            raise TypeError("native mechanical state required")
        positions = np.asarray(state.positions, dtype=float)
        lows = np.asarray(state.position_low, dtype=float)
        cells = np.asarray(state.cell_rotations, dtype=float)
        if (positions.shape != (self.nodal_count//6, 3) or
                lows.shape != positions.shape or
                cells.shape != ((self.retained_count-self.nodal_count)//24,
                                2, 3, 3)):
            raise ValueError("native retained trace state dimensions required")
        left, right = self.nodes
        left_high = _owned(positions[left], (3,))
        left_low = _owned(lows[left], (3,))
        right_high = _owned(positions[right], (3,))
        right_low = _owned(lows[right], (3,))
        rotation = _owned(cells[self.element_index, self.cell], (3, 3))
        if (np.linalg.norm(rotation.T@rotation-np.eye(3)) > 1e-9 or
                np.linalg.det(rotation) < 0.):
            raise ValueError("physical native retained cell rotation required")
        return left_high, left_low, right_high, right_low, rotation

    def pose_pair(self, state):
        return self._pose_from_parts(self.parts(state))

    def _pose_from_parts(self, parts):
        left_high, left_low, right_high, right_low, rotation = parts
        displaced = rotation@self.lift
        pairs = []
        for axis in range(3):
            high = float(left_high[axis])
            low = float(left_low[axis])
            delta_high = float(right_high[axis]-high)
            delta_low = float(right_low[axis]-low)
            pairs.append(split_sum((high, low, float(self.weight*delta_high),
                                    float(self.weight*delta_low),
                                    float(displaced[axis]))))
        position_high, position_low = np.asarray(pairs, dtype=float).T
        return position_high.copy(), position_low.copy(), rotation@self.frame

    def active_beam_slots(self):
        left, right = self.nodes
        cell_slot = self.nodal_count+24*self.element_index+3*self.cell
        return np.r_[np.arange(6*left, 6*left+3),
                     np.arange(6*right, 6*right+3),
                     np.arange(cell_slot, cell_slot+3)]


@dataclass(frozen=True)
class LocalWork:
    value: float
    gap: np.ndarray
    gap_jacobian: np.ndarray
    gradient: np.ndarray
    hessian: np.ndarray
    beam_slots: np.ndarray


@dataclass(frozen=True)
class L1LocalOperator:
    """Shell-frame reference-offset Log work in a 21-coordinate local chart."""

    reference: np.ndarray
    thickness: float
    port: NativeHalfCellPort
    offset: np.ndarray
    shell_base_high: np.ndarray
    shell_base_low: np.ndarray
    reference_offset: np.ndarray
    reference_relative: np.ndarray
    station: float

    @classmethod
    def at_reference(cls, reference, total, thickness, port, state, *,
                     offset=(0., 0., -.05), station=.5):
        if not isinstance(port, NativeHalfCellPort):
            raise TypeError("native half-cell port required")
        reference = _owned(reference, (2, 3))
        total = _owned(total, (2, 6))
        offset = _owned(offset, (3,))
        if (not np.isfinite(thickness) or thickness <= 0. or
                not np.isfinite(station) or not 0. <= station <= 1.):
            raise ValueError("positive thickness and material station required")
        directors = np.array([rotation_matrix_from_vector(row[3:])[:, 2]
                              for row in total])
        top_delta = total[:, :3]+.5*thickness*directors
        base_pair = np.array([
            split_sum((float(reference[0, j]),
                       float(station*(reference[1, j]-reference[0, j]))))
            for j in range(3)])
        center_delta = (1.-station)*top_delta[0]+station*top_delta[1]
        center_pair = np.array([
            split_sum((float(base_pair[j, 0]), float(base_pair[j, 1]),
                       float(center_delta[j]))) for j in range(3)])
        normal = _unit((1.-station)*directors[0]+station*directors[1],
                       "E4 mean director")
        tangent_raw = np.array([
            math.fsum((float(reference[1, j]-reference[0, j]),
                       float(top_delta[1, j]), -float(top_delta[0, j])))
            for j in range(3)])
        tangent = _unit(tangent_raw-normal*(normal@tangent_raw),
                        "projected E4 material-line tangent")
        shell_frame = np.column_stack((tangent, np.cross(normal, tangent), normal))
        high, low, beam_frame = port.pose_pair(state)
        arm = beam_frame@offset
        distance = np.array([
            math.fsum((float(high[j]), float(low[j]), float(arm[j]),
                       -float(center_pair[j, 0]), -float(center_pair[j, 1])))
            for j in range(3)])
        reference_offset = shell_frame.T@distance
        reference_relative = shell_frame.T@beam_frame
        return cls(reference, float(thickness), port, offset,
                   _owned(base_pair[:, 0], (3,)),
                   _owned(base_pair[:, 1], (3,)),
                   _owned(reference_offset, (3,)),
                   _owned(reference_relative, (3, 3)),
                   float(station))

    def evaluate(self, total, state, multiplier):
        """Return exact local first and second variation of multiplier work."""
        total = _owned(total, (2, 6))
        multiplier = _owned(multiplier, (6,))
        parts = self.port.parts(state)
        rotation = parts[-1]
        high, low, beam_frame0 = self.port._pose_from_parts(parts)
        size = 21
        constant = lambda value: ad.Jet2.constant(float(value), size)
        nodal = [[ad.Jet2.variable(total[i, j], 6*i+j, size)
                  for j in range(6)] for i in range(2)]
        ez = [constant(0.), constant(0.), constant(1.)]
        directors = [ad.matvec(ad.so3_exp(row[3:]), ez) for row in nodal]
        top_delta = [[nodal[i][j]+.5*self.thickness*directors[i][j]
                      for j in range(3)] for i in range(2)]
        shell_delta = [(1.-self.station)*top_delta[0][j]+
                       self.station*top_delta[1][j] for j in range(3)]
        mean = [(1.-self.station)*directors[0][j]+self.station*directors[1][j]
                for j in range(3)]
        normal = _jet_unit(mean, "E4 mean director")
        raw = [constant(self.reference[1, j]-self.reference[0, j])+
               top_delta[1][j]-top_delta[0][j] for j in range(3)]
        component = ad.dot(normal, raw)
        tangent = _jet_unit([raw[j]-normal[j]*component for j in range(3)],
                            "projected E4 material-line tangent")
        side = _jet_cross(normal, tangent)
        shell_frame = [[tangent[j], side[j], normal[j]] for j in range(3)]
        spin = [ad.Jet2.variable(0., 18+j, size) for j in range(3)]
        increment = ad.so3_exp(spin)
        beam_frame = ad.matmul(increment, ad.constant_matrix(beam_frame0, size))
        rotated_lift = ad.matvec(increment,
                                 [constant(v) for v in rotation@self.port.lift])
        beam_origin_relative = [constant(math.fsum((
                            float(high[j]), float(low[j]),
                            -float(self.shell_base_high[j]),
                            -float(self.shell_base_low[j]))))+
                       (1.-self.port.weight)*ad.Jet2.variable(0., 12+j, size)+
                       self.port.weight*ad.Jet2.variable(0., 15+j, size)+
                       rotated_lift[j]-constant((rotation@self.port.lift)[j])
                       for j in range(3)]
        arm = ad.matvec(beam_frame, [constant(v) for v in self.offset])
        distance = [beam_origin_relative[j]+arm[j]-shell_delta[j]
                    for j in range(3)]
        translation = [entry-constant(offset) for entry, offset in zip(
            ad.matvec(ad.transpose(shell_frame), distance),
            self.reference_offset)]
        relative = ad.matmul(
            ad.constant_matrix(self.reference_relative.T, size),
            ad.matmul(ad.transpose(shell_frame), beam_frame))
        gaps = translation+ad.so3_log(relative)
        work = sum((constant(multiplier[j])*gaps[j] for j in range(6)),
                   start=constant(0.))
        return LocalWork(
            float(work.value), _owned([gap.value for gap in gaps], (6,)),
            _owned([gap.gradient for gap in gaps], (6, size)),
            _owned(work.gradient, (size,)),
            _owned(work.hessian, (size, size)),
            _owned_slots(self.port.active_beam_slots(), (9,)))
