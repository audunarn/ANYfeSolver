"""Private five-row beam/shell director-bond development candidate.

The shell top position and physical director are tied to a native B3
half-cell pose. Relative spin about the director is deliberately free. This
local operator neither selects a multiplier space nor qualifies a bonded line.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from . import _ge_beam3_mixed_ad as ad
from ._stiffened_shell_l1_local_experimental import (
    LocalWork, NativeHalfCellPort, _jet_unit, _owned, _owned_slots, _unit,
)
from ._ge_beam3_p5.compensated_coordinates import split_sum
from .corotational import rotation_matrix_from_vector


LAW_ID = "BL5_PHYSICAL_DIRECTOR_WEAK_LINE_DEV_V1"
_CHART_MARGIN = 1e-8


class BL5TrialDomainError(ValueError):
    """A finite candidate left the declared director/trace chart."""


@dataclass(frozen=True)
class BL5LocalOperator:
    """Objective position/director work at one material trace station."""

    reference: np.ndarray
    thickness: float
    port: NativeHalfCellPort
    station: float
    shell_base_high: np.ndarray
    shell_base_low: np.ndarray
    reference_arm: np.ndarray
    reference_director: np.ndarray
    director_basis: np.ndarray

    @classmethod
    def at_reference(cls, reference, total, thickness, port, state, *,
                     station=.5):
        if not isinstance(port, NativeHalfCellPort):
            raise TypeError("native half-cell port required")
        reference = _owned(reference, (2, 3))
        total = _owned(total, (2, 6))
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
        center_pair = np.array([
            split_sum((float(base_pair[j, 0]), float(base_pair[j, 1]),
                       float((1.-station)*top_delta[0, j]+station*top_delta[1, j])))
            for j in range(3)])
        director = _unit((1.-station)*directors[0]+station*directors[1],
                         "BL5 physical shell director")
        raw = reference[1]-reference[0]+top_delta[1]-top_delta[0]
        _unit(raw-director*np.dot(director, raw),
              "BL5 projected material-line tangent")
        high, low, beam_frame = port.pose_pair(state)
        distance = np.array([
            math.fsum((float(center_pair[j, 0]), float(center_pair[j, 1]),
                       -float(high[j]), -float(low[j])))
            for j in range(3)])
        arm = beam_frame.T@distance
        nu = _unit(beam_frame.T@director, "BL5 reference director")
        axis = np.eye(3)[int(np.argmin(np.abs(nu)))]
        first = _unit(axis-nu*np.dot(nu, axis), "BL5 director basis")
        basis = np.stack((first, np.cross(nu, first)))
        return cls(reference, float(thickness), port, float(station),
                   _owned(base_pair[:, 0], (3,)),
                   _owned(base_pair[:, 1], (3,)),
                   _owned(arm, (3,)), _owned(nu, (3,)),
                   _owned(basis, (2, 3)))

    def evaluate(self, total, state, multiplier):
        """Return the gap and exact local first/second multiplier-work jets."""
        return self._evaluate(total, state, multiplier)

    def evaluate_spatial(self, total, frames, state, multiplier):
        """Work derivatives for left physical rotations at the current poses.

        Translation values remain reference displacements. Rotation entries in
        total are not interpreted as physical poses: frames own those poses.
        Returned rotational rows/columns are zero-chart spatial variations.
        """
        frames = _owned(frames, (2, 3, 3))
        if (np.max(np.abs(frames.transpose(0, 2, 1)@frames-np.eye(3))) > 1e-9 or
                np.any(np.linalg.det(frames) <= 0.)):
            raise ValueError("physical BL5 shell frames required")
        return self._evaluate(total, state, multiplier, frames=frames)

    def _evaluate(self, total, state, multiplier, *, frames=None):
        total = _owned(total, (2, 6))
        multiplier = _owned(multiplier, (5,))
        parts = self.port.parts(state)
        rotation = parts[-1]
        high, low, beam_frame0 = self.port._pose_from_parts(parts)
        size = 21
        constant = lambda value: ad.Jet2.constant(float(value), size)
        nodal = [[ad.Jet2.variable(0. if frames is not None and j >= 3
                                 else total[i, j], 6*i+j, size)
                  for j in range(6)] for i in range(2)]
        ez = [constant(0.), constant(0.), constant(1.)]
        directors = [ad.matvec(ad.so3_exp(row[3:]), ez) for row in nodal] if frames is None else [
            ad.matvec(ad.so3_exp(row[3:]), [constant(v) for v in frames[i, :, 2]])
            for i, row in enumerate(nodal)]
        top_delta = [[nodal[i][j]+.5*self.thickness*directors[i][j]
                      for j in range(3)] for i in range(2)]
        shell_delta = [(1.-self.station)*top_delta[0][j]+
                       self.station*top_delta[1][j] for j in range(3)]
        mean = [(1.-self.station)*directors[0][j]+self.station*directors[1][j]
                for j in range(3)]
        if frames is not None and ad.dot(mean, mean).value <= 1e-20:
            raise BL5TrialDomainError("degenerate BL5 physical shell director")
        director = _jet_unit(mean, "BL5 physical shell director")
        raw = [constant(self.reference[1, j]-self.reference[0, j])+
               top_delta[1][j]-top_delta[0][j] for j in range(3)]
        normal_value = np.array([item.value for item in director])
        raw_value = np.array([item.value for item in raw])
        projected = raw_value-normal_value*np.dot(normal_value, raw_value)
        if frames is not None and np.linalg.norm(projected) <= 1e-10:
            raise BL5TrialDomainError("degenerate BL5 projected material-line tangent")
        _unit(projected,
              "BL5 projected material-line tangent")

        spin = [ad.Jet2.variable(0., 18+j, size) for j in range(3)]
        increment = ad.so3_exp(spin)
        beam_frame = ad.matmul(increment, ad.constant_matrix(beam_frame0, size))
        rotated_lift = ad.matvec(
            increment, [constant(v) for v in rotation@self.port.lift])
        beam_origin_relative = [constant(math.fsum((
                            float(high[j]), float(low[j]),
                            -float(self.shell_base_high[j]),
                            -float(self.shell_base_low[j]))))+
                       (1.-self.port.weight)*ad.Jet2.variable(0., 12+j, size)+
                       self.port.weight*ad.Jet2.variable(0., 15+j, size)+
                       rotated_lift[j]-constant((rotation@self.port.lift)[j])
                       for j in range(3)]
        relative_position = [shell_delta[j]-beam_origin_relative[j]
                             for j in range(3)]
        translation = [entry-constant(arm) for entry, arm in zip(
            ad.matvec(ad.transpose(beam_frame), relative_position),
            self.reference_arm)]
        director_beam = ad.matvec(ad.transpose(beam_frame), director)
        alignment = ad.dot(director_beam,
                           [constant(v) for v in self.reference_director])
        if not np.isfinite(alignment.value) or alignment.value <= _CHART_MARGIN:
            raise BL5TrialDomainError("BL5 director chart crossed antipodal boundary")
        director_change = [director_beam[j]-constant(self.reference_director[j])
                           for j in range(3)]
        director_gap = [ad.dot([constant(v) for v in basis], director_change)
                        for basis in self.director_basis]
        gaps = translation+director_gap
        work = sum((constant(multiplier[j])*gaps[j] for j in range(5)),
                   start=constant(0.))
        return LocalWork(
            float(work.value), _owned([gap.value for gap in gaps], (5,)),
            _owned([gap.gradient for gap in gaps], (5, size)),
            _owned(work.gradient, (size,)),
            _owned(work.hessian, (size, size)),
            _owned_slots(self.port.active_beam_slots(), (9,)))
