"""Local native B3-GE round-centreline penalty work against a fixed plane."""

from __future__ import annotations

from dataclasses import dataclass
from math import fsum, isfinite

import numpy as np

from anyfesolver._ge_beam3_centered_reference import (
    CenteredCurvedBeam3ReferenceGeometry,
)
from anyfesolver._ge_beam3_p5.arrays import _array, _frames
from anyfesolver._ge_beam3_p5.compensated_coordinates import validate_pair

from .native_retained_station import NativeRetainedRoundStation


def _owned(value: np.ndarray) -> np.ndarray:
    result = np.array(value, dtype=float, copy=True)
    result.setflags(write=False)
    return result


@dataclass(frozen=True)
class NativeRetainedLinePlaneResult:
    potential_j: float
    residual: np.ndarray
    tangent: np.ndarray
    beam_force_n: np.ndarray
    plane_force_n: np.ndarray
    beam_moment_nm: np.ndarray
    plane_moment_nm: np.ndarray
    minimum_gap_m: float
    integration_intervals: int
    integration_evaluations: int
    energy_error_j: float
    force_error_n: float
    tangent_error_n_per_m: float


class NativeRetainedRoundLinePlane:
    """One declared radius; no endpoint caps or general contact admission."""

    def __init__(self, element_id: int,
                 reference: CenteredCurvedBeam3ReferenceGeometry,
                 *, radius_m: float, plane_origin_m: np.ndarray,
                 plane_normal: np.ndarray, line_stiffness_n_per_m2: float,
                 energy_tolerance_j: float, force_tolerance_n: float,
                 tangent_tolerance_n_per_m: float, maximum_intervals: int):
        if type(element_id) is not int or element_id <= 0:
            raise ValueError("native line contact needs positive element identity")
        if type(reference) is not CenteredCurvedBeam3ReferenceGeometry:
            raise TypeError("native line contact needs exact centered B3-GE reference")
        controls = (radius_m, line_stiffness_n_per_m2, energy_tolerance_j,
                    force_tolerance_n, tangent_tolerance_n_per_m)
        if any(type(item) not in (int, float) or not isfinite(item) or item <= 0.0
               for item in controls):
            raise ValueError("native line contact needs positive dimensional controls")
        if type(maximum_intervals) is not int or maximum_intervals < 1:
            raise ValueError("native line contact needs positive interval budget")
        origin = _array(plane_origin_m, (3,), "native line plane origin")
        normal = _array(plane_normal, (3,), "native line plane normal")
        if abs(float(np.linalg.norm(normal))-1.0) > 1e-11:
            raise ValueError("native line contact plane normal must be unit")
        self.element_id = element_id
        self.reference = reference
        self.radius = float(radius_m)
        self.origin = _owned(origin)
        self.normal = _owned(normal)
        self.stiffness = float(line_stiffness_n_per_m2)
        self.energy_tolerance = float(energy_tolerance_j)
        self.force_tolerance = float(force_tolerance_n)
        self.tangent_tolerance = float(tangent_tolerance_n_per_m)
        self.maximum_intervals = maximum_intervals

    def evaluate(self, positions: np.ndarray, position_low: np.ndarray,
                 cell_rotations: np.ndarray) -> NativeRetainedLinePlaneResult:
        high = _array(positions, (3, 3), "native contact nodal positions")
        low = _array(position_low, (3, 3), "native contact nodal low parts")
        rotations = _frames(cell_rotations, 2, "native contact cell rotations")
        for first, second in zip(high.flat, low.flat):
            validate_pair(float(first), float(second))
        normal = self.normal
        stiffness = self.stiffness
        points4, weights4 = np.polynomial.legendre.leggauss(4)
        points8, weights8 = np.polynomial.legendre.leggauss(8)
        evaluations = 0
        intervals = 0
        visited = 0
        errors = np.zeros(3)
        total = np.zeros(1+24+24*24+1+3)
        minimum = float("inf")

        def centerline(cell: int, t: float) -> np.ndarray:
            lift = rotations[cell] @ self.reference.half_cell_lift(cell, t)
            return np.array([
                fsum(((1.0-t)*float(high[cell, axis]),
                      (1.0-t)*float(low[cell, axis]),
                      t*float(high[cell+1, axis]),
                      t*float(low[cell+1, axis]), float(lift[axis])))
                for axis in range(3)
            ])

        def gap(cell: int, t: float) -> float:
            return float(normal @ (centerline(cell, t)-self.origin)-self.radius)

        def integrand(cell: int, t: float) -> np.ndarray:
            nonlocal evaluations
            evaluations += 1
            xi = cell-1.0+t
            state = NativeRetainedRoundStation(
                self.element_id, self.reference, float(xi), self.radius, 0.0,
            ).evaluate(high, low, rotations)
            g = float(normal @ (state.centerline_position-self.origin)-self.radius)
            if g >= 0.0:
                return np.zeros_like(total)
            measure = self.reference.jacobian(xi)
            j = state.centerline_geometric_jacobian.T @ normal
            lift = rotations[cell] @ self.reference.half_cell_lift(cell, t)
            h = np.zeros((24, 24))
            angular = slice(18+3*cell, 21+3*cell)
            h[angular, angular] = (
                0.5*(np.outer(normal, lift)+np.outer(lift, normal))
                - float(normal @ lift)*np.eye(3)
            )
            normal_force = -stiffness*g
            return measure*np.concatenate((
                (0.5*stiffness*g*g,),
                stiffness*g*j,
                (stiffness*(np.outer(j, j)+g*h)).reshape(-1),
                (normal_force,),
                np.cross(state.centerline_position, normal_force*normal),
            ))

        def quadrature(cell: int, left: float, right: float,
                       nodes: np.ndarray, weights: np.ndarray) -> np.ndarray:
            midpoint = 0.5*(left+right)
            radius = 0.5*(right-left)
            value = np.zeros_like(total)
            for point, weight in zip(nodes, weights):
                value += radius*weight*integrand(cell, midpoint+radius*point)
            return value

        def integrate(cell: int, left: float, right: float,
                      energy_budget: float, force_budget: float,
                      tangent_budget: float) -> np.ndarray:
            nonlocal intervals, visited
            visited += 1
            if visited > self.maximum_intervals:
                raise MemoryError("native line contact integration interval budget exhausted")
            coarse = quadrature(cell, left, right, points4, weights4)
            refined = quadrature(cell, left, right, points8, weights8)
            difference = np.abs(refined-coarse)
            energy_error = float(difference[0])
            force_error = float(np.max(difference[1:25]))
            tangent_error = float(np.max(difference[25:25+24*24]))
            if (energy_error <= energy_budget and force_error <= force_budget
                    and tangent_error <= tangent_budget):
                intervals += 1
                errors[:] += (energy_error, force_error, tangent_error)
                return refined
            midpoint = 0.5*(left+right)
            return (integrate(cell, left, midpoint,
                              energy_budget/2.0, force_budget/2.0,
                              tangent_budget/2.0)
                    + integrate(cell, midpoint, right,
                                energy_budget/2.0, force_budget/2.0,
                                tangent_budget/2.0))

        for cell in (0, 1):
            first, middle, last = (gap(cell, t) for t in (0.0, 0.5, 1.0))
            a = 2.0*(last-2.0*middle+first)
            b = last-first-a
            candidates = [first, last]
            if a != 0.0:
                stationary = -b/(2.0*a)
                if 0.0 < stationary < 1.0:
                    candidates.append(gap(cell, stationary))
            minimum = min(minimum, *candidates)
            roots = np.polynomial.Polynomial((first, b, a)).roots()
            cuts = [0.0]
            for root in roots:
                if abs(root.imag) <= 1e-11 and 1e-12 < root.real < 1.0-1e-12:
                    cuts.append(float(root.real))
            cuts.append(1.0)
            cuts.sort()
            for left, right in zip(cuts, cuts[1:]):
                if right-left <= 1e-12 or gap(cell, 0.5*(left+right)) >= 0.0:
                    continue
                share = 0.5*(right-left)
                total += integrate(
                    cell, left, right,
                    self.energy_tolerance*share,
                    self.force_tolerance*share,
                    self.tangent_tolerance*share,
                )
        gradient = np.zeros(42)
        gradient[:24] = total[1:25]
        tangent = np.zeros((42, 42))
        tangent[:24, :24] = total[25:25+24*24].reshape(24, 24)
        normal_force = float(total[25+24*24])
        beam_force = normal_force*normal
        return NativeRetainedLinePlaneResult(
            float(total[0]), _owned(gradient), _owned(tangent),
            _owned(beam_force), _owned(-beam_force),
            _owned(total[-3:]), _owned(-total[-3:]),
            minimum, intervals, evaluations,
            float(errors[0]), float(errors[1]), float(errors[2]),
        )
