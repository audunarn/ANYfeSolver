"""Local retained B3-GE round line contact on one physical S3 interior.

This is a geometry/work kernel. Its shell coordinates are physical skin
vertices, not yet the S3 element's authoritative nodal degrees of freedom.
Rim contact, moving feature ownership and a coupled native/shell solve are
separate admissions.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite

import numpy as np

from anyfesolver._ge_beam3_centered_reference import (
    CenteredCurvedBeam3ReferenceGeometry,
)
from anyfesolver._ge_beam3_p5_seeded.core import sha

from .beam_curve_pair import _bezier_controls, _curve
from .native_retained_curve_cell import NativeRetainedCurveCell
from .native_retained_station import _readonly
from .overlap import PlanarFace


_GAUSS_8 = np.polynomial.legendre.leggauss(8)
_GAUSS_16 = np.polynomial.legendre.leggauss(16)
PROFILE = "NATIVE_B3_GE_S3_INTERIOR_LINE_PENALTY_DEV_V0"


class _Jet:
    """Scalar value and exact first/second derivatives in physical positions."""

    __slots__ = ("value", "gradient", "hessian")

    def __init__(self, value, gradient, hessian):
        self.value = float(value)
        self.gradient = gradient
        self.hessian = hessian

    @classmethod
    def variable(cls, value, index, size):
        gradient = np.zeros(size)
        gradient[index] = 1.0
        return cls(value, gradient, np.zeros((size, size)))

    def _other(self, value):
        if isinstance(value, _Jet):
            return value
        size = len(self.gradient)
        return _Jet(value, np.zeros(size), np.zeros((size, size)))

    def __add__(self, other):
        other = self._other(other)
        return _Jet(self.value+other.value,
                    self.gradient+other.gradient,
                    self.hessian+other.hessian)

    __radd__ = __add__

    def __neg__(self):
        return _Jet(-self.value, -self.gradient, -self.hessian)

    def __sub__(self, other):
        return self+-self._other(other)

    def __rsub__(self, other):
        return self._other(other)+-self

    def __mul__(self, other):
        other = self._other(other)
        gradient = (self.gradient*other.value
                    +other.gradient*self.value)
        hessian = (self.hessian*other.value
                   +other.hessian*self.value
                   +np.outer(self.gradient, other.gradient)
                   +np.outer(other.gradient, self.gradient))
        return _Jet(self.value*other.value, gradient, hessian)

    __rmul__ = __mul__

    def reciprocal(self):
        value = self.value
        return _Jet(1.0/value, -self.gradient/value**2,
                    (2.0*np.outer(self.gradient, self.gradient)/value**3
                     -self.hessian/value**2))

    def __truediv__(self, other):
        return self*self._other(other).reciprocal()

    def sqrt(self):
        root = np.sqrt(self.value)
        return _Jet(root, self.gradient/(2.0*root),
                    self.hessian/(2.0*root)
                    -np.outer(self.gradient, self.gradient)/(4.0*root**3))


def _dot(first, second):
    return sum((a*b for a, b in zip(first, second)), 0)


def _cross(first, second):
    return (
        first[1]*second[2]-first[2]*second[1],
        first[2]*second[0]-first[0]*second[2],
        first[0]*second[1]-first[1]*second[0],
    )


def _shape(station: float) -> np.ndarray:
    t = float(station)
    return np.array((2.0*(t-0.5)*(t-1.0),
                     4.0*t*(1.0-t),
                     2.0*t*(t-0.5)))


@dataclass(frozen=True)
class NativeRetainedLineS3Result:
    potential_j: float
    residual: np.ndarray
    tangent: np.ndarray
    beam_force_n: np.ndarray
    shell_force_n: np.ndarray
    beam_moment_nm: np.ndarray
    shell_moment_nm: np.ndarray
    shell_vertex_force_n: np.ndarray
    minimum_gap_m: float
    minimum_rim_clearance_m: float
    active_intervals: int
    integration_intervals: int
    integration_evaluations: int
    energy_error_j: float
    force_error_n: float
    tangent_error_n_per_m: float


class NativeRetainedRoundLineS3Interior:
    """Two native Q2 half-cells against one deformable physical S3 triangle."""

    def __init__(self, beam_element_id: int,
                 beam_reference: CenteredCurvedBeam3ReferenceGeometry,
                 beam_node_ids: tuple[int, int, int],
                 shell_element_id: int, shell_reference: PlanarFace,
                 shell_node_ids: tuple[int, int, int], *,
                 radius_m: float, line_stiffness_n_per_m2: float,
                 rim_clearance_m: float, geometry_tolerance_m: float,
                 energy_tolerance_j: float, force_tolerance_n: float,
                 tangent_tolerance_n_per_m: float,
                 maximum_integration_intervals: int):
        if (type(beam_reference) is not CenteredCurvedBeam3ReferenceGeometry
                or type(shell_reference) is not PlanarFace
                or len(shell_reference.corners) != 3):
            raise TypeError("native line-S3 needs exact native and physical S3 references")
        if (type(beam_element_id) is not int or beam_element_id <= 0
                or type(shell_element_id) is not int or shell_element_id <= 0):
            raise ValueError("native line-S3 needs positive element identities")
        if (type(shell_reference.feature_id) is not tuple
                or len(shell_reference.feature_id) != 2
                or type(shell_reference.feature_id[0]) is not int
                or shell_reference.feature_id[0] <= 0
                or shell_reference.feature_id[1] != shell_element_id):
            raise ValueError("native line-S3 needs matching shell feature identity")
        for name, ids in (("beam", beam_node_ids), ("shell", shell_node_ids)):
            if (type(ids) is not tuple or len(ids) != 3
                    or any(type(item) is not int or item <= 0 for item in ids)
                    or len(set(ids)) != 3):
                raise ValueError(f"native line-S3 {name} needs three material nodes")
        controls = (radius_m, line_stiffness_n_per_m2,
                    geometry_tolerance_m, energy_tolerance_j,
                    force_tolerance_n, tangent_tolerance_n_per_m)
        if any(type(value) not in (int, float) or not isfinite(value)
               or value <= 0.0 for value in controls):
            raise ValueError("native line-S3 needs positive dimensional controls")
        if (type(rim_clearance_m) not in (int, float)
                or not isfinite(rim_clearance_m) or rim_clearance_m < 0.0):
            raise ValueError("native line-S3 needs nonnegative rim clearance")
        if (type(maximum_integration_intervals) is not int
                or maximum_integration_intervals < 1):
            raise ValueError("native line-S3 needs positive interval budget")
        self.cells = tuple(NativeRetainedCurveCell(
            beam_element_id, beam_reference, cell,
        ) for cell in (0, 1))
        self.reference_curves = tuple(
            cell.evaluate(beam_reference.coordinates,
                          np.zeros((3, 3)),
                          np.tile(np.eye(3), (2, 1, 1))).positions
            for cell in self.cells
        )
        self.beam_reference = beam_reference
        self.beam_element_id = beam_element_id
        self.shell_element_id = shell_element_id
        self.shell_reference = shell_reference
        self.beam_node_ids = beam_node_ids
        self.shell_node_ids = shell_node_ids
        self.radius = float(radius_m)
        self.stiffness = float(line_stiffness_n_per_m2)
        self.rim_clearance = float(rim_clearance_m)
        self.geometry_tolerance = float(geometry_tolerance_m)
        self.energy_tolerance = float(energy_tolerance_j)
        self.force_tolerance = float(force_tolerance_n)
        self.tangent_tolerance = float(tangent_tolerance_n_per_m)
        self.maximum_integration_intervals = maximum_integration_intervals
        self.identity = self._fingerprint()

    def _fingerprint(self):
        return sha(dict(
            profile=PROFILE,
            beam_element=self.beam_element_id,
            beam_reference=self.beam_reference.canonical_data(),
            beam_nodes=self.beam_node_ids,
            shell_element=self.shell_element_id,
            shell_reference=self.shell_reference.corners,
            shell_feature=self.shell_reference.feature_id,
            shell_nodes=self.shell_node_ids,
            radius=self.radius, stiffness=self.stiffness,
            rim_clearance=self.rim_clearance,
            geometry_tolerance=self.geometry_tolerance,
            energy_tolerance=self.energy_tolerance,
            force_tolerance=self.force_tolerance,
            tangent_tolerance=self.tangent_tolerance,
            maximum_integration_intervals=self.maximum_integration_intervals,
        ))

    def _geometry_state(self, positions: np.ndarray, position_low: np.ndarray,
                 cell_rotations: np.ndarray,
                 shell_physical_vertices: np.ndarray):
        if self._fingerprint() != self.identity:
            raise ValueError("native line-S3 definition changed")
        shell = np.asarray(shell_physical_vertices, dtype=float)
        if shell.shape != (3, 3) or not np.all(np.isfinite(shell)):
            raise ValueError("native line-S3 needs three finite physical shell vertices")
        states = tuple(cell.evaluate(positions, position_low, cell_rotations)
                       for cell in self.cells)
        curves = tuple(state.positions for state in states)
        edge_a, edge_b = shell[1]-shell[0], shell[2]-shell[0]
        cross = np.cross(edge_a, edge_b)
        magnitude = float(np.linalg.norm(cross))
        scale = max(1.0, float(np.max(np.abs(shell))),
                    *(float(np.max(np.abs(curve))) for curve in curves))
        roundoff = 256.0*np.finfo(float).eps*scale
        if (roundoff > self.geometry_tolerance
                or magnitude <= self.geometry_tolerance**2):
            raise ValueError("native line-S3 has unresolved or degenerate geometry")
        normal = cross/magnitude
        # The ordered material triangle defines its normal. A normal test
        # against the fixed reference frame would reject valid large rigid
        # rotations; path-wise inversion checks belong to the analysis owner.
        minimum_rim = float("inf")
        for curve in curves:
            for point in _bezier_controls(curve, 0.0, 1.0):
                for index in range(3):
                    edge = shell[(index+1) % 3]-shell[index]
                    inward = float(normal@np.cross(edge, point-shell[index])
                                   /np.linalg.norm(edge))
                    minimum_rim = min(minimum_rim, inward)
        if minimum_rim-roundoff < self.radius+self.rim_clearance:
            raise ValueError("native line-S3 curve approaches unadmitted physical rim")

        return states, curves, shell, normal, roundoff, minimum_rim

    def geometry_gap(self, positions, position_low, cell_rotations, shell_physical_vertices, *, _work_budget=None):
        """Physical quadratic gap/rim without constructing contact derivatives."""
        if _work_budget is not None:_work_budget.debit('geometry')
        _, curves, shell, normal, _, rim = self._geometry_state(
            positions, position_low, cell_rotations, shell_physical_vertices)
        minimum = float('inf')
        for curve in curves:
            gaps = np.array([float(normal@(point-shell[0])-self.radius) for point in curve])
            quadratic = 2.0*(gaps[0]-2.0*gaps[1]+gaps[2])
            linear = -3.0*gaps[0]+4.0*gaps[1]-gaps[2]
            stations = [0.0, 1.0]
            if quadratic != 0.0:
                turning = -linear/(2.0*quadratic)
                if 0.0 < turning < 1.0: stations.append(float(turning))
            minimum = min(minimum, *(float((quadratic*t+linear)*t+gaps[0]) for t in stations))
        return minimum, rim

    def evaluate(self, positions: np.ndarray, position_low: np.ndarray,
                 cell_rotations: np.ndarray,
                 shell_physical_vertices: np.ndarray, *, _work_budget=None, _geometry_predebited=False) -> NativeRetainedLineS3Result:
        if _geometry_predebited and _work_budget is None:
            raise ValueError('private predebit requires aggregate work ledger')
        if _work_budget is not None and not _geometry_predebited:_work_budget.debit('geometry')
        states, curves, shell, normal, roundoff, minimum_rim = self._geometry_state(
            positions, position_low, cell_rotations, shell_physical_vertices)

        variables = tuple(_Jet.variable(value, index, 27)
                          for index, value in enumerate(np.r_[
                              curves[0].reshape(-1), curves[1].reshape(-1),
                              shell.reshape(-1),
                          ]))
        shell_jets = tuple(tuple(variables[18+3*i+j] for j in range(3))
                           for i in range(3))
        first = tuple(a-b for a, b in zip(shell_jets[1], shell_jets[0]))
        second = tuple(a-b for a, b in zip(shell_jets[2], shell_jets[0]))
        normal_jet = _cross(first, second)
        norm_jet = _dot(normal_jet, normal_jet).sqrt()
        normal_jet = tuple(component/norm_jet for component in normal_jet)
        curve_jets = tuple(
            tuple(tuple(variables[9*cell+3*point+axis]
                        for axis in range(3)) for point in range(3))
            for cell in (0, 1)
        )

        total = np.zeros(1+27+27*27)
        errors = np.zeros(3)
        visited = intervals = evaluations = active_intervals = 0
        minimum_gap = float("inf")

        def quadrature(cell, left, right, rule):
            nonlocal evaluations
            points, weights = rule
            result = np.zeros_like(total)
            half = 0.5*(right-left)
            middle = 0.5*(left+right)
            for point, weight in zip(points, weights):
                station = middle+half*float(point)
                shape = _shape(station)
                position = tuple(sum((shape[j]*curve_jets[cell][j][axis]
                                      for j in range(3)), 0)
                                 for axis in range(3))
                delta = tuple(a-b for a, b in zip(position, shell_jets[0]))
                gap = _dot(normal_jet, delta)-self.radius
                evaluations += 1
                if gap.value >= 0.0:
                    continue
                measure = float(np.linalg.norm(
                    _curve(self.reference_curves[cell], station)[1]
                ))
                factor = half*float(weight)*self.stiffness*measure
                result[0] += 0.5*factor*gap.value**2
                result[1:28] += factor*gap.value*gap.gradient
                result[28:] += (factor*(np.outer(gap.gradient, gap.gradient)
                                       +gap.value*gap.hessian)).reshape(-1)
            return result

        def integrate(cell, left, right, budget):
            nonlocal visited, intervals
            if _work_budget is not None:_work_budget.debit('integration')
            visited += 1
            if visited > self.maximum_integration_intervals:
                raise MemoryError(
                    "native line-S3 integration interval budget exhausted: "
                    f"cell {cell}, interval [{left}, {right}]"
                )
            midpoint = 0.5*(left+right)
            if midpoint == left or midpoint == right:
                raise MemoryError("native line-S3 parameter resolution exhausted")
            coarse = quadrature(cell, left, right, _GAUSS_8)
            refined = quadrature(cell, left, right, _GAUSS_16)
            difference = np.abs(refined-coarse)
            measured = np.array((difference[0],
                                 float(np.max(difference[1:28])),
                                 float(np.max(difference[28:]))))
            if np.all(measured <= budget):
                intervals += 1
                errors[:] += measured
                return refined
            return (integrate(cell, left, midpoint, 0.5*budget)
                    +integrate(cell, midpoint, right, 0.5*budget))

        for cell, curve in enumerate(curves):
            gaps = np.array([float(normal@(point-shell[0])-self.radius)
                             for point in curve])
            quadratic = 2.0*(gaps[0]-2.0*gaps[1]+gaps[2])
            linear = -3.0*gaps[0]+4.0*gaps[1]-gaps[2]
            constant = gaps[0]
            stations = [0.0, 1.0]
            extrema = [0.0, 1.0]
            if quadratic != 0.0:
                turning = -linear/(2.0*quadratic)
                if 0.0 < turning < 1.0:
                    extrema.append(float(turning))
                    # Preserve the legacy optional interior partition and
                    # checkpoint diagnostics where the interval is resolved.
                    # A stationary point is not an activation boundary, so
                    # an artificial near-end partition need not be integrated.
                    if min(turning, 1.0-turning) > 64.0*np.finfo(float).eps:
                        stations.append(float(turning))
                roots = np.roots((quadratic, linear, constant))
            elif linear != 0.0:
                roots = (-constant/linear,)
            else:
                roots = ()
            for root in roots:
                if abs(complex(root).imag) > 64.0*np.finfo(float).eps:
                    continue
                station = float(complex(root).real)
                if 0.0 < station < 1.0:
                    stations.append(station)
            stations = sorted(set(stations))
            # Turning points determine the geometric minimum but do not change
            # the active branch. Retain all extrema even when the optional
            # integration partition would be unresolved at a cell end.
            minimum_gap = min(minimum_gap, *(float((quadratic*t+linear)*t
                                                 +constant) for t in extrema))
            for left, right in zip(stations[:-1], stations[1:]):
                if right-left <= 64.0*np.finfo(float).eps:
                    raise ValueError("native line-S3 activation interval is unresolved")
                middle = 0.5*(left+right)
                gap = (quadratic*middle+linear)*middle+constant
                if abs(gap) <= roundoff:
                    raise ValueError("native line-S3 activation is ambiguous")
                if gap > 0.0:
                    continue
                active_intervals += 1
                share = 0.5*(right-left)
                budget = np.array((self.energy_tolerance*share,
                                   self.force_tolerance*share,
                                   self.tangent_tolerance*share))
                total += integrate(cell, left, right, budget)

        gradient = total[1:28]
        hessian = total[28:].reshape(27, 27)
        work_map = np.zeros((27, 51))
        for cell, state in enumerate(states):
            work_map[9*cell:9*cell+9, :42] = state.geometric_jacobian
        work_map[18:, 42:] = np.eye(9)
        residual = work_map.T@gradient
        tangent = work_map.T@hessian@work_map
        for cell, state in enumerate(states):
            _, curvature = state.pullback(gradient[9*cell:9*cell+9],
                                          np.zeros((9, 9)))
            tangent[:42, :42] += curvature
        physical_force = -gradient.reshape(9, 3)
        beam_force = np.sum(physical_force[:6], axis=0)
        shell_force = np.sum(physical_force[6:], axis=0)
        beam_moment = sum((np.cross(curves[cell],
                                    physical_force[3*cell:3*cell+3])
                           for cell in (0, 1)), np.zeros((3, 3))).sum(axis=0)
        shell_moment = np.sum(np.cross(shell, physical_force[6:]), axis=0)
        return NativeRetainedLineS3Result(
            float(total[0]), _readonly(residual), _readonly(tangent),
            _readonly(beam_force), _readonly(shell_force),
            _readonly(beam_moment), _readonly(shell_moment),
            _readonly(physical_force[6:]), float(minimum_gap),
            float(minimum_rim), active_intervals, intervals, evaluations,
            float(errors[0]), float(errors[1]), float(errors[2]),
        )
