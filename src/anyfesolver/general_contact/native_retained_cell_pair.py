"""Internal distributed native B3-GE contact for one declared half-cell pair."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite

import numpy as np

from anyfesolver._ge_beam3_centered_reference import (
    CenteredCurvedBeam3ReferenceGeometry,
)
from anyfesolver._ge_beam3_p5_seeded.core import sha

from .beam_curve_pair import B3PairPotential, certify_b3_pair_gap
from .native_retained_curve_cell import NativeRetainedCurveCell
from .native_retained_station import _readonly


@dataclass(frozen=True)
class NativeRetainedCellPairResult:
    potential_j: float
    residual: np.ndarray
    tangent: np.ndarray
    beam_a_force_n: np.ndarray
    beam_b_force_n: np.ndarray
    beam_a_moment_nm: np.ndarray
    beam_b_moment_nm: np.ndarray
    conservative_lower_gap_m: float
    sampled_gap_m: float
    integration_points: int
    integrated_intervals: int
    projection_evaluations: int
    maximum_relative_integration_error: float
    gap_cells_checked: int


class NativeRetainedSelectedCellPair:
    """One explicit half-cell pair; all undeclared pair proximity fails closed."""

    def __init__(self, element_a: int,
                 reference_a: CenteredCurvedBeam3ReferenceGeometry,
                 cell_a: int, radius_a_m: float,
                 element_b: int,
                 reference_b: CenteredCurvedBeam3ReferenceGeometry,
                 cell_b: int, radius_b_m: float, *,
                 line_stiffness_n_per_m2: float,
                 integration_relative_tolerance: float,
                 maximum_subdivisions: int,
                 gap_accuracy_m: float,
                 maximum_gap_cells: int,
                 maximum_gap_depth: int):
        if type(element_a) is not int or type(element_b) is not int or element_a == element_b:
            raise ValueError("native cell pair needs distinct exact beam identities")
        controls = (radius_a_m, radius_b_m, line_stiffness_n_per_m2,
                    integration_relative_tolerance, gap_accuracy_m)
        if any(type(value) not in (int, float) or not isfinite(value) or value <= 0.0
               for value in controls):
            raise ValueError("native cell pair needs positive dimensional controls")
        if (type(maximum_subdivisions) is not int or maximum_subdivisions < 1
                or type(maximum_gap_cells) is not int or maximum_gap_cells < 1
                or type(maximum_gap_depth) is not int or maximum_gap_depth < 0):
            raise ValueError("native cell pair needs finite integration/search budgets")
        self.cells_a = tuple(NativeRetainedCurveCell(element_a, reference_a, c)
                             for c in (0, 1))
        self.cells_b = tuple(NativeRetainedCurveCell(element_b, reference_b, c)
                             for c in (0, 1))
        if type(cell_a) is not int or cell_a not in (0, 1) or type(cell_b) is not int or cell_b not in (0, 1):
            raise ValueError("native cell pair needs exact selected half-cells")
        self.cell_a = cell_a
        self.cell_b = cell_b
        self.radius_a = float(radius_a_m)
        self.radius_b = float(radius_b_m)
        self.radius = self.radius_a+self.radius_b
        self.stiffness = float(line_stiffness_n_per_m2)
        self.integration_tolerance = float(integration_relative_tolerance)
        self.maximum_subdivisions = maximum_subdivisions
        self.gap_accuracy = float(gap_accuracy_m)
        self.maximum_gap_cells = maximum_gap_cells
        self.maximum_gap_depth = maximum_gap_depth
        virgin_a = self.cells_a[cell_a].evaluate(
            reference_a.coordinates, np.zeros((3, 3)),
            np.tile(np.eye(3), (2, 1, 1)),
        )
        virgin_b = self.cells_b[cell_b].evaluate(
            reference_b.coordinates, np.zeros((3, 3)),
            np.tile(np.eye(3), (2, 1, 1)),
        )
        self.law = B3PairPotential(
            virgin_a.positions, virgin_b.positions, self.radius,
            self.stiffness,
            integration_relative_tolerance=self.integration_tolerance,
            maximum_subdivisions=self.maximum_subdivisions,
            reference_arc_measure="quadratic",
        )
        self.identity = self._fingerprint()

    def _fingerprint(self) -> str:
        return sha(dict(
            profile="NATIVE_RETAINED_SELECTED_Q2_CELL_PAIR_DEV_V0",
            cells_a=[(cell.element_id, cell.cell,
                      cell.reference.canonical_data()) for cell in self.cells_a],
            cells_b=[(cell.element_id, cell.cell,
                      cell.reference.canonical_data()) for cell in self.cells_b],
            cell_a=self.cell_a, radius_a_m=self.radius_a,
            cell_b=self.cell_b, radius_b_m=self.radius_b,
            line_stiffness_n_per_m2=self.stiffness,
            integration_relative_tolerance=self.integration_tolerance,
            maximum_subdivisions=self.maximum_subdivisions,
            gap_accuracy_m=self.gap_accuracy,
            maximum_gap_cells=self.maximum_gap_cells,
            maximum_gap_depth=self.maximum_gap_depth,
            law=dict(reference_a=self.law.reference_a,
                     reference_b=self.law.reference_b,
                     radius=self.law.radius,
                     stiffness=self.law.stiffness,
                     tolerance=self.law.tolerance,
                     maximum_subdivisions=self.law.maximum_subdivisions,
                     reference_arc_measure=self.law.reference_arc_measure),
        ))

    def evaluate(self, positions_a: np.ndarray, position_low_a: np.ndarray,
                 cell_rotations_a: np.ndarray,
                 positions_b: np.ndarray, position_low_b: np.ndarray,
                 cell_rotations_b: np.ndarray) -> NativeRetainedCellPairResult:
        if self._fingerprint() != self.identity:
            raise ValueError("native cell pair definition changed")
        states_a = tuple(cell.evaluate(positions_a, position_low_a,
                                       cell_rotations_a) for cell in self.cells_a)
        states_b = tuple(cell.evaluate(positions_b, position_low_b,
                                       cell_rotations_b) for cell in self.cells_b)
        certificates = {}
        for a in (0, 1):
            for b in (0, 1):
                certificate = certify_b3_pair_gap(
                    states_a[a].positions, states_b[b].positions, self.radius,
                    accuracy=self.gap_accuracy,
                    maximum_cells=self.maximum_gap_cells,
                    maximum_depth=self.maximum_gap_depth,
                )
                certificates[a, b] = certificate
                if (a, b) != (self.cell_a, self.cell_b) and certificate.conservative_lower_gap <= 0.0:
                    raise ValueError(
                        f"undeclared native half-cell pair {(a, b)} may contact"
                    )
        a = states_a[self.cell_a]
        b = states_b[self.cell_b]
        selected = certificates[self.cell_a, self.cell_b]
        if selected.conservative_lower_gap >= 0.0:
            energy = 0.0
            gradient = np.zeros(18)
            hessian = np.zeros((18, 18))
            points = intervals = projections = 0
            error = 0.0
        else:
            result = self.law.evaluate(a.positions, b.positions)
            energy, gradient, hessian = result.energy, result.residual, result.tangent
            points = result.integration_points
            intervals = result.integrated_intervals
            projections = result.projection_evaluations
            error = result.maximum_relative_integration_error
        residual_a, tangent_a = a.pullback(gradient[:9], hessian[:9, :9])
        residual_b, tangent_b = b.pullback(gradient[9:], hessian[9:, 9:])
        residual = np.r_[residual_a, residual_b]
        tangent = np.zeros((84, 84))
        tangent[:42, :42] = tangent_a
        tangent[42:, 42:] = tangent_b
        tangent[:42, 42:] = (
            a.geometric_jacobian.T @ hessian[:9, 9:] @ b.geometric_jacobian
        )
        tangent[42:, :42] = (
            b.geometric_jacobian.T @ hessian[9:, :9] @ a.geometric_jacobian
        )
        forces_a = -gradient[:9].reshape(3, 3)
        forces_b = -gradient[9:].reshape(3, 3)
        return NativeRetainedCellPairResult(
            float(energy), _readonly(residual), _readonly(tangent),
            _readonly(np.sum(forces_a, axis=0)),
            _readonly(np.sum(forces_b, axis=0)),
            _readonly(np.sum(np.cross(a.positions, forces_a), axis=0)),
            _readonly(np.sum(np.cross(b.positions, forces_b), axis=0)),
            selected.conservative_lower_gap, selected.minimum_sampled_gap,
            points, intervals, projections, error,
            sum(value.cells_checked for value in certificates.values()),
        )
