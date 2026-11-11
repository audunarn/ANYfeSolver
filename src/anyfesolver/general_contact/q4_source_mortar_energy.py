"""Moving Q4 Q1 pressure-side mortar penalty against S3 or Q4 targets."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .moving import _has_shared_edge_transition
from .overlap import PlanarFace, planar_projected_overlap
from .q4_source_overlap import (
    projected_q4_source_face, q4_source_overlap_integrals,
)


@dataclass(frozen=True)
class MovingQ4MortarState:
    energy: float
    reference_support: np.ndarray
    reference_weighted_gap: np.ndarray
    normalized_gap: np.ndarray
    pressure: np.ndarray
    pressure_gram: np.ndarray
    overlap_area: float
    reference_overlap_area: float
    minimum_sampled_gap: float | None
    overlap_vertices: int
    area_integration_error: float
    weighted_gap_integration_error: float
    integration_triangles: int

    @property
    def active_pressure_dofs(self) -> int:
        return int(np.count_nonzero(self.pressure))


@dataclass(frozen=True)
class MovingQ4MortarWork:
    state: MovingQ4MortarState
    residual: np.ndarray
    tangent: np.ndarray | None
    energy_evaluations: int


class MovingQ4PrimalMortarPenalty:
    """Differentiate the normalized Q1 source-pressure energy on physical corners."""

    def __init__(self, reference_source: np.ndarray,
                 reference_target: np.ndarray, stiffness: float,
                 *, difference_step: float,
                 area_tolerance: float,
                 weighted_gap_tolerance: float,
                 maximum_integration_triangles: int,
                 maximum_energy_evaluations: int = 1500):
        source = PlanarFace((1, 1), reference_source)
        target = PlanarFace((2, 1), reference_target)
        if len(source.corners) != 4 or len(target.corners) not in {3, 4}:
            raise ValueError("moving Q4 mortar needs Q4 source and S3/Q4 target")
        if (not np.isfinite(stiffness) or stiffness <= 0.0
                or not np.isfinite(difference_step) or difference_step <= 0.0):
            raise ValueError("moving Q4 mortar stiffness and difference step must be positive")
        if (type(maximum_energy_evaluations) is not int
                or maximum_energy_evaluations < 1):
            raise ValueError("moving Q4 mortar energy evaluation budget must be positive")
        self.reference_source = source.corners
        self.reference_target = target.corners
        self.stiffness = float(stiffness)
        self.difference_step = float(difference_step)
        self.area_tolerance = area_tolerance
        self.weighted_gap_tolerance = weighted_gap_tolerance
        self.maximum_integration_triangles = maximum_integration_triangles
        self.maximum_energy_evaluations = maximum_energy_evaluations
        self.n_coordinates = 12+3*len(target.corners)
        if 1+2*self.n_coordinates**2 > maximum_energy_evaluations:
            raise MemoryError("moving Q4 mortar Hessian exceeds evaluation budget")
        # Validate the dimensional quadrature controls before a candidate solve.
        q4_source_overlap_integrals(
            self.reference_source, self.reference_source,
            self.reference_target,
            area_tolerance=self.area_tolerance,
            weighted_gap_tolerance=self.weighted_gap_tolerance,
            maximum_triangles=self.maximum_integration_triangles,
        )

    def evaluate(self, current_source: np.ndarray,
                 current_target: np.ndarray) -> MovingQ4MortarState:
        integrals = q4_source_overlap_integrals(
            self.reference_source, current_source, current_target,
            area_tolerance=self.area_tolerance,
            weighted_gap_tolerance=self.weighted_gap_tolerance,
            maximum_triangles=self.maximum_integration_triangles,
        )
        support = integrals.support
        normalized = np.zeros(4)
        pressure = np.zeros(4)
        if integrals.overlap_vertices:
            if np.any(support <= 0.0) or not np.all(np.isfinite(support)):
                raise ValueError("moving Q4 mortar has nonpositive pressure support")
            normalized = integrals.weighted_gap/support
            pressure = self.stiffness*np.maximum(-normalized, 0.0)
        energy = 0.5*self.stiffness*float(
            support @ np.minimum(normalized, 0.0)**2
        )
        return MovingQ4MortarState(
            energy, support, integrals.weighted_gap, normalized, pressure,
            integrals.pressure_gram, integrals.overlap_area,
            integrals.reference_overlap_area, integrals.minimum_sampled_gap,
            integrals.overlap_vertices, integrals.area_error,
            integrals.weighted_gap_error, integrals.integration_triangles,
        )

    def work(self, current_source: np.ndarray, current_target: np.ndarray,
             *, tangent: bool = True) -> MovingQ4MortarWork:
        positions = np.concatenate((np.asarray(current_source, dtype=float),
                                    np.asarray(current_target, dtype=float))).reshape(-1)
        if positions.size != self.n_coordinates:
            raise ValueError("moving Q4 mortar work has invalid coordinate count")
        source_count = 12

        def state_at(values):
            return self.evaluate(values[:source_count].reshape(4, 3),
                                 values[source_count:].reshape(-1, 3))

        base = state_at(positions)
        if base.energy == 0.0:
            return MovingQ4MortarWork(
                base, np.zeros(self.n_coordinates),
                np.zeros((self.n_coordinates, self.n_coordinates)) if tangent else None,
                1,
            )
        source_face, normal = projected_q4_source_face(
            positions[:source_count].reshape(4, 3),
        )
        target = positions[source_count:].reshape(-1, 3)
        projected_target = target - ((target-source_face.corners[0])@normal)[:, None]*normal
        target_face = PlanarFace((2, 1), projected_target)
        overlap = planar_projected_overlap(source_face, target_face)
        if _has_shared_edge_transition(source_face, target_face, overlap):
            raise ValueError("moving Q4 mortar shared-edge derivative is nonunique")
        evaluations = 1
        cache: dict[tuple[tuple[int, int], ...], float] = {}

        def energy_at(steps):
            nonlocal evaluations
            key = tuple(sorted(steps))
            if key in cache:
                return cache[key]
            evaluations += 1
            if evaluations > self.maximum_energy_evaluations:
                raise MemoryError("moving Q4 mortar differentiation budget exhausted")
            trial = positions.copy()
            for coordinate, sign in steps:
                trial[coordinate] += sign*self.difference_step
            result = state_at(trial)
            cache[key] = result.energy
            return result.energy

        positive = np.empty(self.n_coordinates)
        negative = np.empty(self.n_coordinates)
        residual = np.empty(self.n_coordinates)
        for coordinate in range(self.n_coordinates):
            positive[coordinate] = energy_at(((coordinate, 1),))
            negative[coordinate] = energy_at(((coordinate, -1),))
            residual[coordinate] = (
                positive[coordinate]-negative[coordinate]
            )/(2.0*self.difference_step)
        hessian = None
        if tangent:
            hessian = np.empty((self.n_coordinates, self.n_coordinates))
            for first in range(self.n_coordinates):
                hessian[first, first] = (
                    positive[first]-2.0*base.energy+negative[first]
                )/self.difference_step**2
                for second in range(first+1, self.n_coordinates):
                    hessian[first, second] = hessian[second, first] = (
                        energy_at(((first, 1), (second, 1)))
                        - energy_at(((first, 1), (second, -1)))
                        - energy_at(((first, -1), (second, 1)))
                        + energy_at(((first, -1), (second, -1)))
                    )/(4.0*self.difference_step**2)
        return MovingQ4MortarWork(base, residual, hessian, evaluations)
