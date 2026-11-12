"""Internal pointwise penalty oracle for independently moving S3 skin faces.

The current source triangle supplies the contact normal. The scalar potential
is integrated over the source reference-area measure, using its affine area
map to the current projected overlap. This is a verification specialization,
not the selected primal-pressure mortar law or a public solver profile.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .overlap import PlanarFace, planar_projected_overlap
from .moving import _has_shared_edge_transition


@dataclass(frozen=True)
class TrianglePairEnergyState:
    energy: float
    overlap_area: float
    active_area: float
    minimum_gap: float | None
    overlap_vertices: int
    active_vertices: int
    minimum_vertex_abs_gap: float | None

    @property
    def branch_signature(self) -> tuple[int, int]:
        return self.overlap_vertices, self.active_vertices


@dataclass(frozen=True)
class TrianglePairWorkState:
    energy: TrianglePairEnergyState
    residual: np.ndarray
    tangent: np.ndarray | None
    difference_step: float
    energy_evaluations: int


def _triangle_area(positions: np.ndarray) -> float:
    return 0.5 * float(np.linalg.norm(np.cross(
        positions[1] - positions[0], positions[2] - positions[0],
    )))


def _active_polygon(points: np.ndarray,
                    gaps: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if len(points) < 3:
        return np.empty((0, 3)), np.empty(0)
    output: list[np.ndarray] = []
    output_gaps: list[float] = []
    prior, prior_gap = points[-1], float(gaps[-1])
    for current, value in zip(points, gaps):
        gap = float(value)
        was_active, is_active = prior_gap <= 0.0, gap <= 0.0
        if was_active != is_active:
            fraction = prior_gap / (prior_gap - gap)
            output.append(prior + fraction * (current - prior))
            output_gaps.append(0.0)
        if is_active:
            output.append(current)
            output_gaps.append(gap)
        prior, prior_gap = current, gap
    if not output:
        return np.empty((0, 3)), np.empty(0)
    return np.asarray(output), np.asarray(output_gaps)


class MovingS3PairPenalty:
    """Exact affine-gap potential and numerical smooth-branch work map."""

    def __init__(self, reference_source: np.ndarray,
                 reference_target: np.ndarray, stiffness: float,
                 *, difference_step: float,
                 maximum_energy_evaluations: int = 1500):
        source = np.asarray(reference_source, dtype=float)
        target = np.asarray(reference_target, dtype=float)
        self.reference_source = PlanarFace((1, 1), source).corners
        self.reference_target = PlanarFace((2, 1), target).corners
        if source.shape != (3, 3) or target.shape != (3, 3):
            raise ValueError("moving S3 pair needs two triangular reference faces")
        self.reference_area = _triangle_area(source)
        self.reference_normal = np.cross(source[1] - source[0],
                                         source[2] - source[0])
        self.reference_normal /= np.linalg.norm(self.reference_normal)
        target_normal = np.cross(target[1] - target[0],
                                 target[2] - target[0])
        target_normal /= np.linalg.norm(target_normal)
        if float(self.reference_normal @ target_normal) <= 0.0:
            raise ValueError("S3 pair reference face polarities conflict")
        if not np.isfinite(stiffness) or stiffness <= 0.0:
            raise ValueError("S3 pair stiffness must be finite and positive")
        if not np.isfinite(difference_step) or difference_step <= 0.0:
            raise ValueError("S3 pair difference step must be finite and positive")
        if (type(maximum_energy_evaluations) is not int
                or maximum_energy_evaluations < 1):
            raise ValueError("S3 pair energy evaluation budget must be positive")
        self.stiffness = float(stiffness)
        self.difference_step = float(difference_step)
        self.maximum_energy_evaluations = maximum_energy_evaluations

    def evaluate(self, current_source: np.ndarray,
                 current_target: np.ndarray) -> TrianglePairEnergyState:
        source = PlanarFace((1, 1), current_source)
        target = PlanarFace((2, 1), current_target)
        if source.corners.shape != (3, 3) or target.corners.shape != (3, 3):
            raise ValueError("moving S3 pair needs triangular current faces")
        source_normal = np.cross(source.corners[1] - source.corners[0],
                                 source.corners[2] - source.corners[0])
        source_normal /= np.linalg.norm(source_normal)
        target_normal = np.cross(target.corners[1] - target.corners[0],
                                 target.corners[2] - target.corners[0])
        target_normal /= np.linalg.norm(target_normal)
        if (float(source_normal @ self.reference_normal) <= 0.0
                or float(target_normal @ source_normal) <= 0.0):
            raise ValueError("moving S3 pair inverted or ambiguous face polarity")
        overlap = planar_projected_overlap(source, target)
        if overlap is None:
            return TrianglePairEnergyState(0.0, 0.0, 0.0, None, 0, 0, None)
        gaps = np.asarray([
            float((point - overlap.shape_values(target, point)
                   @ target.corners) @ source_normal)
            for point in overlap.polygon
        ])
        active_points, active_gaps = _active_polygon(overlap.polygon, gaps)
        area_ratio = self.reference_area / _triangle_area(source.corners)
        integral = 0.0
        active_area = 0.0
        for index in range(1, len(active_points) - 1):
            vertices = active_points[[0, index, index + 1]]
            values = active_gaps[[0, index, index + 1]]
            area = _triangle_area(vertices)
            active_area += area
            integral += (area / 6.0) * (
                float(values @ values)
                + float(values[0]*values[1] + values[1]*values[2]
                        + values[2]*values[0])
            )
        return TrianglePairEnergyState(
            0.5 * self.stiffness * area_ratio * integral,
            overlap.area, active_area, float(np.min(gaps)),
            len(overlap.polygon), len(active_points),
            float(np.min(np.abs(gaps))),
        )

    def work(self, current_source: np.ndarray, current_target: np.ndarray,
             *, tangent: bool = True) -> TrianglePairWorkState:
        positions = np.concatenate((np.asarray(current_source, dtype=float),
                                    np.asarray(current_target, dtype=float))).reshape(-1)
        if positions.size != 18:
            raise ValueError("moving S3 pair work map needs eighteen coordinates")
        base = self.evaluate(positions[:9].reshape(3, 3),
                             positions[9:].reshape(3, 3))
        if (base.minimum_vertex_abs_gap is not None
                and base.minimum_vertex_abs_gap <= 8.0 * self.difference_step):
            raise ValueError("moving S3 pair derivative is at a contact branch")
        if base.overlap_vertices:
            source_face = PlanarFace((1, 1), positions[:9].reshape(3, 3))
            target_face = PlanarFace((2, 1), positions[9:].reshape(3, 3))
            overlap = planar_projected_overlap(source_face, target_face)
            if _has_shared_edge_transition(source_face, target_face, overlap):
                raise ValueError("moving S3 pair shared-edge derivative is nonunique")
        evaluations = 1
        cache: dict[tuple[tuple[int, int], ...], float] = {}

        def energy_at(steps: tuple[tuple[int, int], ...]) -> float:
            nonlocal evaluations
            key = tuple(sorted(steps))
            if key in cache:
                return cache[key]
            evaluations += 1
            if evaluations > self.maximum_energy_evaluations:
                raise MemoryError("moving S3 pair differentiation budget exhausted")
            trial = positions.copy()
            for coordinate, sign in steps:
                trial[coordinate] += sign * self.difference_step
            state = self.evaluate(trial[:9].reshape(3, 3),
                                  trial[9:].reshape(3, 3))
            if state.branch_signature != base.branch_signature:
                raise ValueError("moving S3 pair derivative crosses an overlap/contact branch")
            cache[key] = state.energy
            return state.energy

        gradient = np.empty(18)
        positive = np.empty(18)
        negative = np.empty(18)
        for index in range(18):
            positive[index] = energy_at(((index, 1),))
            negative[index] = energy_at(((index, -1),))
            gradient[index] = ((positive[index] - negative[index])
                               / (2.0 * self.difference_step))
        hessian = None
        if tangent:
            hessian = np.empty((18, 18))
            for first in range(18):
                hessian[first, first] = (
                    positive[first] - 2.0*base.energy + negative[first]
                ) / self.difference_step**2
                for second in range(first + 1, 18):
                    hessian[first, second] = hessian[second, first] = (
                        energy_at(((first, 1), (second, 1)))
                        - energy_at(((first, 1), (second, -1)))
                        - energy_at(((first, -1), (second, 1)))
                        + energy_at(((first, -1), (second, -1)))
                    ) / (4.0 * self.difference_step**2)
        return TrianglePairWorkState(
            base, gradient, hessian, self.difference_step, evaluations,
        )
