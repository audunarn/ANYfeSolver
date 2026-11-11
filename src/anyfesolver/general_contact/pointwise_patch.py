"""Separate pointwise penalty control for planar moving shell patches.

This is a verification specialization. It keeps no mortar pressure unknowns and
does not replace the primal pressure-space contact law.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .moving import (
    _boundary_quadrature,
    _has_shared_edge_transition,
    _triangle_samples,
)
from .overlap import planar_projected_overlap
from .patch_mortar import MovingPatchMortar


@dataclass(frozen=True)
class PointwisePatchState:
    overlap_area: float
    overlapping_pairs: int
    candidate_pairs: int
    integration_triangles: int
    active_quadrature_area: float
    energy: float
    normal_residual: np.ndarray
    normal_tangent: np.ndarray
    source_translation_residual: np.ndarray
    target_translation_residual: np.ndarray
    source_local_tangent_residual: np.ndarray
    target_local_tangent_residual: np.ndarray
    source_local_tangent_height_derivative: np.ndarray
    target_local_tangent_height_derivative: np.ndarray
    minimum_vertex_gap: float


@dataclass(frozen=True)
class PointwisePatchReducedSystem:
    state: PointwisePatchState
    residual: np.ndarray
    tangent: np.ndarray
    relative_tangent_asymmetry: float


def _subtriangles(triangle, subdivisions: int):
    triangles = [triangle]
    for _ in range(subdivisions):
        refined = []
        for first, second, third in triangles:
            middle_ab = 0.5 * (first + second)
            middle_bc = 0.5 * (second + third)
            middle_ca = 0.5 * (third + first)
            refined.extend((
                (first, middle_ab, middle_ca),
                (middle_ab, second, middle_bc),
                (middle_ca, middle_bc, third),
                (middle_ab, middle_bc, middle_ca),
            ))
        triangles = refined
    return triangles


class PointwisePatchPenalty(MovingPatchMortar):
    """Evaluate one-sided common-refinement pointwise penalty work.

    The inherited geometry validation and gap certificate are shared with the
    mortar kernel. The constitutive integration, residual and tangent below
    are independent of its weighted-gap pressure law.
    """

    def __init__(self, *args, integration_subdivisions: int,
                 maximum_integration_triangles: int, **kwargs):
        super().__init__(*args, **kwargs)
        if type(integration_subdivisions) is not int or integration_subdivisions < 0:
            raise ValueError("integration_subdivisions must be a nonnegative integer")
        if type(maximum_integration_triangles) is not int or maximum_integration_triangles < 1:
            raise ValueError("maximum_integration_triangles must be a positive integer")
        if integration_subdivisions > 20:
            raise MemoryError("pointwise integration subdivision request exceeds finite resource admission")
        self.integration_subdivisions = integration_subdivisions
        self.maximum_integration_triangles = maximum_integration_triangles

    def evaluate(self, source_translation: np.ndarray,
                 target_translation: np.ndarray,
                 source_normal_dofs: np.ndarray,
                 target_normal_dofs: np.ndarray) -> PointwisePatchState:
        moved, heights = self._fields(
            source_translation, target_translation,
            source_normal_dofs, target_normal_dofs,
        )
        source_count = len(self.source_node_ids)
        dof_count = source_count + len(self.target_node_ids)
        normal_force = np.zeros(dof_count)
        normal_tangent = np.zeros((dof_count, dof_count))
        source_translation_force = np.zeros(3)
        target_translation_force = np.zeros(3)
        source_local_tangent = np.zeros((source_count, 3))
        target_local_tangent = np.zeros((len(self.target_node_ids), 3))
        source_local_height_tangent = np.zeros((source_count, 3, dof_count))
        target_local_height_tangent = np.zeros((len(self.target_node_ids), 3, dof_count))
        energy = overlap_area = active_area = 0.0
        pairs = triangles_checked = 0
        minimum_vertex_gap = float("inf")

        candidates = self._candidate_indices(source_translation, target_translation)
        for source, target in candidates:
            face_a, moved_a = self.source_faces[source], moved[0][source]
            face_b, moved_b = self.target_faces[target], moved[1][target]
            source_rows = np.asarray([self.source_index[node] for node in face_a.node_ids])
            source_heights = heights[0][source_rows]
            target_rows = np.asarray([self.target_index[node] for node in face_b.node_ids])
            target_heights = heights[1][target_rows]
            overlap = planar_projected_overlap(moved_a, moved_b)
            if overlap is None:
                continue
            if _has_shared_edge_transition(moved_a, moved_b, overlap):
                raise ValueError("pointwise shared-edge transition needs a one-sided derivative")
            pairs += 1
            overlap_area += overlap.area
            polygon = overlap.polygon
            if (triangles_checked + len(polygon) * 4**self.integration_subdivisions
                    > self.maximum_integration_triangles):
                raise MemoryError("pointwise contact integration triangle limit exceeded")
            centre = np.mean(polygon, axis=0)
            columns = np.r_[source_rows, source_count + target_rows]

            def gap_at(point):
                shape_a = overlap.shape_values(moved_a, point)
                shape_b = overlap.shape_values(moved_b, point)
                return float(self.normal @ (
                    shape_a @ moved_a.corners - shape_b @ moved_b.corners
                ) + shape_a @ source_heights - shape_b @ target_heights)

            minimum_vertex_gap = min(minimum_vertex_gap,
                                     *(gap_at(point) for point in polygon))
            for index, corner in enumerate(polygon):
                following = polygon[(index + 1) % len(polygon)]
                coarse = (centre, corner, following)
                for triangle in _subtriangles(coarse, self.integration_subdivisions):
                    triangles_checked += 1
                    area = 0.5 * float(np.linalg.norm(np.cross(
                        triangle[1] - triangle[0], triangle[2] - triangle[0],
                    )))
                    if area <= 0.0:
                        continue
                    for point, weight in _triangle_samples(triangle, area):
                        shape_a, gradient_a = overlap.shape_values_and_gradient(moved_a, point)
                        shape_b, gradient_b = overlap.shape_values_and_gradient(moved_b, point)
                        gap = gap_at(point)
                        closure = min(gap, 0.0)
                        if closure == 0.0:
                            continue
                        local_gradient = np.r_[shape_a, -shape_b]
                        global_gradient = np.zeros(dof_count)
                        global_gradient[columns] = local_gradient
                        pressure = self.stiffness * closure
                        energy += 0.5 * weight * pressure * closure
                        normal_force[columns] += weight * pressure * local_gradient
                        normal_tangent[np.ix_(columns, columns)] += (
                            weight * self.stiffness
                            * np.outer(local_gradient, local_gradient)
                        )
                        active_area += weight
                        gap_gradient_a = -(
                            (source_heights + moved_a.corners @ self.normal) @ gradient_a
                        )
                        gap_gradient_b = (
                            (target_heights + moved_b.corners @ self.normal) @ gradient_b
                        )
                        source_translation_force += weight * pressure * gap_gradient_a
                        target_translation_force += weight * pressure * gap_gradient_b
                        source_local_tangent[source_rows] += (
                            weight * pressure * shape_a[:, None] * gap_gradient_a
                        )
                        target_local_tangent[target_rows] += (
                            weight * pressure * shape_b[:, None] * gap_gradient_b
                        )
                        source_gradient_tangent = np.zeros((3, dof_count))
                        target_gradient_tangent = np.zeros((3, dof_count))
                        source_gradient_tangent[:, source_rows] = -gradient_a.T
                        target_gradient_tangent[:, source_count + target_rows] = gradient_b.T
                        source_density_tangent = (
                            self.stiffness * np.outer(gap_gradient_a, global_gradient)
                            + pressure * source_gradient_tangent
                        )
                        target_density_tangent = (
                            self.stiffness * np.outer(gap_gradient_b, global_gradient)
                            + pressure * target_gradient_tangent
                        )
                        for local_index, row in enumerate(source_rows):
                            source_local_height_tangent[row] += (
                                weight * shape_a[local_index] * source_density_tangent
                            )
                        for local_index, row in enumerate(target_rows):
                            target_local_height_tangent[row] += (
                                weight * shape_b[local_index] * target_density_tangent
                            )

            for point, weighted_outward in _boundary_quadrature(moved_a, moved_b, overlap):
                closure = min(gap_at(point), 0.0)
                boundary_force = 0.5 * self.stiffness * closure**2 * weighted_outward
                source_translation_force += boundary_force
                shape_a = overlap.shape_values(moved_a, point)
                source_local_tangent[source_rows] += shape_a[:, None] * boundary_force
                shape_b = overlap.shape_values(moved_b, point)
                gap_gradient = np.zeros(dof_count)
                gap_gradient[columns] = np.r_[shape_a, -shape_b]
                boundary_tangent = self.stiffness * closure * np.outer(
                    weighted_outward, gap_gradient,
                )
                for local_index, row in enumerate(source_rows):
                    source_local_height_tangent[row] += shape_a[local_index] * boundary_tangent
            for point, weighted_outward in _boundary_quadrature(moved_b, moved_a, overlap):
                closure = min(gap_at(point), 0.0)
                boundary_force = 0.5 * self.stiffness * closure**2 * weighted_outward
                target_translation_force += boundary_force
                shape_b = overlap.shape_values(moved_b, point)
                target_local_tangent[target_rows] += shape_b[:, None] * boundary_force
                shape_a = overlap.shape_values(moved_a, point)
                gap_gradient = np.zeros(dof_count)
                gap_gradient[columns] = np.r_[shape_a, -shape_b]
                boundary_tangent = self.stiffness * closure * np.outer(
                    weighted_outward, gap_gradient,
                )
                for local_index, row in enumerate(target_rows):
                    target_local_height_tangent[row] += shape_b[local_index] * boundary_tangent

        source_translation_force += float(np.sum(normal_force[:source_count])) * self.normal
        target_translation_force += float(np.sum(normal_force[source_count:])) * self.normal
        return PointwisePatchState(
            overlap_area, pairs, len(candidates), triangles_checked, active_area, energy,
            normal_force, normal_tangent,
            source_translation_force, target_translation_force,
            source_local_tangent, target_local_tangent,
            source_local_height_tangent, target_local_height_tangent,
            minimum_vertex_gap,
        )

    def reduced_system(self, source_translation: np.ndarray,
                       target_translation: np.ndarray,
                       source_normal_dofs: np.ndarray,
                       target_normal_dofs: np.ndarray,
                       *, difference_step: float) -> PointwisePatchReducedSystem:
        """Build the conservative restricted-coordinate tangent on a smooth chart.

        Translation columns use the analytic transport residual. The normal
        block is the exact quadrature Hessian; its mixed transpose follows
        from the same scalar potential. A projection-topology transition must
        be resolved by path subdivision before this tangent is used.
        """

        step = float(difference_step)
        if not np.isfinite(step) or step <= 0.0:
            raise ValueError("difference_step must be finite and positive")
        variables = np.r_[source_translation, target_translation,
                          source_normal_dofs, target_normal_dofs]
        split = 6 + len(self.source_node_ids)

        def evaluate_at(values):
            state = self.evaluate(values[:3], values[3:6],
                                  values[6:split], values[split:])
            residual = np.r_[state.source_translation_residual,
                             state.target_translation_residual,
                             state.normal_residual]
            return state, residual

        state, residual = evaluate_at(variables)
        tangent = np.zeros((len(variables), len(variables)))
        tangent[6:, 6:] = state.normal_tangent
        for column in range(6):
            increment = np.zeros_like(variables)
            increment[column] = step
            plus, plus_residual = evaluate_at(variables + increment)
            minus, minus_residual = evaluate_at(variables - increment)
            if (plus.overlapping_pairs != state.overlapping_pairs
                    or minus.overlapping_pairs != state.overlapping_pairs
                    or plus.integration_triangles != state.integration_triangles
                    or minus.integration_triangles != state.integration_triangles):
                raise ValueError("pointwise projection transition needs path subdivision")
            tangent[:, column] = (plus_residual - minus_residual) / (2.0 * step)
        tangent[:6, 6:] = tangent[6:, :6].T
        scale = max(np.finfo(float).tiny, float(np.linalg.norm(tangent)))
        asymmetry = float(np.linalg.norm(tangent - tangent.T) / scale)
        return PointwisePatchReducedSystem(state, residual, tangent, asymmetry)
