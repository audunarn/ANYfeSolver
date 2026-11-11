"""Restricted moving-mortar work map onto physical Q4/S3 shell DOFs.

This is a local development owner, not yet a public static solver profile.
Each selected face may translate uniformly in the contact tangent plane;
normal nodal displacement may vary. Candidate shell rotations are fixed, but
their physical-skin virtual work is retained in the assembled force map.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import sparse

from .core import ContactDefinition, ContactSolverSettings, ContactValidationReport
from .moving import (
    MovingSweptGapBound,
    bound_swept_moving_mortar_gap,
    certify_swept_moving_mortar_gap,
    certify_moving_mortar_gap,
    moving_mortar_reduced_system,
    moving_mortar_state,
    moving_mortar_translation_residual,
)
from .overlap import reference_shell_face
from .pair import _definition_parts, _reference_pair_normal, validate_planar_shell_pair


PROFILE = "PLANAR_Q4_S3_PAIR_RESTRICTED_SLIDING_MORTAR_DEV_V0"


def validate_restricted_sliding_pair(model, definition: ContactDefinition) -> ContactValidationReport:
    report = validate_planar_shell_pair(model, definition)
    issues = list(report.issues)
    if len(definition.surfaces) == 2 and any(
        len(surface.element_ids) != 1 for surface in definition.surfaces
    ):
        issues.append("restricted sliding requires one face per surface")
    return ContactValidationReport("invalid" if issues else "ok", tuple(issues), PROFILE)


@dataclass(frozen=True)
class _SlidingResults:
    profile: str
    current_overlap_area: float
    minimum_sampled_gap: float
    conservative_lower_gap: float
    gap_tolerance_met: bool
    gap_triangles_checked: int
    gap_check_method: str
    penalty_energy: float
    integration_force: tuple[float, float, float]
    target_force: tuple[float, float, float]
    integration_moment: tuple[float, float, float]
    target_moment: tuple[float, float, float]
    tangent_asymmetry: float | None

    def to_dict(self) -> dict[str, object]:
        return {
            "profile": self.profile,
            "current_overlap_area": self.current_overlap_area,
            "minimum_sampled_gap": self.minimum_sampled_gap,
            "conservative_lower_gap": self.conservative_lower_gap,
            "gap_tolerance_met": self.gap_tolerance_met,
            "gap_triangles_checked": self.gap_triangles_checked,
            "gap_check_method": self.gap_check_method,
            "penalty_energy": self.penalty_energy,
            "integration_force": list(self.integration_force),
            "target_force": list(self.target_force),
            "integration_moment": list(self.integration_moment),
            "target_moment": list(self.target_moment),
            "tangent_asymmetry": self.tangent_asymmetry,
            "minimum_gap_certified": self.gap_tolerance_met,
        }


class _RestrictedSlidingPairFEManager:
    """Assemble one moving face pair on the authoritative shell coordinates."""

    def __init__(self, model, definition: ContactDefinition,
                 settings: ContactSolverSettings | None = None,
                 *, difference_step: float | None = None,
                 maximum_gap_triangles: int | None = None,
                 initial_displacement: np.ndarray | None = None):
        self.settings = ContactSolverSettings() if settings is None else settings
        report = validate_restricted_sliding_pair(model, definition)
        if not report.valid:
            raise ValueError("unsupported sliding pair: " + "; ".join(report.issues))
        _issues, parts = _definition_parts(model, definition)
        integration, target, interaction, _source_nodes, _target_nodes = parts
        if len(integration.element_ids) != 1 or len(target.element_ids) != 1:
            raise ValueError("restricted sliding owner currently needs one face per surface")
        self.model = model
        self.definition = definition
        self.profile = PROFILE
        self.stiffness = interaction.normal_stiffness
        self.gap_tolerance = interaction.gap_tolerance
        self.swept_trial_cutbacks = 0
        selected_step = (self.settings.sliding_difference_step
                         if difference_step is None else difference_step)
        if selected_step is None:
            raise ValueError("restricted sliding requires an explicit difference_step")
        self.difference_step = float(selected_step)
        if not np.isfinite(self.difference_step) or self.difference_step <= 0.0:
            raise ValueError("difference_step must be finite and positive")
        if maximum_gap_triangles is None:
            maximum_gap_triangles = self.settings.maximum_gap_triangles
        if type(maximum_gap_triangles) is not int or maximum_gap_triangles < 1:
            raise ValueError("maximum_gap_triangles must be a positive integer")
        self.maximum_gap_triangles = maximum_gap_triangles
        self.faces = (
            reference_shell_face(model, 1, integration.element_ids[0], integration.side),
            reference_shell_face(model, 2, target.element_ids[0], target.side),
        )
        self.normal = _reference_pair_normal(
            model, integration, (self.faces[0],), (self.faces[1],)
        )
        self.node_ids = tuple(
            tuple(int(node) for node in model.mesh.elements[element_id].node_ids)
            for element_id in (integration.element_ids[0], target.element_ids[0])
        )
        self.node_dofs = tuple(
            np.asarray([model.mesh.dof_manager.get_node_dofs(node)[:6]
                        for node in ids], dtype=np.intp)
            for ids in self.node_ids
        )
        self.node_reference = {
            node: np.asarray(model.mesh.get_node(node).coords(), dtype=float)
            for ids in self.node_ids for node in ids
        }
        projection = np.eye(3) - np.outer(self.normal, self.normal)
        rows: list[int] = []
        cols: list[int] = []
        data: list[float] = []
        reduced_count = 6 + sum(len(ids) for ids in self.node_ids)
        self.height_offsets = (6, 6 + len(self.node_ids[0]))
        self.face_dofs = []
        for body, (ids, dofs, side) in enumerate(zip(
            self.node_ids, self.node_dofs, (integration.side, target.side)
        )):
            element_id = (integration.element_ids[0], target.element_ids[0])[body]
            element = model.mesh.elements[element_id]
            director = element.physical_reference_director
            if director is None:
                reference = np.asarray(element.get_node_coordinates(model.mesh), dtype=float)
                director = np.cross(reference[1] - reference[0],
                                    reference[-1] - reference[0])
                director /= np.linalg.norm(director)
            from anyfesolver.e4_pl_s3_v2d_element import FORMULATION_ID as s3_id

            coordinate = {"bottom": -1.0, "midsurface": 0.0, "top": 1.0}[side]
            offset = (float(element.material_surface_offset_from_reference(coordinate))
                      if element.formulation_id == s3_id else
                      0.5 * coordinate * float(element.thickness))
            rotation_gradient = offset * np.cross(director, self.normal)
            for local, nodal_dofs in enumerate(dofs):
                for component in range(3):
                    for structural in range(3):
                        coefficient = projection[component, structural] / len(ids)
                        if coefficient:
                            rows.append(3 * body + component)
                            cols.append(int(nodal_dofs[structural]))
                            data.append(float(coefficient))
                    coefficient = self.normal[component]
                    if coefficient:
                        rows.append(self.height_offsets[body] + local)
                        cols.append(int(nodal_dofs[component]))
                        data.append(float(coefficient))
                    coefficient = rotation_gradient[component]
                    if coefficient:
                        rows.append(self.height_offsets[body] + local)
                        cols.append(int(nodal_dofs[3 + component]))
                        data.append(float(coefficient))
            self.face_dofs.append(dofs)
        n_dof = model.mesh.dof_manager.total_dofs
        self.work_map = sparse.coo_matrix(
            (data, (rows, cols)), shape=(reduced_count, n_dof)
        ).tocsr()
        scale = max(float(np.max(np.linalg.norm(face.corners - face.corners[0], axis=1)))
                    for face in self.faces)
        self.motion_tolerance = 256.0 * np.finfo(float).eps * scale
        initial = (np.zeros(n_dof) if initial_displacement is None
                   else np.asarray(initial_displacement, dtype=float).reshape(-1))
        initial_q = self._reduced_pose(initial)
        split = self.height_offsets[1]
        certificate = certify_moving_mortar_gap(
            *self.faces, self.normal,
            initial_q[:3], initial_q[3:6], initial_q[6:split], initial_q[split:],
            gap_tolerance=self.motion_tolerance,
            maximum_triangles=self.maximum_gap_triangles,
        )
        if certificate.minimum_sampled_gap < -self.motion_tolerance:
            raise ValueError("restricted sliding has unresolved initial penetration")
        if not certificate.tolerance_met:
            raise ValueError("restricted sliding initial gap could not be certified")

    def _reduced_pose(self, displacement: np.ndarray) -> np.ndarray:
        u = np.asarray(displacement, dtype=float).reshape(-1)
        if (u.size != self.model.mesh.dof_manager.total_dofs
                or not np.all(np.isfinite(u))):
            raise ValueError("sliding pair displacement has invalid size or nonfinite values")
        projection = np.eye(3) - np.outer(self.normal, self.normal)
        for dofs in self.face_dofs:
            tangent = u[dofs[:, :3]] @ projection
            if float(np.max(np.linalg.norm(tangent - np.mean(tangent, axis=0), axis=1))) > self.motion_tolerance:
                raise ValueError("restricted sliding requires uniform face tangential translation")
            if float(np.max(np.linalg.norm(u[dofs[:, 3:]], axis=1))) > 256.0 * np.finfo(float).eps:
                raise ValueError("restricted sliding requires fixed shell rotations")
        return np.asarray(self.work_map @ u).reshape(-1)

    def swept_trial_bound(self, start: np.ndarray, end: np.ndarray):
        """Diagnose the represented linear trial path without admitting it."""

        first = self._reduced_pose(start)
        last = self._reduced_pose(end)
        split = self.height_offsets[1]
        global_bound = bound_swept_moving_mortar_gap(
            *self.faces, self.normal,
            first[:3], last[:3], first[3:6], last[3:6],
            first[6:split], last[6:split], first[split:], last[split:],
            gap_tolerance=self.gap_tolerance,
        )
        if global_bound.tolerance_certified:
            return global_bound
        relative_motion = ((last[:3] - last[3:6])
                           - (first[:3] - first[3:6]))
        tangent_motion = relative_motion - self.normal * float(relative_motion @ self.normal)
        if np.any(tangent_motion != 0.0):
            return certify_swept_moving_mortar_gap(
                *self.faces, self.normal,
                first[:3], last[:3], first[3:6], last[3:6],
                first[6:split], last[6:split], first[split:], last[split:],
                gap_tolerance=self.gap_tolerance,
                maximum_boxes=self.maximum_gap_triangles,
            )
        certificates = tuple(
            certify_moving_mortar_gap(
                *self.faces, self.normal,
                pose[:3], pose[3:6], pose[6:split], pose[split:],
                gap_tolerance=self.gap_tolerance,
                maximum_triangles=self.maximum_gap_triangles,
            )
            for pose in (first, last)
        )
        if not all(item.tolerance_met for item in certificates):
            return global_bound
        return MovingSweptGapBound(
            global_bound.overlap_interval,
            min(item.conservative_lower_gap for item in certificates),
            True,
        )

    def admissible_step_fraction(self, start: np.ndarray, end: np.ndarray) -> float:
        if self.swept_trial_bound(start, end).tolerance_certified:
            return 1.0
        first = np.asarray(start, dtype=float).reshape(-1)
        last = np.asarray(end, dtype=float).reshape(-1)
        lower, upper = 0.0, 1.0
        for _ in range(30):
            middle = 0.5 * (lower + upper)
            candidate = first + middle * (last - first)
            if self.swept_trial_bound(first, candidate).tolerance_certified:
                lower = middle
            else:
                upper = middle
        self.swept_trial_cutbacks += 1
        return max(0.0, 0.98 * lower)

    def restart_contract(self) -> dict[str, object]:
        return {
            "profile": self.profile,
            "surfaces": [
                {"name": item.name, "element_ids": list(item.element_ids),
                 "side": item.side}
                for item in self.definition.surfaces
            ],
            "interactions": [
                {"name": item.name, "surface": item.surface,
                 "target": item.target, "normal_stiffness": item.normal_stiffness,
                 "gap_tolerance": item.gap_tolerance,
                 "normal_law": item.normal_law}
                for item in self.definition.interactions
            ],
            "settings": {
                "pair_motion": self.settings.pair_motion,
                "maximum_candidate_pairs": self.settings.maximum_candidate_pairs,
                "sliding_difference_step": self.difference_step,
                "maximum_gap_triangles": self.maximum_gap_triangles,
            },
        }

    def evaluate(self, displacement: np.ndarray, *, tangent: bool = True):
        q = self._reduced_pose(displacement)
        split = self.height_offsets[1]
        gap_certificate = certify_moving_mortar_gap(
            *self.faces, self.normal, q[:3], q[3:6], q[6:split], q[split:],
            gap_tolerance=self.gap_tolerance,
            maximum_triangles=self.maximum_gap_triangles,
        )
        if tangent:
            reduced = moving_mortar_reduced_system(
                *self.faces, self.normal, self.stiffness,
                q[:3], q[3:6], q[6:split], q[split:],
                difference_step=self.difference_step,
            )
            state = reduced.state
            reduced_force = reduced.residual
            stiffness = (self.work_map.T @ sparse.csr_matrix(reduced.tangent)
                         @ self.work_map).tocsr()
            asymmetry = reduced.relative_tangent_asymmetry
        else:
            state = moving_mortar_state(
                *self.faces, self.normal, self.stiffness,
                q[:3], q[3:6], q[6:split], q[split:],
            )
            source, target = moving_mortar_translation_residual(
                *self.faces, self.normal, self.stiffness,
                q[:3], q[3:6], q[6:split], q[split:],
            )
            reduced_force = np.r_[source, target, state.normal_residual]
            stiffness = None
            asymmetry = None
        force = np.asarray(self.work_map.T @ reduced_force).reshape(-1)
        body_forces = []
        body_moments = []
        u = np.asarray(displacement, dtype=float).reshape(-1)
        for ids, dofs in zip(self.node_ids, self.node_dofs):
            total_force = np.zeros(3)
            total_moment = np.zeros(3)
            for node, nodal_dofs in zip(ids, dofs):
                physical_force = -force[nodal_dofs[:3]]
                total_force += physical_force
                total_moment += np.cross(
                    self.node_reference[node] + u[nodal_dofs[:3]], physical_force
                ) - force[nodal_dofs[3:]]
            body_forces.append(tuple(float(value) for value in total_force))
            body_moments.append(tuple(float(value) for value in total_moment))
        result = _SlidingResults(
            PROFILE, state.overlap_area,
            gap_certificate.minimum_sampled_gap,
            gap_certificate.conservative_lower_gap,
            gap_certificate.tolerance_met,
            gap_certificate.triangles_checked,
            gap_certificate.method,
            state.energy, body_forces[0], body_forces[1],
            body_moments[0], body_moments[1], asymmetry,
        )
        return force, stiffness, result
