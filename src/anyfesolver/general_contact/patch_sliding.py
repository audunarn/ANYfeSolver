"""Restricted multi-face planar shell sliding through the static contact hook."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import sparse

from .core import ContactDefinition, ContactSolverSettings, ContactValidationReport
from .moving import bound_swept_moving_mortar_gap
from .overlap import reference_shell_face
from .pair import _definition_parts, _reference_pair_normal, validate_planar_shell_pair
from .patch_mortar import MovingPatchMortar, PatchFace
from .pointwise_patch import PointwisePatchPenalty


PROFILE = "PLANAR_Q4_S3_PATCH_RESTRICTED_SLIDING_MORTAR_DEV_V0"
AUGMENTED_PROFILE = "PLANAR_Q4_S3_PATCH_RESTRICTED_SLIDING_AUGMENTED_MORTAR_DEV_V0"
POINTWISE_PROFILE = "PLANAR_Q4_S3_PATCH_RESTRICTED_SLIDING_POINTWISE_PENALTY_DEV_V0"


def _patch_faces(model, surface, body_id: int) -> tuple[PatchFace, ...]:
    return tuple(
        PatchFace(
            reference_shell_face(model, body_id, element_id, surface.side),
            tuple(int(node) for node in model.mesh.elements[element_id].node_ids),
        )
        for element_id in surface.element_ids
    )


def validate_restricted_sliding_patch(
        model, definition: ContactDefinition,
        settings: ContactSolverSettings | None = None,
        ) -> ContactValidationReport:
    report = validate_planar_shell_pair(
        model, definition, require_reference_overlap=False,
    )
    issues = list(report.issues)
    controls = ContactSolverSettings() if settings is None else settings
    if (controls.normal_discretization == "pointwise_penalty"
            and any(item.normal_law != "penalty" for item in definition.interactions)):
        issues.append("pointwise patch specialization requires penalty normal law")
    if len(definition.surfaces) == 2 and all(
        len(surface.element_ids) == 1 for surface in definition.surfaces
    ):
        issues.append("patch sliding requires multiple selected faces")
    if not issues:
        _issues, parts = _definition_parts(model, definition)
        integration, target, interaction, _source_nodes, _target_nodes = parts
        source_faces = _patch_faces(model, integration, 1)
        target_faces = _patch_faces(model, target, 2)
        normal = _reference_pair_normal(
            model, integration,
            tuple(face.geometry for face in source_faces),
            tuple(face.geometry for face in target_faces),
        )
        try:
            arguments = {
                "maximum_candidate_pairs": controls.maximum_candidate_pairs,
                "maximum_normal_dofs": controls.maximum_patch_normal_dofs,
            }
            if controls.normal_discretization == "pointwise_penalty":
                PointwisePatchPenalty(
                    source_faces, target_faces, normal,
                    interaction.normal_stiffness,
                    integration_subdivisions=controls.pointwise_integration_subdivisions,
                    maximum_integration_triangles=controls.maximum_pointwise_triangles,
                    **arguments,
                )
            else:
                MovingPatchMortar(
                    source_faces, target_faces, normal,
                    interaction.normal_stiffness, **arguments,
                )
        except (ValueError, MemoryError) as exc:
            issues.append(str(exc))
    profile = (POINTWISE_PROFILE if controls.normal_discretization == "pointwise_penalty"
               else AUGMENTED_PROFILE if any(
        item.normal_law == "augmented" for item in definition.interactions
    ) else PROFILE)
    return ContactValidationReport(
        "invalid" if issues else "ok", tuple(issues), profile,
    )


@dataclass(frozen=True)
class PatchSweptBound:
    tolerance_certified: bool
    conservative_lower_gap: float
    overlapping_pair_intervals: int


@dataclass(frozen=True)
class _SlidingPatchResults:
    profile: str
    current_overlap_area: float
    overlapping_pairs: int
    candidate_pairs: int
    pressure_rank: int | None
    active_pressure_dofs: int | None
    minimum_sampled_gap: float
    conservative_lower_gap: float
    gap_tolerance_met: bool
    gap_triangles_checked: int
    gap_check_method: str
    gap_resource_exhausted: bool
    penalty_energy: float
    integration_force: tuple[float, float, float]
    target_force: tuple[float, float, float]
    integration_moment: tuple[float, float, float]
    target_moment: tuple[float, float, float]
    tangent_asymmetry: float | None
    normal_law: str = "penalty"
    multiplier_norm: float = 0.0
    minimum_positive_support_m2: float | None = None
    maximum_positive_support_m2: float | None = None
    pressure_gradient_condition: float | None = None
    minimum_weighted_gap_m: float | None = None
    maximum_algorithmic_pressure_Pa: float | None = 0.0
    weighted_to_physical_gap_span_m: float | None = None
    active_pointwise_area_m2: float | None = None
    pointwise_integration_triangles: int | None = None

    def to_dict(self) -> dict[str, object]:
        return {
            "profile": self.profile,
            "current_overlap_area": self.current_overlap_area,
            "overlapping_pairs": self.overlapping_pairs,
            "candidate_pairs": self.candidate_pairs,
            "pressure_rank": self.pressure_rank,
            "active_pressure_dofs": self.active_pressure_dofs,
            "minimum_positive_support_m2": self.minimum_positive_support_m2,
            "maximum_positive_support_m2": self.maximum_positive_support_m2,
            "pressure_gradient_condition": self.pressure_gradient_condition,
            "minimum_weighted_gap_m": self.minimum_weighted_gap_m,
            "maximum_algorithmic_pressure_Pa": self.maximum_algorithmic_pressure_Pa,
            "weighted_to_physical_gap_span_m": self.weighted_to_physical_gap_span_m,
            "active_pointwise_area_m2": self.active_pointwise_area_m2,
            "pointwise_integration_triangles": self.pointwise_integration_triangles,
            "minimum_sampled_gap": self.minimum_sampled_gap,
            "conservative_lower_gap": self.conservative_lower_gap,
            "gap_tolerance_met": self.gap_tolerance_met,
            "gap_triangles_checked": self.gap_triangles_checked,
            "gap_check_method": self.gap_check_method,
            "gap_resource_exhausted": self.gap_resource_exhausted,
            "penalty_energy": (self.penalty_energy if self.normal_law == "penalty"
                               else None),
            "algorithmic_contact_potential": self.penalty_energy,
            "normal_law": self.normal_law,
            "multiplier_norm": self.multiplier_norm,
            "integration_force": list(self.integration_force),
            "target_force": list(self.target_force),
            "integration_moment": list(self.integration_moment),
            "target_moment": list(self.target_moment),
            "tangent_asymmetry": self.tangent_asymmetry,
            "minimum_gap_certified": self.gap_tolerance_met,
        }


class _RestrictedSlidingPatchFEManager:
    """Map one global patch pressure space to authoritative shell DOFs."""

    def __init__(self, model, definition: ContactDefinition,
                 settings: ContactSolverSettings | None = None,
                 initial_displacement: np.ndarray | None = None):
        self.settings = ContactSolverSettings() if settings is None else settings
        report = validate_restricted_sliding_patch(model, definition, self.settings)
        if not report.valid:
            raise ValueError("unsupported sliding patch: " + "; ".join(report.issues))
        _issues, parts = _definition_parts(model, definition)
        integration, target, interaction, _source_nodes, _target_nodes = parts
        self.model = model
        self.definition = definition
        self.profile = report.profile
        self.stiffness = interaction.normal_stiffness
        self.normal_law = interaction.normal_law
        self.gap_tolerance = interaction.gap_tolerance
        self.difference_step = self.settings.sliding_difference_step
        if self.difference_step is None:
            raise ValueError("restricted sliding requires sliding_difference_step")
        self.swept_trial_cutbacks = 0
        source_faces = _patch_faces(model, integration, 1)
        target_faces = _patch_faces(model, target, 2)
        self.normal = _reference_pair_normal(
            model, integration,
            tuple(face.geometry for face in source_faces),
            tuple(face.geometry for face in target_faces),
        )
        self.pointwise = self.settings.normal_discretization == "pointwise_penalty"
        arguments = {
            "maximum_candidate_pairs": self.settings.maximum_candidate_pairs,
            "maximum_normal_dofs": self.settings.maximum_patch_normal_dofs,
        }
        if self.pointwise:
            self.patch = PointwisePatchPenalty(
                source_faces, target_faces, self.normal, self.stiffness,
                integration_subdivisions=self.settings.pointwise_integration_subdivisions,
                maximum_integration_triangles=self.settings.maximum_pointwise_triangles,
                **arguments,
            )
        else:
            self.patch = MovingPatchMortar(
                source_faces, target_faces, self.normal, self.stiffness,
                **arguments,
            )
        self.normal_multipliers = np.zeros(len(self.patch.source_node_ids))
        self.node_ids = (self.patch.source_node_ids, self.patch.target_node_ids)
        self.node_dofs = tuple(
            np.asarray([model.mesh.dof_manager.get_node_dofs(node)[:6]
                        for node in ids], dtype=np.intp)
            for ids in self.node_ids
        )
        self.node_reference = {
            node: np.asarray(model.mesh.get_node(node).coords(), dtype=float)
            for ids in self.node_ids for node in ids
        }
        scale = max(float(np.max(np.linalg.norm(
            face.geometry.corners - face.geometry.corners[0], axis=1
        ))) for face in (*source_faces, *target_faces))
        self.motion_tolerance = 256.0 * np.finfo(float).eps * scale
        surface_levers: dict[tuple[int, int], np.ndarray] = {}
        from anyfesolver.e4_pl_s3_v2d_element import FORMULATION_ID as s3_id

        for body, (surface, faces) in enumerate(zip(
            (integration, target), (source_faces, target_faces)
        )):
            coordinate = {"bottom": -1.0, "midsurface": 0.0, "top": 1.0}[surface.side]
            for face in faces:
                element = model.mesh.elements[face.geometry.feature_id[1]]
                director = element.physical_reference_director
                if director is None:
                    reference = np.asarray(element.get_node_coordinates(model.mesh), dtype=float)
                    director = np.cross(reference[1] - reference[0],
                                        reference[-1] - reference[0])
                    director /= np.linalg.norm(director)
                offset = (
                    float(element.material_surface_offset_from_reference(coordinate))
                    if element.formulation_id == s3_id else
                    0.5 * coordinate * float(element.thickness)
                )
                lever = offset * director
                for node in face.node_ids:
                    key = (body, node)
                    if (key in surface_levers and
                            float(np.linalg.norm(surface_levers[key] - lever))
                            > self.motion_tolerance):
                        raise ValueError(
                            f"shared patch node {node} has inconsistent physical directors"
                        )
                    surface_levers[key] = lever
        projection = np.eye(3) - np.outer(self.normal, self.normal)
        self.surface_levers = surface_levers
        reduced_count = 6 + sum(len(ids) for ids in self.node_ids)
        self.height_offsets = (6, 6 + len(self.node_ids[0]))
        rows: list[int] = []
        cols: list[int] = []
        data: list[float] = []
        for body, (ids, dofs) in enumerate(zip(self.node_ids, self.node_dofs)):
            for local, (node, nodal_dofs) in enumerate(zip(ids, dofs)):
                lever = surface_levers[(body, node)]
                rotation_gradient = np.cross(lever, self.normal)
                for component in range(3):
                    for structural in range(3):
                        coefficient = projection[component, structural] / len(ids)
                        if coefficient:
                            rows.append(3 * body + component)
                            cols.append(int(nodal_dofs[structural]))
                            data.append(float(coefficient))
                    tangent_rotation = np.cross(lever, projection[component]) / len(ids)
                    for structural in range(3):
                        coefficient = tangent_rotation[structural]
                        if coefficient:
                            rows.append(3 * body + component)
                            cols.append(int(nodal_dofs[3 + structural]))
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
        n_dof = model.mesh.dof_manager.total_dofs
        self.work_map = sparse.coo_matrix(
            (data, (rows, cols)), shape=(reduced_count, n_dof)
        ).tocsr()
        initial = (np.zeros(n_dof) if initial_displacement is None
                   else np.asarray(initial_displacement, dtype=float).reshape(-1))
        q = self._reduced_pose(initial)
        initial_gap = self._certificate(q, self.motion_tolerance)
        if initial_gap.minimum_sampled_gap < -self.motion_tolerance:
            raise ValueError("sliding patch has unresolved initial penetration")
        if not initial_gap.tolerance_met:
            raise ValueError("sliding patch initial gap could not be certified")

    def _reduced_pose(self, displacement: np.ndarray) -> np.ndarray:
        u = np.asarray(displacement, dtype=float).reshape(-1)
        if (u.size != self.model.mesh.dof_manager.total_dofs
                or not np.all(np.isfinite(u))):
            raise ValueError("sliding patch displacement has invalid size or nonfinite values")
        projection = np.eye(3) - np.outer(self.normal, self.normal)
        for dofs in self.node_dofs:
            tangent = u[dofs[:, :3]] @ projection
            if float(np.max(np.linalg.norm(
                    tangent - np.mean(tangent, axis=0), axis=1))) > self.motion_tolerance:
                raise ValueError("restricted sliding needs uniform patch tangential translation")
            if float(np.max(np.linalg.norm(u[dofs[:, 3:]], axis=1))) > 256.0 * np.finfo(float).eps:
                raise ValueError("restricted sliding requires fixed shell rotations")
        return np.asarray(self.work_map @ u).reshape(-1)

    def _certificate(self, q: np.ndarray, tolerance: float):
        split = self.height_offsets[1]
        return self.patch.certify_gap(
            q[:3], q[3:6], q[6:split], q[split:],
            gap_tolerance=tolerance,
            maximum_triangles_per_pair=self.settings.maximum_gap_triangles,
            maximum_total_triangles=self.settings.maximum_gap_triangles,
        )

    def swept_trial_bound(self, start: np.ndarray, end: np.ndarray) -> PatchSweptBound:
        first = self._reduced_pose(start)
        last = self._reduced_pose(end)
        split = self.height_offsets[1]
        certified = True
        lower = float("inf")
        intervals = 0
        for face_a in self.patch.source_faces:
            rows = np.asarray([self.patch.source_index[node]
                               for node in face_a.node_ids])
            for face_b in self.patch.target_faces:
                columns = np.asarray([self.patch.target_index[node]
                                      for node in face_b.node_ids])
                bound = bound_swept_moving_mortar_gap(
                    face_a.geometry, face_b.geometry, self.normal,
                    first[:3], last[:3], first[3:6], last[3:6],
                    first[6:split][rows], last[6:split][rows],
                    first[split:][columns], last[split:][columns],
                    gap_tolerance=self.gap_tolerance,
                )
                if bound.overlap_interval is not None:
                    intervals += 1
                certified = certified and bound.tolerance_certified
                lower = min(lower, bound.conservative_lower_gap)
        if certified:
            return PatchSweptBound(True, lower, intervals)
        relative_motion = ((last[:3] - last[3:6])
                           - (first[:3] - first[3:6]))
        tangent_motion = relative_motion - self.normal * float(relative_motion @ self.normal)
        if np.any(tangent_motion != 0.0):
            return PatchSweptBound(False, lower, intervals)
        endpoints = (self._certificate(first, self.gap_tolerance),
                     self._certificate(last, self.gap_tolerance))
        if not all(item.tolerance_met for item in endpoints):
            return PatchSweptBound(False, lower, intervals)
        return PatchSweptBound(
            True, min(item.conservative_lower_gap for item in endpoints),
            intervals,
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
        contract = {
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
                "maximum_patch_normal_dofs": self.settings.maximum_patch_normal_dofs,
                "sliding_difference_step": self.difference_step,
                "maximum_gap_triangles": self.settings.maximum_gap_triangles,
            },
        }
        if self.pointwise:
            contract["settings"]["normal_discretization"] = "pointwise_penalty"
            contract["settings"]["pointwise_integration_subdivisions"] = (
                self.settings.pointwise_integration_subdivisions
            )
            contract["settings"]["maximum_pointwise_triangles"] = (
                self.settings.maximum_pointwise_triangles
            )
        if self.normal_law == "augmented":
            contract["settings"]["maximum_augmentations"] = self.settings.maximum_augmentations
            contract["settings"]["pressure_tolerance"] = self.settings.pressure_tolerance
        return contract

    def set_normal_multipliers(self, estimate: np.ndarray) -> None:
        if self.normal_law != "augmented":
            raise ValueError("normal multipliers require augmented contact")
        self.normal_multipliers = self.patch._owned_multiplier(estimate)

    def projected_normal_multipliers(self, displacement: np.ndarray) -> np.ndarray:
        if self.normal_law != "augmented":
            raise ValueError("normal multiplier update requires augmented contact")
        q = self._reduced_pose(displacement)
        split = self.height_offsets[1]
        state = self.patch.evaluate(
            q[:3], q[3:6], q[6:split], q[split:],
            multiplier_estimate=self.normal_multipliers,
        )
        return self.patch.projected_multiplier(state, self.normal_multipliers)

    def _local_full_force(self, normal_residual: np.ndarray,
                          local_tangential: tuple[np.ndarray, np.ndarray]) -> np.ndarray:
        """Map local tangential and normal work to authoritative shell DOFs."""

        reduced_normal = np.r_[np.zeros(6), normal_residual]
        force = np.asarray(self.work_map.T @ reduced_normal).reshape(-1)
        for body, (ids, dofs, local) in enumerate(zip(
            self.node_ids, self.node_dofs, local_tangential,
        )):
            for index, (node, nodal_dofs) in enumerate(zip(ids, dofs)):
                force[nodal_dofs[:3]] += local[index]
                force[nodal_dofs[3:]] += np.cross(
                    self.surface_levers[(body, node)], local[index],
                )
        return force

    def _pointwise_full_force(self, state) -> np.ndarray:
        return self._local_full_force(
            state.normal_residual,
            (state.source_local_tangent_residual,
             state.target_local_tangent_residual),
        )

    def _mortar_full_force(self, q: np.ndarray, state) -> np.ndarray:
        split = self.height_offsets[1]
        local = self.patch.local_tangential_residual(
            q[:3], q[3:6], q[6:split], q[split:],
            multiplier_estimate=self.normal_multipliers,
            state=state,
        )
        return self._local_full_force(state.normal_residual, local)

    def _mortar_full_tangent(self, q: np.ndarray, state,
                             force: np.ndarray) -> sparse.csr_matrix:
        """Differentiate the mapped mortar residual at fixed multiplier."""

        split = self.height_offsets[1]
        derivative = np.zeros((len(force), len(q)))
        multiplier = self.normal_multipliers.copy()
        active = ((state.support > 0.0)
                  & (state.normalized_gap - multiplier / self.stiffness <= 0.0))

        def evaluate_at(values):
            trial = self.patch.evaluate(
                values[:3], values[3:6], values[6:split], values[split:],
                multiplier_estimate=multiplier,
            )
            local_force = self._mortar_full_force(values, trial)
            trial_active = ((trial.support > 0.0)
                            & (trial.normalized_gap - multiplier / self.stiffness <= 0.0))
            return trial, trial_active, local_force

        for column in range(len(q)):
            perturbation = np.zeros_like(q)
            perturbation[column] = self.difference_step
            plus, plus_active, plus_force = evaluate_at(q + perturbation)
            minus, minus_active, minus_force = evaluate_at(q - perturbation)
            plus_matches = (np.array_equal(plus_active, active)
                            and plus.overlapping_pairs == state.overlapping_pairs)
            minus_matches = (np.array_equal(minus_active, active)
                             and minus.overlapping_pairs == state.overlapping_pairs)
            if plus_matches and minus_matches:
                derivative[:, column] = (
                    plus_force - minus_force
                ) / (2.0 * self.difference_step)
            elif plus_matches:
                derivative[:, column] = (
                    plus_force - force
                ) / self.difference_step
            elif minus_matches:
                derivative[:, column] = (
                    force - minus_force
                ) / self.difference_step
            else:
                derivative[:, column] = (
                    plus_force - force
                ) / self.difference_step
        return (sparse.csr_matrix(derivative) @ self.work_map).tocsr()

    def _pointwise_full_tangent(self, q: np.ndarray, state) -> sparse.csr_matrix:
        """Differentiate the local force map along admitted restricted motion."""

        count = self.model.mesh.dof_manager.total_dofs
        derivative = np.zeros((count, len(q)))
        normal_map = self.work_map.T.tocsr()[:, 6:]
        derivative[:, 6:] = np.asarray(normal_map @ state.normal_tangent)
        for body, (ids, dofs, local_derivative) in enumerate(zip(
            self.node_ids, self.node_dofs,
            (state.source_local_tangent_height_derivative,
             state.target_local_tangent_height_derivative),
        )):
            for index, (node, nodal_dofs) in enumerate(zip(ids, dofs)):
                contribution = local_derivative[index]
                derivative[nodal_dofs[:3], 6:] += contribution
                lever = self.surface_levers[(body, node)]
                derivative[nodal_dofs[3:], 6:] += np.cross(
                    lever[:, None], contribution, axis=0,
                )
        split = self.height_offsets[1]
        for column in range(6):
            perturbation = np.zeros_like(q)
            perturbation[column] = self.difference_step
            plus = self.patch.evaluate(
                (q + perturbation)[:3], (q + perturbation)[3:6],
                (q + perturbation)[6:split], (q + perturbation)[split:],
            )
            minus = self.patch.evaluate(
                (q - perturbation)[:3], (q - perturbation)[3:6],
                (q - perturbation)[6:split], (q - perturbation)[split:],
            )
            if (plus.overlapping_pairs != state.overlapping_pairs
                    or minus.overlapping_pairs != state.overlapping_pairs
                    or plus.integration_triangles != state.integration_triangles
                    or minus.integration_triangles != state.integration_triangles):
                raise ValueError("pointwise projection transition needs path subdivision")
            derivative[:, column] = (
                self._pointwise_full_force(plus)
                - self._pointwise_full_force(minus)
            ) / (2.0 * self.difference_step)
        return (sparse.csr_matrix(derivative) @ self.work_map).tocsr()

    def evaluate(self, displacement: np.ndarray, *, tangent: bool = True):
        q = self._reduced_pose(displacement)
        split = self.height_offsets[1]
        certificate = self._certificate(q, self.gap_tolerance)
        if self.pointwise:
            state = self.patch.evaluate(q[:3], q[3:6], q[6:split], q[split:])
            force = self._pointwise_full_force(state)
            if tangent:
                stiffness = self._pointwise_full_tangent(q, state)
                tangent_scale = max(np.finfo(float).tiny, float(sparse.linalg.norm(stiffness)))
                asymmetry = float(sparse.linalg.norm(stiffness - stiffness.T) / tangent_scale)
            else:
                stiffness = None
                asymmetry = None
        elif tangent:
            state = self.patch.evaluate(
                q[:3], q[3:6], q[6:split], q[split:],
                multiplier_estimate=self.normal_multipliers,
            )
            force = self._mortar_full_force(q, state)
            stiffness = self._mortar_full_tangent(q, state, force)
            tangent_scale = max(np.finfo(float).tiny, float(sparse.linalg.norm(stiffness)))
            asymmetry = float(sparse.linalg.norm(stiffness - stiffness.T) / tangent_scale)
        else:
            state = self.patch.evaluate(
                q[:3], q[3:6], q[6:split], q[split:],
                multiplier_estimate=self.normal_multipliers,
            )
            force = self._mortar_full_force(q, state)
            stiffness = None
            asymmetry = None
        u = np.asarray(displacement, dtype=float).reshape(-1)
        body_forces = []
        body_moments = []
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
        if self.pointwise:
            pressure_rank = active_pressure_dofs = None
            minimum_support = maximum_support = minimum_weighted_gap = gap_span = None
            pressure_condition = None
            maximum_algorithmic_pressure = None
            pointwise_area = state.active_quadrature_area
            pointwise_triangles = state.integration_triangles
        else:
            supported = state.support > 0.0
            pressure_rank = state.pressure_rank
            active_pressure_dofs = int(np.count_nonzero(
                supported & (state.normalized_gap - self.normal_multipliers
                             / self.stiffness < 0.0)
            ))
            pressure_condition = state.pressure_gradient_condition
            pointwise_area = pointwise_triangles = None
            if np.any(supported):
                minimum_support = float(np.min(state.support[supported]))
                maximum_support = float(np.max(state.support[supported]))
                minimum_weighted_gap = float(np.min(state.normalized_gap[supported]))
                gap_span = minimum_weighted_gap - certificate.conservative_lower_gap
                maximum_algorithmic_pressure = float(np.max(np.maximum(
                    0.0, self.normal_multipliers[supported]
                    - self.stiffness * state.normalized_gap[supported],
                )))
            else:
                minimum_support = maximum_support = minimum_weighted_gap = gap_span = None
                maximum_algorithmic_pressure = 0.0
        result = _SlidingPatchResults(
            self.profile, state.overlap_area,
            state.overlapping_pairs, state.candidate_pairs,
            pressure_rank, active_pressure_dofs,
            certificate.minimum_sampled_gap,
            certificate.conservative_lower_gap,
            certificate.tolerance_met,
            certificate.triangles_checked,
            ",".join(certificate.methods) if certificate.methods else "empty",
            certificate.resource_exhausted,
            state.energy, body_forces[0], body_forces[1],
            body_moments[0], body_moments[1], asymmetry,
            self.normal_law, float(np.linalg.norm(self.normal_multipliers)),
            minimum_support, maximum_support, pressure_condition,
            minimum_weighted_gap, maximum_algorithmic_pressure, gap_span,
            pointwise_area, pointwise_triangles,
        )
        return force, stiffness, result
