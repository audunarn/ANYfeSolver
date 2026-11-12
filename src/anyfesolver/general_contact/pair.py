"""Exploratory planar Q4/S3 pair mortar under guided normal translation.

The projected reference overlap is fixed in this narrow development profile.
Candidate kinematics are checked so that the fixed overlap remains exact: each
face may translate only along one common normal, uniformly at its nodes.
This is an end-to-end force-map slice, not finite-sliding C1 admission.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import sparse

from .core import ContactDefinition, ContactSolverSettings, ContactValidationReport
from .overlap import PlanarFace, planar_projected_overlap, reference_shell_face


PROFILE = "PLANAR_Q4_PAIR_NORMAL_TRANSLATION_PENALTY_MORTAR_V0"
S3_PROFILE = "PLANAR_Q4_S3_PAIR_NORMAL_TRANSLATION_PENALTY_MORTAR_V0"
_MAX_VALIDATION_CANDIDATES = 100_000


@dataclass(frozen=True)
class _PairResults:
    profile: str
    minimum_gap: float
    penalty_energy: float
    integration_force: tuple[float, float, float]
    target_force: tuple[float, float, float]
    integration_moment: tuple[float, float, float]
    target_moment: tuple[float, float, float]
    active_pressure_dofs: int
    gap_tolerance_met: bool
    reference_overlap_area: float

    def to_dict(self) -> dict[str, object]:
        return {
            "profile": self.profile,
            "minimum_gap": self.minimum_gap,
            "penalty_energy": self.penalty_energy,
            "integration_force": list(self.integration_force),
            "target_force": list(self.target_force),
            "integration_moment": list(self.integration_moment),
            "target_moment": list(self.target_moment),
            "active_pressure_dofs": self.active_pressure_dofs,
            "gap_tolerance_met": self.gap_tolerance_met,
            "reference_overlap_area": self.reference_overlap_area,
        }


def _definition_parts(model, definition: ContactDefinition):
    issues: list[str] = []
    if (len(definition.surfaces), len(definition.rigid_bodies),
            len(definition.interactions)) != (2, 0, 1):
        issues.append("guided pair profile needs two surfaces and one interaction")
        return issues, None
    interaction = definition.interactions[0]
    surfaces = {surface.name: surface for surface in definition.surfaces}
    if (interaction.surface not in surfaces or interaction.target not in surfaces
            or interaction.surface == interaction.target):
        issues.append("pair interaction must name two different shell surfaces")
        return issues, None
    integration = surfaces[interaction.surface]
    target = surfaces[interaction.target]
    from anyfesolver.e4_pl_element import FORMULATION_ID as q4_id
    from anyfesolver.e4_pl_s3_v2d_element import FORMULATION_ID as s3_id

    for surface in (integration, target):
        for element_id in surface.element_ids:
            element = model.mesh.elements.get(element_id)
            if element is None or getattr(element, "formulation_id", None) not in {q4_id, s3_id}:
                issues.append(f"surface {surface.name}: element {element_id} is not admitted Q4/S3")
    if issues:
        return issues, None
    integration_nodes = {int(node) for element_id in integration.element_ids
                         for node in model.mesh.elements[element_id].node_ids}
    target_nodes = {int(node) for element_id in target.element_ids
                    for node in model.mesh.elements[element_id].node_ids}
    if integration_nodes & target_nodes:
        issues.append("the two guided contact surfaces must have disjoint nodes")
    return issues, (integration, target, interaction, integration_nodes, target_nodes)


def _reference_pair_normal(model, integration, faces_a, faces_b) -> np.ndarray:
    normal = np.cross(faces_a[0].corners[1] - faces_a[0].corners[0],
                      faces_a[0].corners[-1] - faces_a[0].corners[0])
    normal /= np.linalg.norm(normal)
    separation = np.mean(faces_a[0].corners, axis=0) - np.mean(
        faces_b[0].corners, axis=0
    )
    scale = max(
        float(np.max(np.linalg.norm(face.corners - face.corners[0], axis=1)))
        for face in (faces_a[0], faces_b[0])
    )
    tolerance = 256.0 * np.finfo(float).eps * scale
    signed_separation = float(separation @ normal)
    if signed_separation < -tolerance:
        normal = -normal
    elif abs(signed_separation) <= tolerance and integration.side != "midsurface":
        source_element = model.mesh.elements[faces_a[0].feature_id[1]]
        director = source_element.physical_reference_director
        if director is None:
            director = normal
        preferred = np.asarray(director, dtype=float)
        if integration.side == "top":
            preferred = -preferred
        if float(preferred @ normal) < 0.0:
            normal = -normal
    return normal


def validate_planar_shell_pair(model, definition: ContactDefinition, *,
                               require_reference_overlap: bool = True,
                               ) -> ContactValidationReport:
    issues, parts = _definition_parts(model, definition)
    profile = _profile_for_pair(model, definition)
    if parts is not None:
        integration, target, _interaction, _source_nodes, _target_nodes = parts
        if len(integration.element_ids) * len(target.element_ids) > _MAX_VALIDATION_CANDIDATES:
            issues.append("guided pair exceeds the validation candidate budget")
            return ContactValidationReport("invalid", tuple(issues), profile)
        try:
            faces_a = [reference_shell_face(model, 1, element_id, integration.side)
                       for element_id in integration.element_ids]
            faces_b = [reference_shell_face(model, 2, element_id, target.side)
                       for element_id in target.element_ids]
            normals = [np.cross(face.corners[1] - face.corners[0],
                                face.corners[-1] - face.corners[0])
                       for face in (*faces_a, *faces_b)]
            normals = [normal / np.linalg.norm(normal) for normal in normals]
            normal = normals[0]
            if any(abs(float(other @ normal)) < 1.0 - 1.0e-10
                   for other in normals):
                issues.append("guided pair faces must be parallel")
            normal = _reference_pair_normal(model, integration, faces_a, faces_b)
            for surface, faces, expected_sign in (
                (integration, faces_a, -1.0), (target, faces_b, 1.0),
            ):
                if surface.side == "midsurface":
                    continue
                side_sign = -1.0 if surface.side == "bottom" else 1.0
                for face in faces:
                    element = model.mesh.elements[face.feature_id[1]]
                    director = element.physical_reference_director
                    if director is None:
                        director = np.cross(face.corners[1] - face.corners[0],
                                            face.corners[-1] - face.corners[0])
                        director /= np.linalg.norm(director)
                    if expected_sign * side_sign * float(np.asarray(director) @ normal) <= 0.0:
                        issues.append(f"surface {surface.name}: physical side faces away from its pair")
                        break
            for name, faces in ((integration.name, faces_a), (target.name, faces_b)):
                origin = faces[0].corners[0]
                scale = max(float(np.max(np.linalg.norm(face.corners - face.corners[0], axis=1)))
                            for face in faces)
                distance = max(float(np.max(np.abs((face.corners - origin) @ normal)))
                               for face in faces)
                if distance > 1.0e-10 * scale:
                    issues.append(f"guided surface {name} must occupy one physical plane")
            if (require_reference_overlap and
                    not any(planar_projected_overlap(a, b) is not None
                            for a in faces_a for b in faces_b)):
                issues.append("the pair has no projected reference overlap")
        except ValueError as exc:
            issues.append(str(exc))
    return ContactValidationReport("invalid" if issues else "ok", tuple(issues), profile)


def _profile_for_pair(model, definition: ContactDefinition) -> str:
    from anyfesolver.e4_pl_s3_v2d_element import FORMULATION_ID as s3_id

    if any(getattr(model.mesh.elements.get(element_id), "formulation_id", None) == s3_id
           for surface in definition.surfaces for element_id in surface.element_ids):
        return S3_PROFILE
    return PROFILE


class _PlanarShellPairManager:
    def __init__(self, model, definition: ContactDefinition,
                 settings: ContactSolverSettings | None = None,
                 initial_displacement: np.ndarray | None = None):
        self.settings = ContactSolverSettings() if settings is None else settings
        if (len(definition.surfaces) == 2
                and len(definition.surfaces[0].element_ids)
                    * len(definition.surfaces[1].element_ids)
                > self.settings.maximum_candidate_pairs):
            raise MemoryError("contact candidate limit exceeded; no pair was discarded")
        report = validate_planar_shell_pair(model, definition)
        if not report.valid:
            raise ValueError("unsupported guided shell pair: " + "; ".join(report.issues))
        self.model = model
        self.definition = definition
        self.profile = report.profile
        self.stiffness = definition.interactions[0].normal_stiffness
        self.gap_tolerance = definition.interactions[0].gap_tolerance
        self.swept_trial_cutbacks = 0
        _issues, parts = _definition_parts(model, definition)
        integration, target, _interaction, integration_nodes, target_nodes = parts
        self.integration_nodes = tuple(sorted(integration_nodes))
        self.target_nodes = tuple(sorted(target_nodes))
        self.node_dofs = {
            node_id: np.asarray(model.mesh.dof_manager.get_node_dofs(node_id)[:6], dtype=np.intp)
            for node_id in (*self.integration_nodes, *self.target_nodes)
        }
        self.node_reference = {
            node_id: np.asarray(model.mesh.get_node(node_id).coords(), dtype=float)
            for node_id in self.node_dofs
        }
        self.faces_a = [reference_shell_face(model, 1, element_id, integration.side)
                        for element_id in integration.element_ids]
        self.faces_b = [reference_shell_face(model, 2, element_id, target.side)
                        for element_id in target.element_ids]
        self.normal = _reference_pair_normal(model, integration, self.faces_a, self.faces_b)
        self.face_gradients = {}
        from anyfesolver.e4_pl_s3_v2d_element import FORMULATION_ID as s3_id

        for face, side in (
            *((face, integration.side) for face in self.faces_a),
            *((face, target.side) for face in self.faces_b),
        ):
            element = model.mesh.elements[face.feature_id[1]]
            reference = np.asarray(element.get_node_coordinates(model.mesh), dtype=float)
            director = element.physical_reference_director
            if director is None:
                director = np.cross(reference[1] - reference[0],
                                    reference[-1] - reference[0])
                director /= np.linalg.norm(director)
            coordinate = {"bottom": -1.0, "midsurface": 0.0, "top": 1.0}[side]
            if element.formulation_id == s3_id:
                offset = float(element.material_surface_offset_from_reference(coordinate))
            else:
                offset = 0.5 * float(element.thickness) * coordinate
            self.face_gradients[face.feature_id] = np.r_[
                self.normal, offset * np.cross(director, self.normal)
            ]
        self.reference_scale = max(
            *(float(np.max(np.linalg.norm(face.corners - face.corners[0], axis=1)))
              for face in (*self.faces_a, *self.faces_b))
        )
        self.motion_tolerance = 256.0 * np.finfo(float).eps * self.reference_scale
        n_dof = model.mesh.dof_manager.total_dofs
        pressure_nodes = self.integration_nodes
        pressure_index = {node_id: row for row, node_id in enumerate(pressure_nodes)}
        support = np.zeros(len(pressure_nodes), dtype=float)
        weighted_reference_gap = np.zeros(len(pressure_nodes), dtype=float)
        rows: list[int] = []
        cols: list[int] = []
        data: list[float] = []
        self.samples: list[tuple] = []
        self.motion_faces = []
        for face in (*self.faces_a, *self.faces_b):
            element = model.mesh.elements[face.feature_id[1]]
            self.motion_faces.append(
                np.asarray([self.node_dofs[int(node)] for node in element.node_ids],
                           dtype=np.intp)
            )
        candidate_count = len(self.faces_a) * len(self.faces_b)
        if candidate_count > self.settings.maximum_candidate_pairs:
            raise MemoryError("contact candidate limit exceeded; no pair was discarded")
        overlap_area = 0.0
        for source in self.faces_a:
            source_element = model.mesh.elements[source.feature_id[1]]
            source_nodes = tuple(int(node) for node in source_element.node_ids)
            source_dofs = np.asarray([self.node_dofs[node] for node in source_nodes],
                                     dtype=np.intp)
            source_gradient = self.face_gradients[source.feature_id]
            for destination in self.faces_b:
                overlap = planar_projected_overlap(source, destination)
                if overlap is None:
                    continue
                destination_element = model.mesh.elements[destination.feature_id[1]]
                destination_nodes = tuple(int(node) for node in destination_element.node_ids)
                destination_dofs = np.asarray(
                    [self.node_dofs[node] for node in destination_nodes],
                    dtype=np.intp,
                )
                target_gradient = self.face_gradients[destination.feature_id]
                overlap_area += overlap.area
                centroid = np.mean(overlap.polygon, axis=0)
                for index, corner in enumerate(overlap.polygon):
                    next_corner = overlap.polygon[(index + 1) % len(overlap.polygon)]
                    triangle_area = 0.5 * float(np.linalg.norm(
                        np.cross(corner - centroid, next_corner - centroid)
                    ))
                    if triangle_area <= 0.0:
                        continue
                    triangle = (centroid, corner, next_corner)
                    for bary in ((2/3, 1/6, 1/6), (1/6, 2/3, 1/6),
                                 (1/6, 1/6, 2/3)):
                        point = sum(weight * vertex for weight, vertex in zip(bary, triangle))
                        source_shape = overlap.shape_values(source, point)
                        target_shape = overlap.shape_values(destination, point)
                        reference_gap = float(self.normal @ (
                            source_shape @ source.corners
                            - target_shape @ destination.corners
                        ))
                        if abs(reference_gap) <= self.motion_tolerance:
                            reference_gap = 0.0
                        weight = triangle_area / 3.0
                        self.samples.append((reference_gap, source_shape, target_shape,
                                             source_dofs, destination_dofs,
                                             source_gradient, target_gradient))
                        for local_row, node_id in enumerate(source_nodes):
                            pressure_row = pressure_index[node_id]
                            coefficient = weight * float(source_shape[local_row])
                            support[pressure_row] += coefficient
                            weighted_reference_gap[pressure_row] += coefficient * reference_gap
                            for nodes_dofs, signs, shape, gradient in (
                                (source_dofs, 1.0, source_shape, source_gradient),
                                (destination_dofs, -1.0, target_shape, target_gradient),
                            ):
                                for local_node, nodal_dofs in enumerate(nodes_dofs):
                                    for component in range(6):
                                        value = coefficient * signs * float(shape[local_node]) * gradient[component]
                                        if value:
                                            rows.append(pressure_row)
                                            cols.append(int(nodal_dofs[component]))
                                            data.append(value)
                # Vertex checks are independent of the integration points.
                for point in overlap.polygon:
                    source_shape = overlap.shape_values(source, point)
                    target_shape = overlap.shape_values(destination, point)
                    reference_gap = float(self.normal @ (
                        source_shape @ source.corners
                        - target_shape @ destination.corners
                    ))
                    if abs(reference_gap) <= self.motion_tolerance:
                        reference_gap = 0.0
                    self.samples.append((reference_gap, source_shape, target_shape,
                                         source_dofs, destination_dofs,
                                         source_gradient, target_gradient))
        if overlap_area <= 0.0:
            raise ValueError("guided pair has no reference overlap")
        active_rows = np.flatnonzero(support > 0.0)
        self.pressure_nodes = tuple(pressure_nodes[int(row)] for row in active_rows)
        self.support = support[active_rows]
        self.reference_weighted_gap = weighted_reference_gap[active_rows]
        pressure_selector = np.full(len(pressure_nodes), -1, dtype=int)
        pressure_selector[active_rows] = np.arange(len(active_rows))
        filtered = [(int(pressure_selector[row]), col, value)
                    for row, col, value in zip(rows, cols, data)
                    if pressure_selector[row] >= 0]
        self.gap_gradient = sparse.coo_matrix(
            ([value for _row, _col, value in filtered],
             ([row for row, _col, _value in filtered],
              [col for _row, col, _value in filtered])),
            shape=(len(active_rows), n_dof),
        ).tocsr()
        self.reference_overlap_area = overlap_area
        initial_u = (np.zeros(n_dof) if initial_displacement is None
                     else np.asarray(initial_displacement, dtype=float))
        self._check_motion(initial_u)
        initial_gap = self.evaluate(initial_u, tangent=False)[2].minimum_gap
        if initial_gap < -self.motion_tolerance:
            raise ValueError(f"initial contact overclosure {initial_gap:.6g} m is not admitted")

    def _check_motion(self, displacement: np.ndarray) -> None:
        u = np.asarray(displacement, dtype=float).reshape(-1)
        if u.size != self.model.mesh.dof_manager.total_dofs or not np.all(np.isfinite(u)):
            raise ValueError("contact displacement has invalid size or nonfinite values")
        common_tangent = None
        for face_dofs in self.motion_faces:
            movement = u[face_dofs[:, :3]]
            normal_movement = movement @ self.normal
            tangent_movement = movement - normal_movement[:, None] * self.normal
            if common_tangent is None:
                common_tangent = tangent_movement[0]
            if (float(np.max(np.linalg.norm(tangent_movement - common_tangent, axis=1)))
                    > self.motion_tolerance
                    or float(np.ptp(normal_movement)) > self.motion_tolerance):
                raise ValueError("guided pair profile requires uniform normal translation of each face and common tangential motion")
            rotations = u[face_dofs[:, 3:]]
            if float(np.max(np.linalg.norm(rotations, axis=1))) > 256.0 * np.finfo(float).eps:
                raise ValueError("guided pair profile requires fixed shell rotations")

    def _minimum_gap(self, u: np.ndarray) -> float:
        return min(
            reference_gap + float(
                source_shape @ (u[source_dofs] @ source_gradient)
                - target_shape @ (u[target_dofs] @ target_gradient)
            )
            for reference_gap, source_shape, target_shape,
                source_dofs, target_dofs, source_gradient, target_gradient
                in self.samples
        )

    def evaluate(self, displacement: np.ndarray, *, tangent: bool = True):
        self._check_motion(displacement)
        u = np.asarray(displacement, dtype=float).reshape(-1)
        gap = (self.reference_weighted_gap + np.asarray(self.gap_gradient @ u).reshape(-1)) / self.support
        active_gap = np.minimum(gap, 0.0)
        active = gap <= 0.0
        energy = 0.5 * self.stiffness * float(self.support @ (active_gap * active_gap))
        force = np.asarray(self.gap_gradient.T @ (self.stiffness * active_gap)).reshape(-1)
        stiffness = None
        if tangent:
            gradient = self.gap_gradient[active]
            stiffness = (
                self.stiffness * gradient.T
                @ sparse.diags(1.0 / self.support[active])
                @ gradient
            ).tocsr()
        source_force = np.zeros(3)
        target_force = np.zeros(3)
        source_moment = np.zeros(3)
        target_moment = np.zeros(3)
        for node_ids, total_force, total_moment in (
            (self.integration_nodes, source_force, source_moment),
            (self.target_nodes, target_force, target_moment),
        ):
            for node_id in node_ids:
                dofs = self.node_dofs[node_id]
                physical_force = -force[dofs[:3]]
                total_force += physical_force
                total_moment += np.cross(
                    self.node_reference[node_id] + u[dofs[:3]], physical_force
                ) + (-force[dofs[3:]])
        minimum_gap = self._minimum_gap(u)
        result = _PairResults(
            self.profile,
            minimum_gap, energy,
            tuple(float(value) for value in source_force),
            tuple(float(value) for value in target_force),
            tuple(float(value) for value in source_moment),
            tuple(float(value) for value in target_moment),
            int(np.count_nonzero(gap < 0.0)),
            minimum_gap >= -self.gap_tolerance,
            self.reference_overlap_area,
        )
        return force, stiffness, result

    def admissible_step_fraction(self, start: np.ndarray, end: np.ndarray) -> float:
        self._check_motion(start)
        self._check_motion(end)
        first = self._minimum_gap(np.asarray(start, dtype=float).reshape(-1))
        last = self._minimum_gap(np.asarray(end, dtype=float).reshape(-1))
        if last >= -self.gap_tolerance:
            return 1.0
        if first <= -self.gap_tolerance:
            self.swept_trial_cutbacks += 1
            return 0.0
        fraction = (first + self.gap_tolerance) / (first - last)
        self.swept_trial_cutbacks += 1
        return max(0.0, 0.98 * fraction)

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
                "maximum_candidate_pairs": self.settings.maximum_candidate_pairs,
            },
        }
