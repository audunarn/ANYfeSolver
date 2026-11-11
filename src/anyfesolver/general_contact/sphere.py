"""Fixed radial rigid obstacles against qualified small-rotation shell skins.

This is a primal nodal-pressure penalty profile.  It uses the shell's
physical reference side and differentiates the exact radial distance at the
contact quadrature points.  The represented surface is independently bounded
by adaptive parametric cells before an increment can be accepted.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import sparse

from .core import (
    ContactDefinition, ContactResults, ContactSolverSettings,
    ContactValidationReport,
)
from .overlap import reference_shell_face


PROFILE = "Q4_S3_FIXED_SPHERE_PENALTY_MORTAR_STATIC_DEV_V0"
CYLINDER_PROFILE = "Q4_S3_FIXED_INFINITE_CYLINDER_PENALTY_MORTAR_STATIC_DEV_V0"
MOVING_SPHERE_PROFILE = "Q4_S3_TRANSLATING_SPHERE_PENALTY_MORTAR_FORCE_STATIC_DEV_V0"
MOVING_CYLINDER_PROFILE = "Q4_S3_TRANSLATING_INFINITE_CYLINDER_PENALTY_MORTAR_FORCE_STATIC_DEV_V0"


class _GapCertificateLimitError(MemoryError):
    """A trial path could not be certified within its geometry budget."""


def _skew(vector: np.ndarray) -> np.ndarray:
    x, y, z = vector
    return np.array(((0.0, -z, y), (z, 0.0, -x), (-y, x, 0.0)))


def _sphere_parts(model, definition: ContactDefinition):
    issues: list[str] = []
    if (len(definition.surfaces), len(definition.rigid_bodies),
            len(definition.interactions)) != (1, 1, 1):
        return ["radial profile requires one shell surface, one obstacle and one interaction"], None
    surface = definition.surfaces[0]
    body = definition.rigid_bodies[0]
    interaction = definition.interactions[0]
    if body.kind not in {"sphere", "cylinder"}:
        issues.append("radial profile requires an analytic sphere or cylinder")
    obstacle = body.kind
    if interaction.surface != surface.name or interaction.target != body.name:
        issues.append(f"{obstacle} interaction must name its shell surface and rigid obstacle")
    if interaction.normal_law != "penalty":
        issues.append(f"{obstacle} profile currently admits penalty normal law only")
    if surface.side == "midsurface":
        issues.append(f"{obstacle} contact requires an explicit top or bottom shell side")
    from anyfesolver.e4_pl_element import FORMULATION_ID as q4_id
    from anyfesolver.e4_pl_s3_v2d_element import FORMULATION_ID as s3_id
    from anymaterial.isotropic import IsotropicMaterial

    for element in model.mesh.elements.values():
        material = model.get_material(element.material_name)
        if type(material) is not IsotropicMaterial or material.hardening_curve is not None:
            issues.append(f"{obstacle} static profile currently admits elastic isotropic materials only")
            break

    if not issues:
        center = np.asarray(body.point, dtype=float)
        metric = (np.eye(3) if body.kind == "sphere" else
                  np.eye(3) - np.outer(body.normal, body.normal))
        for element_id in surface.element_ids:
            element = model.mesh.elements.get(element_id)
            if element is None or getattr(element, "formulation_id", None) not in {q4_id, s3_id}:
                issues.append(f"{obstacle} surface element {element_id} is not admitted Q4/S3")
                continue
            try:
                face = reference_shell_face(model, 1, element_id, surface.side)
            except ValueError as exc:
                issues.append(f"{obstacle} surface element {element_id}: {exc}")
                continue
            director = element.physical_reference_director
            if director is None:
                reference = np.asarray(element.get_node_coordinates(model.mesh), dtype=float)
                director = np.cross(reference[1] - reference[0],
                                    reference[-1] - reference[0])
                director /= np.linalg.norm(director)
            outward = np.asarray(director, dtype=float)
            if surface.side == "bottom":
                outward = -outward
            projected = metric @ (center - np.mean(face.corners, axis=0))
            if float(projected @ outward) <= 0.0:
                issues.append(f"{obstacle} lies behind selected side of element {element_id}")
    return issues, (surface, body, interaction) if not issues else None


def validate_sphere_shell_contact(model, definition: ContactDefinition,
                                  settings: ContactSolverSettings | None = None,
                                  ) -> ContactValidationReport:
    controls = ContactSolverSettings() if settings is None else settings
    issues, parts = _sphere_parts(model, definition)
    obstacle = definition.rigid_bodies[0].kind if definition.rigid_bodies else "radial"
    if controls.pair_motion != "general":
        issues.append(f"{obstacle} contact requires pair_motion='general'")
    if controls.maximum_shell_tilt is None or controls.maximum_shell_tilt > 0.1:
        issues.append(f"{obstacle} contact requires explicit maximum_shell_tilt at or below 0.1 rad")
    if parts is not None:
        surface, _body, _interaction = parts
        if len(surface.element_ids) > controls.maximum_candidate_pairs:
            issues.append(f"{obstacle} contact candidate limit exceeded")
        node_ids = {int(node) for element_id in surface.element_ids
                    for node in model.mesh.elements[element_id].node_ids}
        if len(node_ids) > controls.maximum_patch_normal_dofs:
            issues.append(f"{obstacle} pressure-node resource limit exceeded")
    moving = parts is not None and parts[1].path_translation is not None
    profile = (
        MOVING_CYLINDER_PROFILE if obstacle == "cylinder" and moving else
        CYLINDER_PROFILE if obstacle == "cylinder" else
        MOVING_SPHERE_PROFILE if moving else PROFILE
    )
    return ContactValidationReport("invalid" if issues else "ok", tuple(issues), profile)


@dataclass(frozen=True)
class _SphereQuadrature:
    shape: np.ndarray
    weight: float
    reference_point: np.ndarray
    jacobian: np.ndarray


@dataclass(frozen=True)
class _SphereFace:
    element_id: int
    node_ids: tuple[int, ...]
    dofs: np.ndarray
    reference: np.ndarray
    director: np.ndarray
    offset: float
    element: object
    quadrature: tuple[_SphereQuadrature, ...]

    def shape(self, coordinates: tuple[float, float]) -> np.ndarray:
        if len(self.node_ids) == 3:
            shape, _dxi, _deta = self.element.compute_native_surface_shape_functions(
                float(coordinates[0]), float(coordinates[1])
            )
        else:
            shape, _dxi, _deta = self.element.compute_shape_functions(
                float(coordinates[0]), float(coordinates[1])
            )
        return np.asarray(shape, dtype=float)

    def point(self, coordinates: tuple[float, float], u: np.ndarray) -> np.ndarray:
        shape = self.shape(coordinates)
        jacobian = _physical_jacobian(shape, self.director, self.offset)
        return (shape @ self.reference + self.offset * self.director
                + jacobian @ u[self.dofs.reshape(-1)])


@dataclass(frozen=True)
class SphereContactResults(ContactResults):
    minimum_sampled_gap: float
    minimum_weighted_gap: float
    pressure_node_count: int
    gap_cells_checked: int
    candidate_faces: int

    def to_dict(self) -> dict[str, object]:
        values = super().to_dict()
        values.update({
            "conservative_lower_gap": self.minimum_gap,
            "minimum_sampled_gap": self.minimum_sampled_gap,
            "minimum_weighted_gap": self.minimum_weighted_gap,
            "pressure_node_count": self.pressure_node_count,
            "gap_cells_checked": self.gap_cells_checked,
            "candidate_faces": self.candidate_faces,
        })
        return values


def _physical_jacobian(shape: np.ndarray, director: np.ndarray,
                       offset: float) -> np.ndarray:
    jacobian = np.zeros((3, 6 * len(shape)))
    rotation = -offset * _skew(director)
    for index, value in enumerate(shape):
        jacobian[:, 6 * index:6 * index + 3] = value * np.eye(3)
        jacobian[:, 6 * index + 3:6 * index + 6] = value * rotation
    return jacobian


def _prepare_face(model, surface, element_id: int) -> _SphereFace:
    element = model.mesh.elements[element_id]
    reference = np.asarray(element.get_node_coordinates(model.mesh), dtype=float)
    node_ids = tuple(int(node) for node in element.node_ids)
    dofs = np.asarray([model.mesh.dof_manager.get_node_dofs(node)[:6]
                       for node in node_ids], dtype=np.intp)
    director = element.physical_reference_director
    if director is None:
        director = np.cross(reference[1] - reference[0],
                            reference[-1] - reference[0])
        director /= np.linalg.norm(director)
    director = np.asarray(director, dtype=float)
    from anyfesolver.e4_pl_s3_v2d_element import FORMULATION_ID as s3_id

    side = 1.0 if surface.side == "top" else -1.0
    offset = (float(element.material_surface_offset_from_reference(side))
              if element.formulation_id == s3_id else
              0.5 * side * float(element.thickness))
    if element.formulation_id == s3_id:
        points = tuple((float(point[0]), float(point[1]), float(weight))
                       for point, weight in zip(element.gauss_points,
                                                element.gauss_weights))
    else:
        abscissae, weights = np.polynomial.legendre.leggauss(4)
        points = tuple((float(r), float(s), float(wr * ws))
                       for r, wr in zip(abscissae, weights)
                       for s, ws in zip(abscissae, weights))
    quadrature = []
    for r, s, weight in points:
        if element.formulation_id == s3_id:
            shape, dxi, deta = element.compute_native_surface_shape_functions(r, s)
        else:
            shape, dxi, deta = element.compute_shape_functions(r, s)
        shape = np.asarray(shape, dtype=float)
        measure = float(np.linalg.norm(np.cross(dxi @ reference, deta @ reference)))
        if not np.isfinite(measure) or measure <= 0.0:
            raise ValueError(f"radial contact face {element_id} has invalid reference measure")
        quadrature.append(_SphereQuadrature(
            shape, weight * measure,
            shape @ reference + offset * director,
            _physical_jacobian(shape, director, offset),
        ))
    return _SphereFace(element_id, node_ids, dofs, reference, director,
                       offset, element, tuple(quadrature))


def _initial_cells(face: _SphereFace):
    if len(face.node_ids) == 3:
        return [((0.0, 0.0), (1.0, 0.0), (0.0, 1.0))]
    return [((-1.0, -1.0), (1.0, -1.0), (1.0, 1.0), (-1.0, 1.0))]


def _split_cell(cell):
    if len(cell) == 3:
        a, b, c = (np.asarray(point) for point in cell)
        ab, bc, ca = (a + b) / 2, (b + c) / 2, (c + a) / 2
        return ((tuple(a), tuple(ab), tuple(ca)),
                (tuple(ab), tuple(b), tuple(bc)),
                (tuple(ca), tuple(bc), tuple(c)),
                (tuple(ab), tuple(bc), tuple(ca)))
    a, b, c, d = (np.asarray(point) for point in cell)
    ab, bc, cd, da = (a + b) / 2, (b + c) / 2, (c + d) / 2, (d + a) / 2
    center = (a + b + c + d) / 4
    return ((tuple(a), tuple(ab), tuple(center), tuple(da)),
            (tuple(ab), tuple(b), tuple(bc), tuple(center)),
            (tuple(center), tuple(bc), tuple(c), tuple(cd)),
            (tuple(da), tuple(center), tuple(cd), tuple(d)))


class _SphereShellManager:
    def __init__(self, model, definition: ContactDefinition,
                 settings: ContactSolverSettings | None = None,
                 initial_displacement: np.ndarray | None = None):
        controls = ContactSolverSettings() if settings is None else settings
        report = validate_sphere_shell_contact(model, definition, controls)
        if not report.valid:
            raise ValueError("unsupported radial contact: " + "; ".join(report.issues))
        self.model = model
        self.definition = definition
        self.settings = controls
        self.profile = report.profile
        surface, body, interaction = _sphere_parts(model, definition)[1]
        self.side = surface.side
        self.kind = body.kind
        self.reference_center = np.asarray(body.point, dtype=float)
        self.path_translation = (np.zeros(3) if body.path_translation is None
                                 else np.asarray(body.path_translation, dtype=float))
        self.path_factor = 0.0
        self.center = self.reference_center.copy()
        self.radius = float(body.radius)
        self.axis = (None if body.kind == "sphere" else
                     np.asarray(body.normal, dtype=float))
        self.metric = (np.eye(3) if self.axis is None else
                       np.eye(3) - np.outer(self.axis, self.axis))
        self.stiffness = interaction.normal_stiffness
        self.gap_tolerance = interaction.gap_tolerance
        self.swept_trial_cutbacks = 0
        self.faces = tuple(_prepare_face(model, surface, element_id)
                           for element_id in surface.element_ids)
        self.node_ids = tuple(sorted({node for face in self.faces
                                      for node in face.node_ids}))
        self.pressure_index = {node: index for index, node in enumerate(self.node_ids)}
        n_dof = model.mesh.dof_manager.total_dofs
        self.n_dof = n_dof
        initial = (np.zeros(n_dof) if initial_displacement is None
                   else np.asarray(initial_displacement, dtype=float).reshape(-1))
        if initial.shape != (n_dof,) or not np.all(np.isfinite(initial)):
            raise ValueError(f"{self.kind} initial displacement is invalid")
        self._check_tilt(initial)
        self._check_side(initial)
        roundoff = 256.0 * np.finfo(float).eps * max(
            self.radius, *(float(np.max(np.linalg.norm(face.reference, axis=1)))
                           for face in self.faces),
        )
        lower, sampled, _checked = self._certify_gap(initial, None, roundoff)
        if sampled < -roundoff:
            raise ValueError(f"{self.kind} has unresolved initial penetration {sampled:.6g} m")
        if lower < -roundoff:
            raise ValueError(f"{self.kind} initial gap could not be certified")

    def restart_contract(self) -> dict[str, object]:
        surface = self.definition.surfaces[0]
        body = self.definition.rigid_bodies[0]
        interaction = self.definition.interactions[0]
        return {
            "profile": self.profile,
            "surface": {"name": surface.name, "element_ids": list(surface.element_ids),
                        "side": surface.side},
            "body": {"name": body.name, "kind": body.kind,
                     "center": list(body.point),
                     **({"axis": list(body.normal)} if body.kind == "cylinder" else {}),
                     **({"path_translation": list(body.path_translation)}
                        if body.path_translation is not None else {}),
                     "radius": body.radius},
            "interaction": {"name": interaction.name,
                            "normal_stiffness": interaction.normal_stiffness,
                            "gap_tolerance": interaction.gap_tolerance,
                            "normal_law": interaction.normal_law},
            "settings": {"pair_motion": self.settings.pair_motion,
                         "maximum_candidate_pairs": self.settings.maximum_candidate_pairs,
                         "maximum_patch_normal_dofs": self.settings.maximum_patch_normal_dofs,
                         "maximum_gap_triangles": self.settings.maximum_gap_triangles,
                         "maximum_shell_tilt": self.settings.maximum_shell_tilt},
        }

    def center_at(self, factor: float) -> np.ndarray:
        value = float(factor)
        if not np.isfinite(value) or value < 0.0:
            raise ValueError("rigid-tool path factor must be finite and nonnegative")
        center = self.reference_center + value * self.path_translation
        if not np.all(np.isfinite(center)):
            raise ValueError("rigid-tool path position is nonfinite")
        return center

    def set_path_factor(self, factor: float) -> None:
        self.center = self.center_at(factor)
        self.path_factor = float(factor)

    def _project(self, points: np.ndarray,
                 center: np.ndarray | None = None) -> np.ndarray:
        origin = self.center if center is None else center
        return (np.asarray(points, dtype=float) - origin) @ self.metric

    def _check_tilt(self, u: np.ndarray) -> None:
        for face in self.faces:
            rotation = u[face.dofs[:, 3:]]
            tilt = rotation - np.outer(rotation @ face.director, face.director)
            if float(np.max(np.linalg.norm(tilt, axis=1))) > self.settings.maximum_shell_tilt:
                raise ValueError(f"{self.kind} shell tilt exceeds the admitted small-rotation limit")

    def _check_side(self, u: np.ndarray,
                    center: np.ndarray | None = None) -> None:
        sign = 1.0 if self.side == "top" else -1.0
        for face in self.faces:
            parameters = ((1.0 / 3.0, 1.0 / 3.0) if len(face.node_ids) == 3
                          else (0.0, 0.0))
            shape = face.shape(parameters)
            rotation = shape @ u[face.dofs[:, 3:]]
            director = face.director + np.cross(rotation, face.director)
            outward = sign * director
            position = face.point(parameters, u)
            if float((-self._project(position, center)) @ outward) <= 0.0:
                raise ValueError(
                    f"{self.kind} lies behind selected {self.side} side of face {face.element_id}"
                )

    def _check_geometry(self, u: np.ndarray) -> None:
        for face in self.faces:
            translations = u[face.dofs[:, :3]]
            rotations = u[face.dofs[:, 3:]]
            physical = (face.reference + face.offset * face.director
                        + translations + face.offset * np.cross(
                            rotations, face.director,
                        ))
            points = (((1.0 / 3.0, 1.0 / 3.0),) if len(face.node_ids) == 3
                      else ((-1.0, -1.0), (1.0, -1.0),
                            (1.0, 1.0), (-1.0, 1.0), (0.0, 0.0)))
            for r, s in points:
                if len(face.node_ids) == 3:
                    _shape, dxi, deta = face.element.compute_native_surface_shape_functions(r, s)
                else:
                    _shape, dxi, deta = face.element.compute_shape_functions(r, s)
                reference_cross = np.cross(dxi @ face.reference, deta @ face.reference)
                physical_cross = np.cross(dxi @ physical, deta @ physical)
                reference_area = float(reference_cross @ reference_cross)
                if (not np.all(np.isfinite(physical_cross))
                        or float(physical_cross @ reference_cross)
                        <= 1.0e-8 * reference_area):
                    raise ValueError(f"{self.kind} contact face {face.element_id} is inverted or collapsed")

    def _check_swept_geometry(self, start: np.ndarray, end: np.ndarray) -> None:
        for face in self.faces:
            configurations = []
            for pose in (start, end):
                translations = pose[face.dofs[:, :3]]
                rotations = pose[face.dofs[:, 3:]]
                configurations.append(
                    face.reference + face.offset * face.director
                    + translations + face.offset * np.cross(rotations, face.director)
                )
            points = (((1.0 / 3.0, 1.0 / 3.0),) if len(face.node_ids) == 3
                      else ((-1.0, -1.0), (1.0, -1.0),
                            (1.0, 1.0), (-1.0, 1.0)))
            for r, s in points:
                if len(face.node_ids) == 3:
                    _shape, dxi, deta = face.element.compute_native_surface_shape_functions(r, s)
                else:
                    _shape, dxi, deta = face.element.compute_shape_functions(r, s)
                reference_cross = np.cross(dxi @ face.reference, deta @ face.reference)
                scale = float(reference_cross @ reference_cross)

                def orientation(fraction: float) -> float:
                    physical = ((1.0 - fraction) * configurations[0]
                                + fraction * configurations[1])
                    cross = np.cross(dxi @ physical, deta @ physical)
                    return float(cross @ reference_cross)

                initial, middle, final = (orientation(value)
                                          for value in (0.0, 0.5, 1.0))
                quadratic = 2.0 * (initial + final - 2.0 * middle)
                linear = final - initial - quadratic
                minimum = min(initial, final)
                if quadratic > 0.0:
                    vertex = float(np.clip(-linear / (2.0 * quadratic), 0.0, 1.0))
                    minimum = min(minimum, quadratic * vertex * vertex
                                  + linear * vertex + initial)
                if minimum <= 1.0e-8 * scale:
                    raise ValueError(
                        f"{self.kind} contact face {face.element_id} collapses along the trial path"
                    )

    def _certify_gap(self, start: np.ndarray, end: np.ndarray | None,
                     tolerance: float, *, start_factor: float | None = None,
                     end_factor: float | None = None) -> tuple[float, float, int]:
        start_center = (self.center if start_factor is None else
                        self.center_at(start_factor))
        end_center = (self.center if end_factor is None else
                      self.center_at(end_factor))
        self._check_tilt(start)
        self._check_geometry(start)
        if end is not None:
            self._check_tilt(end)
            self._check_geometry(end)
            self._check_swept_geometry(start, end)
        minimum_lower = float("inf")
        minimum_sampled = float("inf")
        checked = 0
        for face in self.faces:
            stack = _initial_cells(face)
            while stack:
                cell = stack.pop()
                checked += 1
                if checked > self.settings.maximum_gap_triangles:
                    raise _GapCertificateLimitError(
                        f"{self.kind} gap certificate resource limit exceeded "
                        f"(sampled={minimum_sampled:.6g} m, "
                        f"lower={minimum_lower:.6g} m, "
                        f"tolerance={tolerance:.6g} m)"
                    )
                parameter_samples = (*cell, tuple(np.mean(cell, axis=0)))
                vertices = np.asarray([face.point(point, start) for point in cell])
                radial_vertices = self._project(vertices, start_center)
                if end is not None:
                    end_vertices = np.asarray([face.point(point, end)
                                               for point in cell])
                    radial_vertices = np.vstack((
                        radial_vertices, self._project(end_vertices, end_center),
                    ))
                lower_corner = np.min(radial_vertices, axis=0)
                upper_corner = np.max(radial_vertices, axis=0)
                closest = np.maximum(lower_corner,
                                     np.minimum(0.0, upper_corner))
                lower = float(np.linalg.norm(closest) - self.radius)
                start_samples = self._project(np.asarray([
                    face.point(point, start) for point in parameter_samples
                ]), start_center)
                if end is None:
                    gaps = np.linalg.norm(start_samples, axis=1) - self.radius
                    minimum_sampled = min(minimum_sampled, float(np.min(gaps)))
                else:
                    end_samples = self._project(np.asarray([
                        face.point(point, end) for point in parameter_samples
                    ]), end_center)
                    motion = end_samples - start_samples
                    squared = np.einsum("ij,ij->i", motion, motion)
                    parameter = np.zeros(len(parameter_samples))
                    moving = squared > 0.0
                    parameter[moving] = np.clip(
                        np.einsum("ij,ij->i", -start_samples, motion)[moving]
                        / squared[moving], 0.0, 1.0,
                    )
                    closest_path = start_samples + parameter[:, None] * motion
                    gaps = np.linalg.norm(closest_path, axis=1) - self.radius
                    minimum_sampled = min(minimum_sampled, float(np.min(gaps)))
                    reference_start = radial_vertices[0]
                    reference_end = radial_vertices[len(cell)]
                    direction = reference_end - reference_start
                    squared_direction = float(direction @ direction)
                    parameter0 = (0.0 if squared_direction == 0.0 else
                                  float(np.clip((-reference_start) @ direction
                                                / squared_direction, 0.0, 1.0)))
                    path_distance = float(np.linalg.norm(
                        reference_start + parameter0 * direction
                    ))
                    span = max(float(np.linalg.norm(
                        (face.point(point, pose) - face.point(cell[0], pose))
                        @ self.metric
                    )) for point in cell for pose in (start, end))
                    lower = max(lower, path_distance - span - self.radius)
                if minimum_sampled < -tolerance:
                    return min(minimum_lower, lower), minimum_sampled, checked
                if lower < -tolerance:
                    stack.extend(_split_cell(cell))
                else:
                    minimum_lower = min(minimum_lower, lower)
        return minimum_lower, minimum_sampled, checked

    def evaluate(self, displacement: np.ndarray, *, tangent: bool = True):
        u = np.asarray(displacement, dtype=float).reshape(-1)
        if u.shape != (self.n_dof,) or not np.all(np.isfinite(u)):
            raise ValueError(f"{self.kind} contact displacement is invalid")
        self._check_tilt(u)
        self._check_side(u)
        self._check_geometry(u)
        count = len(self.node_ids)
        support = np.zeros(count)
        weighted_gap = np.zeros(count)
        row_indices: list[int] = []
        col_indices: list[int] = []
        coefficients: list[float] = []
        samples = []
        for face in self.faces:
            local_dofs = face.dofs.reshape(-1)
            pressure_rows = [self.pressure_index[node] for node in face.node_ids]
            local_u = u[local_dofs]
            for point in face.quadrature:
                position = point.reference_point + point.jacobian @ local_u
                relative = self._project(position)
                radius = float(np.linalg.norm(relative))
                if radius <= 0.0:
                    raise ValueError(f"{self.kind} projection reached its undefined axis or center")
                normal = relative / radius
                gap = radius - self.radius
                local_gradient = normal @ point.jacobian
                for local, pressure_row in enumerate(pressure_rows):
                    weight = point.weight * point.shape[local]
                    support[pressure_row] += weight
                    weighted_gap[pressure_row] += weight * gap
                    row_indices.extend([pressure_row] * len(local_dofs))
                    col_indices.extend(local_dofs)
                    coefficients.extend(weight * local_gradient)
                samples.append((face, point, local_dofs, normal, radius))
        if np.any(support <= 0.0):
            raise ValueError(f"{self.kind} contact pressure support is nonpositive")
        roundoff = 256.0 * np.finfo(float).eps * self.radius
        weighted_gap[np.abs(weighted_gap) <= roundoff * support] = 0.0
        normalized_gap = weighted_gap / support
        active_gap = np.minimum(normalized_gap, 0.0)
        signed_pressure = self.stiffness * active_gap
        energy = 0.5 * self.stiffness * float(support @ (active_gap * active_gap))
        gradient = sparse.coo_matrix(
            (coefficients, (row_indices, col_indices)),
            shape=(count, self.n_dof),
        ).tocsr()
        force = np.asarray(gradient.T @ signed_pressure).reshape(-1)
        stiffness = None
        if tangent:
            active = normalized_gap <= 0.0
            active_gradient = gradient[active]
            stiffness = (self.stiffness * active_gradient.T
                         @ sparse.diags(1.0 / support[active])
                         @ active_gradient).tocsr()
            geometric_rows: list[int] = []
            geometric_cols: list[int] = []
            geometric_values: list[float] = []
            for face, point, local_dofs, normal, radius in samples:
                pressure = sum(signed_pressure[self.pressure_index[node]] * value
                               for node, value in zip(face.node_ids, point.shape))
                if pressure == 0.0:
                    continue
                projector = (self.metric - np.outer(normal, normal)) / radius
                local_hessian = (point.weight * pressure
                                 * point.jacobian.T @ projector @ point.jacobian)
                geometric_rows.extend(np.repeat(local_dofs, len(local_dofs)))
                geometric_cols.extend(np.tile(local_dofs, len(local_dofs)))
                geometric_values.extend(local_hessian.reshape(-1))
            if geometric_values:
                stiffness += sparse.coo_matrix(
                    (geometric_values, (geometric_rows, geometric_cols)),
                    shape=(self.n_dof, self.n_dof),
                ).tocsr()
        minimum_gap, sampled, checked = self._certify_gap(u, None, self.gap_tolerance)
        shell_force = np.zeros(3)
        shell_moment = np.zeros(3)
        for node_id in self.node_ids:
            dofs = np.asarray(self.model.mesh.dof_manager.get_node_dofs(node_id)[:6],
                              dtype=np.intp)
            physical_force = -force[dofs[:3]]
            position = (np.asarray(self.model.mesh.get_node(node_id).coords(), dtype=float)
                        + u[dofs[:3]])
            shell_force += physical_force
            shell_moment += np.cross(position, physical_force) - force[dofs[3:]]
        results = SphereContactResults(
            self.profile, minimum_gap, energy,
            tuple(float(value) for value in shell_force),
            tuple(float(-value) for value in shell_force),
            tuple(float(value) for value in shell_moment),
            tuple(float(-value) for value in shell_moment),
            int(np.count_nonzero(normalized_gap < 0.0)),
            minimum_gap >= -self.gap_tolerance,
            sampled, float(np.min(normalized_gap)), count,
            checked, len(self.faces),
        )
        return force, stiffness, results

    def admissible_step_fraction(self, start: np.ndarray, end: np.ndarray) -> float:
        first = np.asarray(start, dtype=float).reshape(-1)
        last = np.asarray(end, dtype=float).reshape(-1)
        try:
            full_step_lower = self._certify_gap(first, last, self.gap_tolerance)[0]
        except _GapCertificateLimitError:
            full_step_lower = -float("inf")
        if full_step_lower >= -self.gap_tolerance:
            return 1.0
        lower, upper = 0.0, 1.0
        for _ in range(30):
            middle = 0.5 * (lower + upper)
            candidate = first + middle * (last - first)
            try:
                candidate_lower = self._certify_gap(
                    first, candidate, self.gap_tolerance,
                )[0]
            except _GapCertificateLimitError:
                candidate_lower = -float("inf")
            if candidate_lower >= -self.gap_tolerance:
                lower = middle
            else:
                upper = middle
        self.swept_trial_cutbacks += 1
        return max(0.0, 0.98 * lower)

    def swept_increment_admissible(self, start: np.ndarray, end: np.ndarray,
                                   start_factor: float,
                                   end_factor: float) -> bool:
        if not np.any(self.path_translation):
            return True
        first_center = self.center_at(start_factor)
        last_center = self.center_at(end_factor)
        self._check_side(start, first_center)
        try:
            self._check_side(end, last_center)
        except ValueError:
            self.swept_trial_cutbacks += 1
            return False
        try:
            lower = self._certify_gap(
                start, end, self.gap_tolerance,
                start_factor=start_factor, end_factor=end_factor,
            )[0]
        except _GapCertificateLimitError:
            lower = -float("inf")
        if lower >= -self.gap_tolerance:
            return True
        self.swept_trial_cutbacks += 1
        return False

    def tool_work_increment(self, start: np.ndarray, end: np.ndarray,
                            start_factor: float, end_factor: float) -> float:
        if not np.any(self.path_translation):
            return 0.0
        previous_factor = self.path_factor
        try:
            self.set_path_factor(start_factor)
            first_force = np.asarray(self.evaluate(start, tangent=False)[2].rigid_force)
            self.set_path_factor(end_factor)
            last_force = np.asarray(self.evaluate(end, tangent=False)[2].rigid_force)
        finally:
            self.set_path_factor(previous_factor)
        motion = (float(end_factor) - float(start_factor)) * self.path_translation
        work = float(0.5 * (first_force + last_force) @ motion)
        if not np.isfinite(work):
            raise ValueError("rigid-tool contact work is nonfinite")
        return work
