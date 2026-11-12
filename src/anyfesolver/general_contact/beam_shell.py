"""Interior classical round-beam line contact on one physical shell face.

The first solver admissions are restricted to a planar S3 or Q4 face whose
nodes undergo common translation. Rim/corner contact and a warped current Q4
are separate capabilities. A Bézier hull check proves that the whole B2/B3
centreline projection stays inside the declared face clearance.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import sparse

from .beam_plane import _active_intervals, _gap_coefficients, _minimum_polynomial
from .core import ContactDefinition, ContactSolverSettings, ContactValidationReport
from .overlap import PlanarFace, ProjectedOverlap, reference_shell_face


_GAUSS_8 = np.polynomial.legendre.leggauss(8)
_GAUSS_16 = np.polynomial.legendre.leggauss(16)
PROFILE = "CLASSICAL_B2_B3_ROUND_TUBE_S3_INTERIOR_LINE_PENALTY_STATIC_DEV_V0"
PROFILE_Q4 = "CLASSICAL_B2_B3_ROUND_TUBE_Q4_INTERIOR_LINE_PENALTY_STATIC_DEV_V0"
PROFILE_PATCH = "CLASSICAL_B2_B3_ROUND_TUBE_PLANAR_SHELL_PATCH_LINE_PENALTY_STATIC_DEV_V0"


@dataclass(frozen=True)
class InteriorBeamShellState:
    energy: float
    beam_residual: np.ndarray
    shell_residual: np.ndarray
    minimum_gap: float
    active_reference_length: float
    active_intervals: int
    integration_points: int
    maximum_relative_integration_error: float
    shell_normal: np.ndarray
    minimum_rim_clearance: float


def _beam_shape(xi: float, count: int) -> np.ndarray:
    if count == 2:
        return np.array((0.5 * (1.0 - xi), 0.5 * (1.0 + xi)))
    return np.array((0.5 * xi * (xi - 1.0), 1.0 - xi * xi,
                     0.5 * xi * (xi + 1.0)))


def _bezier_controls(positions: np.ndarray) -> np.ndarray:
    if len(positions) == 2:
        return positions
    return np.array((positions[0],
                     2.0 * positions[1] - 0.5 * (positions[0] + positions[2]),
                     positions[2]))


class InteriorBeamShellLine:
    """One B2/B3 reference line against one convex physical planar shell face."""

    def __init__(self, beam_reference: np.ndarray, shell_reference: PlanarFace,
                 side_normal: np.ndarray, radius: float, line_stiffness: float,
                 rim_clearance: float, integration_relative_tolerance: float,
                 maximum_subdivisions: int):
        reference = np.asarray(beam_reference, dtype=float)
        if (reference.shape not in {(2, 3), (3, 3)}
                or not np.all(np.isfinite(reference))):
            raise ValueError("beam-shell reference needs two or three finite beam nodes")
        length = float(np.linalg.norm(reference[-1] - reference[0]))
        if length <= 0.0:
            raise ValueError("beam-shell reference length must be positive")
        if len(reference) == 3 and float(np.linalg.norm(
            reference[1] - 0.5 * (reference[0] + reference[2])
        )) > 1.0e-10 * length:
            raise ValueError("classical B3 beam-shell reference midpoint must be straight")
        if type(shell_reference) is not PlanarFace:
            raise ValueError("beam-shell target needs a physical planar face")
        normal = np.asarray(side_normal, dtype=float)
        if normal.shape != (3,) or not np.all(np.isfinite(normal)):
            raise ValueError("beam-shell side normal must be a finite 3-vector")
        magnitude = float(np.linalg.norm(normal))
        if magnitude <= 0.0:
            raise ValueError("beam-shell side normal must be nonzero")
        for name, value in (("radius", radius), ("line_stiffness", line_stiffness),
                            ("rim_clearance", rim_clearance),
                            ("integration_relative_tolerance", integration_relative_tolerance)):
            if not np.isfinite(value) or value <= 0.0:
                raise ValueError(f"beam-shell {name} must be finite and positive")
        if rim_clearance < radius:
            raise ValueError("beam-shell rim_clearance must cover the declared radius")
        if type(maximum_subdivisions) is not int or maximum_subdivisions < 1:
            raise ValueError("beam-shell maximum_subdivisions must be positive integer")
        self.reference = reference.copy()
        self.reference_length = length
        self.shell_reference = shell_reference
        self.side_normal = normal / magnitude
        self.radius = float(radius)
        self.stiffness = float(line_stiffness)
        self.rim_clearance = float(rim_clearance)
        self.integration_relative_tolerance = float(integration_relative_tolerance)
        self.maximum_subdivisions = maximum_subdivisions
        reference_geometric_normal = np.cross(
            shell_reference.corners[1] - shell_reference.corners[0],
            shell_reference.corners[-1] - shell_reference.corners[0],
        )
        reference_geometric_normal /= np.linalg.norm(reference_geometric_normal)
        alignment = float(reference_geometric_normal @ self.side_normal)
        if abs(alignment) < 1.0 - 1.0e-10:
            raise ValueError("beam-shell side normal must match a physical face orientation")
        self.orientation = 1.0 if alignment > 0.0 else -1.0

    def _frame(self, shell: PlanarFace):
        corners = shell.corners
        geometric = np.cross(corners[1] - corners[0], corners[-1] - corners[0])
        geometric /= np.linalg.norm(geometric)
        reference_geometric = self.orientation * self.side_normal
        if float(geometric @ reference_geometric) <= 0.0:
            raise ValueError("beam-shell face inverted or crossed its physical side")
        tangent = corners[1] - corners[0]
        tangent /= np.linalg.norm(tangent)
        binormal = np.cross(geometric, tangent)
        xy = np.column_stack(((corners - corners[0]) @ tangent,
                              (corners - corners[0]) @ binormal))
        area = 0.5 * abs(float(sum(
            xy[i, 0] * xy[(i + 1) % len(xy), 1]
            - xy[i, 1] * xy[(i + 1) % len(xy), 0]
            for i in range(len(xy))
        )))
        projector = ProjectedOverlap(
            shell.feature_id, shell.feature_id, corners, area,
            corners[0], tangent, binormal, geometric,
        )
        return self.orientation * geometric, tangent, binormal, xy, projector

    def _inside_clearance(self, controls: np.ndarray, origin: np.ndarray,
                          normal: np.ndarray, tangent: np.ndarray,
                          binormal: np.ndarray, polygon_xy: np.ndarray) -> float:
        minimum = float("inf")
        for position in controls:
            projected = position - normal * float((position - origin) @ normal)
            point = np.array(((projected - origin) @ tangent,
                              (projected - origin) @ binormal))
            for index, corner in enumerate(polygon_xy):
                edge = polygon_xy[(index + 1) % len(polygon_xy)] - corner
                inward_distance = float(np.linalg.det(np.stack((edge, point - corner)))
                                        / np.linalg.norm(edge))
                minimum = min(minimum, inward_distance)
        if minimum < self.rim_clearance:
            raise ValueError("beam-shell curve approaches an unadmitted physical rim")
        return minimum

    def evaluate(self, beam_positions: np.ndarray,
                 shell_positions: np.ndarray, *,
                 integration_window: tuple[float, float] | None = None,
                 check_rim: bool = True) -> InteriorBeamShellState:
        beam = np.asarray(beam_positions, dtype=float)
        shell_points = np.asarray(shell_positions, dtype=float)
        if (beam.shape != self.reference.shape
                or shell_points.shape != self.shell_reference.corners.shape
                or not np.all(np.isfinite(beam))
                or not np.all(np.isfinite(shell_points))):
            raise ValueError("beam-shell trial positions have invalid size or nonfinite values")
        shell = PlanarFace(self.shell_reference.feature_id, shell_points)
        normal, tangent, binormal, polygon_xy, projector = self._frame(shell)
        minimum_rim = (self._inside_clearance(
            _bezier_controls(beam), shell.corners[0], normal,
            tangent, binormal, polygon_xy,
        ) if check_rim else float("inf"))
        coefficients = _gap_coefficients(
            (beam - shell.corners[0]) @ normal - self.radius
        )
        minimum_gap = _minimum_polynomial(coefficients)
        intervals = _active_intervals(coefficients)
        if integration_window is not None:
            window_left, window_right = integration_window
            if not (-1.0 <= window_left < window_right <= 1.0):
                raise ValueError("beam-shell integration window must lie within the beam cell")
            intervals = tuple(
                (max(left, window_left), min(right, window_right))
                for left, right in intervals
                if min(right, window_right) > max(left, window_left)
            )
        beam_force = np.zeros_like(beam)
        shell_force = np.zeros_like(shell_points)
        energy = 0.0
        points = 0
        maximum_error = 0.0

        def quadrature(left: float, right: float, rule):
            nodes, weights = rule
            middle, half = 0.5 * (left + right), 0.5 * (right - left)
            local_energy = 0.0
            local_beam = np.zeros_like(beam)
            local_shell = np.zeros_like(shell_points)
            for abscissa, weight in zip(nodes, weights):
                xi = middle + half * float(abscissa)
                beam_shape = _beam_shape(xi, len(beam))
                position = beam_shape @ beam
                projected = position - normal * float((position - shell.corners[0]) @ normal)
                shell_shape = projector.shape_values(shell, projected)
                gap = (coefficients[0] * xi + coefficients[1]) * xi + coefficients[2]
                measure = half * 0.5 * self.reference_length * float(weight)
                if gap < 0.0:
                    traction = self.stiffness * gap * measure * normal
                    local_energy += 0.5 * self.stiffness * gap * gap * measure
                    local_beam += beam_shape[:, None] * traction
                    local_shell -= shell_shape[:, None] * traction
            return local_energy, local_beam, local_shell

        def refine(left: float, right: float, depth: int):
            nonlocal points, maximum_error
            low = quadrature(left, right, _GAUSS_8)
            high = quadrature(left, right, _GAUSS_16)
            points += 24
            errors = (
                abs(high[0] - low[0]) / max(abs(high[0]), np.finfo(float).tiny),
                float(np.linalg.norm(high[1] - low[1]))
                / max(float(np.linalg.norm(high[1])), np.finfo(float).tiny),
                float(np.linalg.norm(high[2] - low[2]))
                / max(float(np.linalg.norm(high[2])), np.finfo(float).tiny),
            )
            error = max(errors)
            if error <= self.integration_relative_tolerance:
                maximum_error = max(maximum_error, error)
                return high
            if depth >= self.maximum_subdivisions:
                raise RuntimeError("beam-shell line integration subdivision limit exhausted")
            middle = 0.5 * (left + right)
            first = refine(left, middle, depth + 1)
            second = refine(middle, right, depth + 1)
            return tuple(a + b for a, b in zip(first, second))

        for left, right in intervals:
            local_energy, local_beam, local_shell = refine(left, right, 0)
            energy += local_energy
            beam_force += local_beam
            shell_force += local_shell
        active_length = (0.5 * self.reference_length
                         * sum(right - left for left, right in intervals))
        return InteriorBeamShellState(
            energy, beam_force, shell_force, minimum_gap,
            active_length, len(intervals), points,
            maximum_error, normal, minimum_rim,
        )


def validate_beam_shell_contact(model, definition: ContactDefinition,
                                settings: ContactSolverSettings | None = None,
                                ) -> ContactValidationReport:
    from anyfesolver.e4_pl_s3_v2d_element import FORMULATION_ID as s3_id
    issues: list[str] = []
    controls = ContactSolverSettings() if settings is None else settings
    if (len(definition.surfaces) != 2 or definition.rigid_bodies
            or len(definition.interactions) != 1):
        return ContactValidationReport("invalid", (
            "interior beam-shell profile requires one beam, one shell and one interaction",
        ), PROFILE)
    beams = [surface for surface in definition.surfaces
             if surface.beam_radius is not None]
    shells = [surface for surface in definition.surfaces
              if surface.beam_radius is None]
    if len(beams) != 1 or len(shells) != 1:
        return ContactValidationReport("invalid", (
            "interior beam-shell profile requires one declared round beam surface",
        ), PROFILE)
    beam, shell = beams[0], shells[0]
    interaction = definition.interactions[0]
    if len(shell.element_ids) > 1 and controls.beam_shell_geometry == "translated_planar":
        from .beam_shell_patch import validate_beam_shell_patch_contact

        return validate_beam_shell_patch_contact(model, definition, controls)
    if len(beam.element_ids) != 1 or len(shell.element_ids) != 1:
        issues.append("interior beam-shell profile admits one cell and one face")
    if controls.pair_motion != "general":
        issues.append("beam-shell contact requires pair_motion='general'")
    if (controls.beam_shell_rim_clearance is None
            or controls.beam_shell_rim_clearance < beam.beam_radius):
        issues.append("beam-shell requires explicit rim clearance at least its radius")
    if controls.beam_shell_difference_step is None and controls.beam_shell_geometry == "translated_planar":
        issues.append("beam-shell requires an explicit dimensional difference step")
    if controls.line_integration_relative_tolerance is None:
        issues.append("beam-shell requires explicit line integration tolerance")
    if shell.side not in {"top", "bottom"}:
        issues.append("beam-shell requires a physical top or bottom shell side")
    if beam.beam_end_caps != "none":
        issues.append("interior beam-shell profile does not admit endpoint caps")
    if (interaction.surface != beam.name or interaction.target != shell.name
            or interaction.normal_law != "penalty"
            or interaction.normal_stiffness is not None
            or interaction.line_stiffness is None
            or interaction.endpoint_stiffness is not None):
        issues.append("beam-shell interaction requires directed line penalty and no other stiffness")
    profile = PROFILE
    if len(beam.element_ids) == 1 and len(shell.element_ids) == 1:
        from anyfesolver.elements import BeamElement, QuadraticBeamElement
        from anyfesolver.e4_pl_element import FORMULATION_ID as q4_id
        from anyfesolver.e4_pl_s3_v2d_element import FORMULATION_ID as s3_id
        from anymaterial import IsotropicMaterial

        beam_element = model.mesh.elements.get(beam.element_ids[0])
        shell_element = model.mesh.elements.get(shell.element_ids[0])
        if type(beam_element) not in {BeamElement, QuadraticBeamElement}:
            issues.append("beam-shell source is not an admitted classical B2/B3")
        else:
            if (beam_element.generalized_section is not None
                    or beam_element._fiber_plasticity is not None
                    or getattr(beam_element, "_geometric_nonlinearity", None) != "von_karman"
                    or np.any(getattr(beam_element, "eccentricity", np.zeros(3)))):
                issues.append("beam-shell source needs a concentric scalar von Karman section")
            material = model.materials.get(beam_element.material_name)
            if (type(material) is not IsotropicMaterial
                    or material.yield_stress != 0.0
                    or material.hardening_curve is not None):
                issues.append("beam-shell source needs an isotropic elastic material")
        shell_formulation = getattr(shell_element, "formulation_id", None)
        if shell_formulation == q4_id:
            profile = PROFILE_Q4
        if shell_formulation not in {q4_id, s3_id}:
            issues.append("interior beam-shell profile admits qualified planar Q4 or S3 V2D")
        else:
            material = model.materials.get(shell_element.material_name)
            if (type(material) is not IsotropicMaterial
                    or material.yield_stress != 0.0
                    or material.hardening_curve is not None):
                issues.append("beam-shell target needs an isotropic elastic material")
            try:
                reference_shell_face(model, 2, shell.element_ids[0], shell.side)
            except (KeyError, ValueError) as exc:
                issues.append(f"beam-shell target: {exc}")
        if beam_element is not None and shell_element is not None and (
            set(beam_element.node_ids) & set(shell_element.node_ids)
        ):
            issues.append("beam-shell participants must not share material nodes")
    if controls.beam_shell_geometry == "jet2_deformable_s3_dev":
        from .deformable_beam_s3_static import PROFILE as deformable_profile
        profile = deformable_profile
        if len(shell.element_ids) != 1 or getattr(model.mesh.elements.get(shell.element_ids[0]), "formulation_id", None) != s3_id:
            issues.append("differentiated deformable target requires exactly one S3 V2D")
        if controls.beam_shell_maximum_swept_intervals is None:
            issues.append("deformable target requires an explicit swept interval budget")
    return ContactValidationReport("invalid" if issues else "ok", tuple(issues), profile)


@dataclass(frozen=True)
class _BeamShellResults:
    profile: str
    minimum_gap: float
    line_contact_energy: float
    active_line_length: float
    active_intervals: int
    integration_points: int
    integration_relative_error: float
    minimum_rim_clearance: float
    beam_force: tuple[float, float, float]
    shell_force: tuple[float, float, float]
    beam_moment: tuple[float, float, float]
    shell_moment: tuple[float, float, float]
    gap_tolerance_met: bool
    line_stiffness: float

    def to_dict(self) -> dict[str, object]:
        return {
            "profile": self.profile,
            "minimum_gap": self.minimum_gap,
            "line_contact_energy": self.line_contact_energy,
            "algorithmic_contact_potential": self.line_contact_energy,
            "active_line_length": self.active_line_length,
            "active_intervals": self.active_intervals,
            "integration_points": self.integration_points,
            "integration_relative_error": self.integration_relative_error,
            "minimum_rim_clearance": self.minimum_rim_clearance,
            "beam_force": list(self.beam_force),
            "shell_force": list(self.shell_force),
            "beam_moment": list(self.beam_moment),
            "shell_moment": list(self.shell_moment),
            "gap_tolerance_met": self.gap_tolerance_met,
            "normal_law": "penalty",
            "line_stiffness_N_per_m2": self.line_stiffness,
        }


class _BeamShellManager:
    def __init__(self, model, definition: ContactDefinition,
                 settings: ContactSolverSettings | None = None,
                 initial_displacement: np.ndarray | None = None):
        self.settings = ContactSolverSettings() if settings is None else settings
        report = validate_beam_shell_contact(model, definition, self.settings)
        if not report.valid:
            raise ValueError("unsupported interior beam-shell contact: " + "; ".join(report.issues))
        self.model = model
        self.definition = definition
        self.profile = report.profile
        beam = next(item for item in definition.surfaces if item.beam_radius is not None)
        shell = next(item for item in definition.surfaces if item.beam_radius is None)
        interaction = definition.interactions[0]
        self.beam_surface = beam
        self.shell_surface = shell
        self.gap_tolerance = interaction.gap_tolerance
        self.stiffness = interaction.line_stiffness
        self.difference_step = self.settings.beam_shell_difference_step
        self.swept_trial_cutbacks = 0
        beam_element = model.mesh.elements[beam.element_ids[0]]
        shell_element = model.mesh.elements[shell.element_ids[0]]
        self.beam_ids = tuple(int(item) for item in beam_element.node_ids)
        self.shell_ids = tuple(int(item) for item in shell_element.node_ids)
        self.beam_reference = np.asarray(
            beam_element.get_node_coordinates(model.mesh), dtype=float,
        )
        self.shell_reference = reference_shell_face(
            model, 2, shell.element_ids[0], shell.side,
        )
        self.beam_dofs = np.asarray(
            [model.mesh.dof_manager.get_node_dofs(node)[:6] for node in self.beam_ids],
            dtype=np.intp,
        )
        self.shell_dofs = np.asarray(
            [model.mesh.dof_manager.get_node_dofs(node)[:6] for node in self.shell_ids],
            dtype=np.intp,
        )
        self.shell_node_reference = np.asarray(
            [model.mesh.get_node(node).coords() for node in self.shell_ids], dtype=float,
        )
        director = shell_element.physical_reference_director
        if director is None:
            director = np.cross(
                self.shell_node_reference[1] - self.shell_node_reference[0],
                self.shell_node_reference[-1] - self.shell_node_reference[0],
            )
            director /= np.linalg.norm(director)
        sign = 1.0 if shell.side == "top" else -1.0
        side_normal = sign * np.asarray(director, dtype=float)
        coordinate = sign
        from anyfesolver.e4_pl_s3_v2d_element import FORMULATION_ID as s3_id

        offset = (
            float(shell_element.material_surface_offset_from_reference(coordinate))
            if shell_element.formulation_id == s3_id else
            0.5 * coordinate * float(shell_element.thickness)
        )
        self.shell_lever = offset * np.asarray(director, dtype=float)
        self.kernel = InteriorBeamShellLine(
            self.beam_reference, self.shell_reference, side_normal,
            beam.beam_radius, self.stiffness,
            self.settings.beam_shell_rim_clearance,
            self.settings.line_integration_relative_tolerance,
            self.settings.maximum_line_subdivisions,
        )
        self.motion_tolerance = (256.0 * np.finfo(float).eps
                                 * max(self.kernel.reference_length,
                                       self.kernel.radius))
        # Newton trial clipping can leave a start pose a few ulps outside the
        # requested gap. The final equilibrium check still uses the exact
        # user tolerance; this guard prevents a spurious trial exception.
        self.gap_roundoff = max(
            1.0e-10 * self.kernel.reference_length,
            16.0 * self.motion_tolerance,
        )
        total = model.mesh.dof_manager.total_dofs
        initial = (np.zeros(total) if initial_displacement is None
                   else np.asarray(initial_displacement, dtype=float).reshape(-1))
        if initial.size != total or not np.all(np.isfinite(initial)):
            raise ValueError("beam-shell initial displacement is invalid")
        initial_state = self._assemble(initial, enforce_motion=True)[1]
        if initial_state.minimum_gap < -self.motion_tolerance:
            raise ValueError("beam-shell contact has unresolved initial penetration")

    def _positions(self, u: np.ndarray, *, enforce_motion: bool):
        if u.size != self.model.mesh.dof_manager.total_dofs or not np.all(np.isfinite(u)):
            raise ValueError("beam-shell trial displacement is invalid")
        shell_translations = u[self.shell_dofs[:, :3]]
        shell_rotations = u[self.shell_dofs[:, 3:]]
        if enforce_motion and (
            float(np.max(np.linalg.norm(shell_translations - shell_translations[0], axis=1)))
            > self.motion_tolerance
            or float(np.max(np.linalg.norm(shell_rotations, axis=1)))
            > self.motion_tolerance
        ):
            raise ValueError("interior beam-shell profile requires common shell translation and fixed rotations")
        beam_positions = self.beam_reference + u[self.beam_dofs[:, :3]]
        shell_positions = (self.shell_reference.corners + shell_translations
                           + np.cross(shell_rotations, self.shell_lever))
        return beam_positions, shell_positions

    def _assemble(self, u: np.ndarray, *, enforce_motion: bool):
        beam_positions, shell_positions = self._positions(u, enforce_motion=enforce_motion)
        state = self.kernel.evaluate(beam_positions, shell_positions)
        force = np.zeros(self.model.mesh.dof_manager.total_dofs)
        for dofs, local in zip(self.beam_dofs, state.beam_residual):
            force[dofs[:3]] += local
        for dofs, local in zip(self.shell_dofs, state.shell_residual):
            force[dofs[:3]] += local
            force[dofs[3:]] += np.cross(self.shell_lever, local)
        return force, state

    def evaluate(self, displacement: np.ndarray, *, tangent: bool = True):
        u = np.asarray(displacement, dtype=float).reshape(-1)
        force, state = self._assemble(u, enforce_motion=True)
        tangent_matrix = None
        if tangent:
            beam_columns = sorted(set(map(int, self.beam_dofs[:, :3].ravel())))
            if len(self.shell_ids) == 4 or getattr(self, "shell_common_translation_only", False):
                # A single Q4-node perturbation would warp the face. The
                # admitted Q4 motion has only three collective translation
                # directions; distribute each collective derivative equally
                # so the matrix acts exactly on that tangent space.
                column_groups = [(column,) for column in beam_columns]
                column_groups += [
                    tuple(map(int, self.shell_dofs[:, axis]))
                    for axis in range(3)
                ]
            else:
                touched = sorted(set(map(int, np.r_[self.beam_dofs[:, :3].ravel(),
                                                     self.shell_dofs.ravel()])))
                column_groups = [(column,) for column in touched]
            rows = []
            columns = []
            values = []
            for group in column_groups:
                perturbation = np.zeros_like(u)
                perturbation[list(group)] = self.difference_step
                plus_force, plus = self._assemble(u + perturbation, enforce_motion=False)
                minus_force, minus = self._assemble(u - perturbation, enforce_motion=False)
                if (plus.active_intervals == state.active_intervals
                        and minus.active_intervals == state.active_intervals):
                    derivative = (plus_force - minus_force) / (2.0 * self.difference_step)
                elif plus.active_intervals == state.active_intervals:
                    derivative = (plus_force - force) / self.difference_step
                elif minus.active_intervals == state.active_intervals:
                    derivative = (force - minus_force) / self.difference_step
                else:
                    raise ValueError("beam-shell active interval transition needs path subdivision")
                nonzero = np.flatnonzero(derivative)
                for column in group:
                    rows.extend(int(row) for row in nonzero)
                    columns.extend([column] * len(nonzero))
                    values.extend(float(derivative[row]) / len(group)
                                  for row in nonzero)
            tangent_matrix = sparse.coo_matrix(
                (values, (rows, columns)), shape=(len(u), len(u)),
            ).tocsr()
        beam_force = np.zeros(3)
        shell_force = np.zeros(3)
        beam_moment = np.zeros(3)
        shell_moment = np.zeros(3)
        for reference, dofs in zip(self.beam_reference, self.beam_dofs):
            physical = -force[dofs[:3]]
            beam_force += physical
            beam_moment += np.cross(reference + u[dofs[:3]], physical)
        for reference, dofs in zip(self.shell_node_reference, self.shell_dofs):
            physical = -force[dofs[:3]]
            shell_force += physical
            shell_moment += (np.cross(reference + u[dofs[:3]], physical)
                             - force[dofs[3:]])
        result = _BeamShellResults(
            self.profile, state.minimum_gap, state.energy,
            state.active_reference_length, state.active_intervals,
            state.integration_points, state.maximum_relative_integration_error,
            state.minimum_rim_clearance,
            tuple(map(float, beam_force)), tuple(map(float, shell_force)),
            tuple(map(float, beam_moment)), tuple(map(float, shell_moment)),
            state.minimum_gap >= -self.gap_tolerance, self.stiffness,
        )
        return force, tangent_matrix, result

    def admissible_step_fraction(self, start: np.ndarray,
                                 end: np.ndarray) -> float:
        first = np.asarray(start, dtype=float).reshape(-1)
        last = np.asarray(end, dtype=float).reshape(-1)
        first_state = self._assemble(first, enforce_motion=True)[1]
        numerical_limit = self.gap_tolerance + self.gap_roundoff
        try:
            last_state = self._assemble(last, enforce_motion=True)[1]
        except ValueError as exc:
            if "rim" not in str(exc):
                raise
            lower, upper = 0.0, 1.0
            for _ in range(30):
                middle = 0.5 * (lower + upper)
                try:
                    self._assemble(first + middle * (last - first), enforce_motion=True)
                    lower = middle
                except ValueError as trial_exc:
                    if "rim" not in str(trial_exc):
                        raise
                    upper = middle
            self.swept_trial_cutbacks += 1
            return max(0.0, 0.98 * lower)
        if first_state.minimum_gap < -numerical_limit:
            # The caller can supply a disposable Newton iterate at the clipped
            # limit. Permit a correction toward opening; final increment
            # acceptance checks the exact declared gap tolerance separately.
            self.swept_trial_cutbacks += 1
            return 1.0 if last_state.minimum_gap >= first_state.minimum_gap else 0.0
        if last_state.minimum_gap >= -numerical_limit:
            return 1.0
        self.swept_trial_cutbacks += 1
        denominator = first_state.minimum_gap - last_state.minimum_gap
        fraction = ((first_state.minimum_gap + numerical_limit) / denominator
                    if denominator > 0.0 else 0.0)
        return max(0.0, min(1.0, 0.98 * fraction))

    def swept_increment_admissible(self, start: np.ndarray, end: np.ndarray,
                                   _start_factor: float, _end_factor: float) -> bool:
        first_state = self._assemble(np.asarray(start, dtype=float).reshape(-1),
                                     enforce_motion=True)[1]
        numerical_limit = self.gap_tolerance + self.gap_roundoff
        if first_state.minimum_gap < -numerical_limit:
            raise ValueError(
                "beam-shell increment start exceeds the gap tolerance: "
                f"gap={first_state.minimum_gap:.12g} m, "
                f"tolerance={self.gap_tolerance:.12g} m"
            )
        try:
            last_state = self._assemble(np.asarray(end, dtype=float).reshape(-1),
                                        enforce_motion=True)[1]
        except ValueError as exc:
            if "rim" not in str(exc):
                raise
            self.swept_trial_cutbacks += 1
            return False
        if last_state.minimum_gap < -numerical_limit:
            self.swept_trial_cutbacks += 1
            return False
        # A common shell translation fixes the plane normal. Every material
        # gap and each Bézier control's inward rim distance is affine along
        # the linear trial path, so both endpoint certificates cover it.
        return True

    def restart_contract(self) -> dict[str, object]:
        interaction = self.definition.interactions[0]
        return {
            "profile": self.profile,
            "beam": {"name": self.beam_surface.name,
                     "element_ids": list(self.beam_surface.element_ids),
                     "radius": self.beam_surface.beam_radius,
                     "end_caps": self.beam_surface.beam_end_caps},
            "shell": {"name": self.shell_surface.name,
                      "element_ids": list(self.shell_surface.element_ids),
                      "side": self.shell_surface.side},
            "interaction": {"name": interaction.name,
                            "line_stiffness": interaction.line_stiffness,
                            "gap_tolerance": interaction.gap_tolerance,
                            "normal_law": interaction.normal_law},
            "settings": {"pair_motion": self.settings.pair_motion,
                         "rim_clearance": self.settings.beam_shell_rim_clearance,
                         "difference_step": self.difference_step,
                         "line_integration_relative_tolerance": self.settings.line_integration_relative_tolerance,
                         "maximum_line_subdivisions": self.settings.maximum_line_subdivisions},
        }
