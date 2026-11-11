"""Fixed sphere/cylinder line contact for declared round classical B2/B3 beams.

The exterior gap is the radial distance of the deformed centreline minus the
sum of obstacle and beam radii. The line penalty is integrated on reference
arclength. This is a bounded static development profile without end caps.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import sparse

from .beam_plane import _BeamCell, _beam_types
from .core import ContactDefinition, ContactSolverSettings, ContactValidationReport


PROFILE = "CLASSICAL_B2_B3_ROUND_TUBE_FIXED_RADIAL_LINE_PENALTY_STATIC_DEV_V0"
PROFILE_MOVING = "CLASSICAL_B2_B3_ROUND_TUBE_PRESCRIBED_RADIAL_LINE_PENALTY_STATIC_DEV_V0"
_GAUSS_8 = np.polynomial.legendre.leggauss(8)
_GAUSS_16 = np.polynomial.legendre.leggauss(16)


@dataclass(frozen=True)
class _RadialResults:
    profile: str
    obstacle_kind: str
    minimum_gap: float
    line_contact_energy: float
    active_line_length: float
    active_line_cells: int
    active_intervals: int
    integration_points: int
    maximum_integration_relative_error: float
    candidate_cells: int
    beam_force: tuple[float, float, float]
    rigid_force: tuple[float, float, float]
    beam_moment: tuple[float, float, float]
    rigid_moment: tuple[float, float, float]
    gap_tolerance_met: bool
    line_stiffness: float

    def to_dict(self) -> dict[str, object]:
        return {
            "profile": self.profile,
            "obstacle_kind": self.obstacle_kind,
            "minimum_gap": self.minimum_gap,
            "line_contact_energy": self.line_contact_energy,
            "algorithmic_contact_potential": self.line_contact_energy,
            "active_line_length": self.active_line_length,
            "active_line_cells": self.active_line_cells,
            "active_intervals": self.active_intervals,
            "integration_points": self.integration_points,
            "maximum_integration_relative_error": self.maximum_integration_relative_error,
            "candidate_cells": self.candidate_cells,
            "beam_force": list(self.beam_force),
            "rigid_force": list(self.rigid_force),
            "beam_moment": list(self.beam_moment),
            "rigid_moment": list(self.rigid_moment),
            "gap_tolerance_met": self.gap_tolerance_met,
            "normal_law": "penalty",
            "line_stiffness_N_per_m2": self.line_stiffness,
        }


def validate_beam_radial_contact(model, definition: ContactDefinition,
                                 settings: ContactSolverSettings | None = None
                                 ) -> ContactValidationReport:
    issues: list[str] = []
    if tuple(map(len, (definition.surfaces, definition.rigid_bodies,
                       definition.interactions))) != (1, 1, 1):
        return ContactValidationReport("invalid", (
            "radial beam profile requires one beam surface, one rigid body and one interaction",
        ), PROFILE)
    surface, body, interaction = (definition.surfaces[0], definition.rigid_bodies[0],
                                  definition.interactions[0])
    profile = PROFILE_MOVING if body.path_translation is not None else PROFILE
    controls = ContactSolverSettings() if settings is None else settings
    if controls.pair_motion != "general":
        issues.append("radial beam contact requires pair_motion='general'")
    if controls.line_integration_relative_tolerance is None:
        issues.append("radial beam contact requires explicit line_integration_relative_tolerance")
    if surface.beam_radius is None or surface.side != "midsurface" or surface.beam_end_caps != "none":
        issues.append("radial beam contact requires explicit round radius and no endpoint caps")
    if body.kind not in {"sphere", "cylinder"}:
        issues.append("radial beam profile requires a sphere or infinite cylinder")
    if (interaction.surface, interaction.target) != (surface.name, body.name):
        issues.append("radial beam interaction must name its beam and obstacle")
    if (interaction.normal_law != "penalty" or interaction.normal_stiffness is not None
            or interaction.line_stiffness is None or interaction.endpoint_stiffness is not None):
        issues.append("radial beam contact requires penalty line_stiffness only")
    if len(surface.element_ids) > controls.maximum_candidate_pairs:
        issues.append("radial beam candidate limit exceeded; no cell is dropped")
    b2, b3 = _beam_types()
    from anymaterial import IsotropicMaterial

    for element_id in surface.element_ids:
        element = model.mesh.elements.get(element_id)
        if type(element) not in {b2, b3}:
            issues.append(f"element {element_id} is not an admitted classical B2/B3 beam")
            continue
        if (element.generalized_section is not None
                or element._fiber_plasticity is not None
                or getattr(element, "_geometric_nonlinearity", None) != "von_karman"
                or np.any(getattr(element, "eccentricity", np.zeros(3)))):
            issues.append(f"element {element_id} needs a concentric elastic scalar von Karman section")
        material = model.materials.get(element.material_name)
        if (type(material) is not IsotropicMaterial or material.yield_stress != 0.0
                or material.hardening_curve is not None):
            issues.append(f"element {element_id} needs an isotropic elastic material")
        try:
            reference = np.asarray(element.get_node_coordinates(model.mesh), dtype=float)
            if not np.all(np.isfinite(reference)) or float(np.linalg.norm(
                reference[-1] - reference[0]
            )) <= 0.0:
                raise ValueError("zero or invalid reference length")
        except (ValueError, KeyError) as exc:
            issues.append(f"element {element_id}: {exc}")
    if (surface.beam_radius is not None and body.radius is not None
            and interaction.gap_tolerance >= surface.beam_radius + body.radius):
        issues.append("radial beam gap tolerance must be smaller than combined radius")
    return ContactValidationReport("invalid" if issues else "ok", tuple(issues), profile)


def _curve_coefficients(positions: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if len(positions) == 2:
        return positions[0], positions[1] - positions[0], np.zeros(3)
    first, middle, last = positions
    return first, -3.0 * first + 4.0 * middle - last, 2.0 * first - 4.0 * middle + 2.0 * last


def _real_interval_roots(coefficients: np.ndarray) -> tuple[float, ...]:
    scale = float(np.max(np.abs(coefficients)))
    if scale == 0.0:
        return ()
    polynomial = np.polynomial.Polynomial(
        np.asarray(coefficients, dtype=float) / scale,
    ).trim(tol=64.0 * np.finfo(float).eps)
    if polynomial.degree() < 1:
        return ()
    return tuple(sorted(set(float(root.real) for root in polynomial.roots()
                            if abs(root.imag) <= 1.0e-10 and 0.0 < root.real < 1.0)))


def _distance_polynomial(positions: np.ndarray, point: np.ndarray,
                         projector: np.ndarray, radius: float) -> np.ndarray:
    coefficients = list(_curve_coefficients(positions))
    coefficients[0] = coefficients[0] - point
    squared = np.zeros(5)
    for i in range(3):
        for j in range(3):
            squared[i + j] += float(coefficients[i] @ projector @ coefficients[j])
    squared[0] -= radius * radius
    return squared


def _minimum_gap(positions: np.ndarray, point: np.ndarray,
                 projector: np.ndarray, radius: float) -> float:
    polynomial = _distance_polynomial(positions, point, projector, radius)
    stationary = _real_interval_roots(np.polynomial.polynomial.polyder(polynomial))
    a, b, c = _curve_coefficients(positions)
    minimum_squared = min(float(relative @ projector @ relative)
                          for s in (0.0, 1.0, *stationary)
                          for relative in (a + s * b + s * s * c - point,))
    return float(np.sqrt(max(0.0, minimum_squared))) - radius


def _active_intervals(positions: np.ndarray, point: np.ndarray,
                      projector: np.ndarray, radius: float) -> tuple[tuple[float, float], ...]:
    polynomial = _distance_polynomial(positions, point, projector, radius)
    cuts = (0.0, *_real_interval_roots(polynomial), 1.0)
    return tuple((left, right) for left, right in zip(cuts[:-1], cuts[1:])
                 if float(np.polynomial.polynomial.polyval(
                     0.5 * (left + right), polynomial,
                 )) < 0.0)


class _BeamRadialManager:
    def __init__(self, model, definition: ContactDefinition,
                 settings: ContactSolverSettings | None = None,
                 initial_displacement: np.ndarray | None = None):
        self.settings = ContactSolverSettings() if settings is None else settings
        report = validate_beam_radial_contact(model, definition, self.settings)
        if not report.valid:
            raise ValueError("unsupported radial beam contact: " + "; ".join(report.issues))
        self.model = model
        self.definition = definition
        self.profile = report.profile
        self.kind = definition.rigid_bodies[0].kind
        self.reference_point = np.asarray(definition.rigid_bodies[0].point, dtype=float)
        self.point = self.reference_point.copy()
        translation = definition.rigid_bodies[0].path_translation
        self.path_translation = (np.zeros(3) if translation is None
                                 else np.asarray(translation, dtype=float))
        self.path_factor = 0.0
        axis = definition.rigid_bodies[0].normal
        self.projector = (np.eye(3) if self.kind == "sphere" else
                          np.eye(3) - np.outer(axis, axis))
        self.radius = (definition.surfaces[0].beam_radius
                       + definition.rigid_bodies[0].radius)
        self.stiffness = definition.interactions[0].line_stiffness
        self.gap_tolerance = definition.interactions[0].gap_tolerance
        self.end_caps = "none"
        self.swept_trial_cutbacks = 0
        self.cells: list[_BeamCell] = []
        for element_id in definition.surfaces[0].element_ids:
            element = model.mesh.elements[element_id]
            reference = np.asarray(element.get_node_coordinates(model.mesh), dtype=float)
            dofs = np.asarray([model.mesh.dof_manager.get_node_dofs(node)[:3]
                               for node in element.node_ids], dtype=np.intp)
            self.cells.append(_BeamCell(
                element_id, tuple(int(node) for node in element.node_ids),
                reference, dofs, float(np.linalg.norm(reference[-1] - reference[0])),
            ))
        n_dof = model.mesh.dof_manager.total_dofs
        initial = np.zeros(n_dof) if initial_displacement is None else np.asarray(
            initial_displacement, dtype=float,
        ).reshape(-1)
        if initial.size != n_dof or not np.all(np.isfinite(initial)):
            raise ValueError("radial beam initial displacement is invalid")
        if self._minimum_gap(initial) < -256.0 * np.finfo(float).eps * max(
            self.radius, *(cell.length for cell in self.cells)
        ):
            raise ValueError("radial beam contact has unresolved initial penetration")

    @staticmethod
    def _check_positions(positions: np.ndarray) -> None:
        _, b, quadratic = _curve_coefficients(positions)
        if len(positions) == 2:
            minimum_speed = float(np.linalg.norm(b))
        else:
            candidates = [0.0, 1.0]
            denominator = float(quadratic @ quadratic)
            if denominator > 0.0:
                candidates.append(float(np.clip(-float(b @ quadratic) / (2.0 * denominator),
                                                0.0, 1.0)))
            minimum_speed = min(float(np.linalg.norm(b + 2.0 * s * quadratic))
                                for s in candidates)
        if not np.all(np.isfinite(positions)) or minimum_speed <= 1.0e-15:
            raise ValueError("deformed radial beam has zero or invalid centreline speed")

    def _positions(self, cell: _BeamCell, u: np.ndarray) -> np.ndarray:
        positions = cell.reference + u[cell.dofs]
        self._check_positions(positions)
        return positions

    def _minimum_gap(self, u: np.ndarray) -> float:
        return min(_minimum_gap(self._positions(cell, u), self.point,
                                self.projector, self.radius) for cell in self.cells)

    def _integrate_cell(self, cell: _BeamCell, positions: np.ndarray
                        ) -> tuple[float, np.ndarray, np.ndarray, float, int, int, float]:
        intervals = _active_intervals(positions, self.point, self.projector, self.radius)
        count = len(cell.node_ids)
        energy = 0.0
        force = np.zeros(3 * count)
        tangent = np.zeros((3 * count, 3 * count))
        active_length = sum(right - left for left, right in intervals) * cell.length
        integration_points = 0
        maximum_error = 0.0

        def quadrature(left: float, right: float, rule):
            nodes, weights = rule
            midpoint, half = 0.5 * (left + right), 0.5 * (right - left)
            local_energy = 0.0
            local_force = np.zeros(3 * count)
            local_tangent = np.zeros((3 * count, 3 * count))
            for node, weight in zip(nodes, weights):
                s = midpoint + half * float(node)
                shape = cell.shape(2.0 * s - 1.0)
                point = shape @ positions
                relative = point - self.point
                radial = self.projector @ relative
                distance = float(np.linalg.norm(radial))
                if distance <= 0.0:
                    raise ValueError("radial beam contact has undefined obstacle normal")
                gap = distance - self.radius
                direction = radial / distance
                curvature = (self.projector - np.outer(direction, direction)) / distance
                measure = half * cell.length * float(weight)
                gradient = np.kron(shape, direction)
                local_energy += 0.5 * self.stiffness * gap * gap * measure
                local_force += self.stiffness * gap * gradient * measure
                local_tangent += self.stiffness * (
                    np.outer(gradient, gradient)
                    + gap * np.kron(np.outer(shape, shape), curvature)
                ) * measure
            return local_energy, local_force, local_tangent

        def refine(left: float, right: float, depth: int):
            nonlocal integration_points, maximum_error
            low = quadrature(left, right, _GAUSS_8)
            high = quadrature(left, right, _GAUSS_16)
            integration_points += 24
            energy_scale = max(abs(high[0]), 1.0e-14 * self.stiffness
                               * self.radius**2 * cell.length)
            force_scale = max(float(np.linalg.norm(high[1])),
                              1.0e-14 * self.stiffness * self.radius * cell.length)
            tangent_scale = max(float(np.linalg.norm(high[2])),
                                1.0e-14 * self.stiffness * cell.length)
            error = max(abs(high[0] - low[0]) / energy_scale,
                        float(np.linalg.norm(high[1] - low[1])) / force_scale,
                        float(np.linalg.norm(high[2] - low[2])) / tangent_scale)
            if error <= self.settings.line_integration_relative_tolerance:
                maximum_error = max(maximum_error, error)
                return high
            if depth >= self.settings.maximum_line_subdivisions:
                raise ValueError("radial beam integration subdivision budget exhausted")
            middle = 0.5 * (left + right)
            first = refine(left, middle, depth + 1)
            second = refine(middle, right, depth + 1)
            return (first[0] + second[0], first[1] + second[1],
                    first[2] + second[2])

        for left, right in intervals:
            result = refine(left, right, 0)
            energy += result[0]
            force += result[1]
            tangent += result[2]
        return (energy, force, tangent, active_length, len(intervals),
                integration_points, maximum_error)

    def evaluate(self, displacement: np.ndarray, *, tangent: bool = True):
        n_dof = self.model.mesh.dof_manager.total_dofs
        u = np.asarray(displacement, dtype=float).reshape(-1)
        if u.size != n_dof or not np.all(np.isfinite(u)):
            raise ValueError("radial beam displacement is invalid")
        force = np.zeros(n_dof)
        entries: dict[tuple[int, int], float] = {}
        energy = active_length = maximum_error = 0.0
        active_cells = intervals = points = 0
        for cell in self.cells:
            positions = self._positions(cell, u)
            local = self._integrate_cell(cell, positions)
            energy += local[0]
            active_length += local[3]
            active_cells += bool(local[4])
            intervals += local[4]
            points += local[5]
            maximum_error = max(maximum_error, local[6])
            dofs = cell.dofs.reshape(-1)
            np.add.at(force, dofs, local[1])
            if tangent:
                for row, global_row in enumerate(dofs):
                    for col, global_col in enumerate(dofs):
                        key = (int(global_row), int(global_col))
                        entries[key] = entries.get(key, 0.0) + float(local[2][row, col])
        stiffness = (sparse.coo_matrix(
            ([value for value in entries.values()],
             ([key[0] for key in entries], [key[1] for key in entries])),
            shape=(n_dof, n_dof),
        ).tocsr() if tangent else None)
        beam_force = np.zeros(3)
        beam_moment = np.zeros(3)
        unique_nodes: dict[int, tuple[np.ndarray, np.ndarray]] = {}
        for cell in self.cells:
            for node, reference, dofs in zip(cell.node_ids, cell.reference, cell.dofs):
                unique_nodes[node] = (reference, dofs)
        for reference, dofs in unique_nodes.values():
            physical = -force[dofs]
            beam_force += physical
            beam_moment += np.cross(reference + u[dofs], physical)
        minimum = self._minimum_gap(u)
        result = _RadialResults(
            self.profile, self.kind, minimum, energy, active_length, active_cells,
            intervals, points, maximum_error, len(self.cells),
            tuple(float(x) for x in beam_force),
            tuple(float(x) for x in -beam_force),
            tuple(float(x) for x in beam_moment),
            tuple(float(x) for x in -beam_moment),
            minimum >= -self.gap_tolerance, self.stiffness,
        )
        return force, stiffness, result

    def point_at(self, factor: float) -> np.ndarray:
        value = float(factor)
        if not np.isfinite(value) or value < 0.0:
            raise ValueError("radial tool path factor must be finite and nonnegative")
        point = self.reference_point + value * self.path_translation
        if not np.all(np.isfinite(point)):
            raise ValueError("radial tool path position is nonfinite")
        return point

    def set_path_factor(self, factor: float) -> None:
        self.point = self.point_at(factor)
        self.path_factor = float(factor)

    def _swept_admissible(self, start: np.ndarray, end: np.ndarray,
                         start_factor: float | None = None,
                         end_factor: float | None = None) -> bool:
        first_factor = self.path_factor if start_factor is None else float(start_factor)
        last_factor = self.path_factor if end_factor is None else float(end_factor)
        first_point = self.point_at(first_factor)
        last_point = self.point_at(last_factor)
        for cell in self.cells:
            first = self._positions(cell, start)
            last = self._positions(cell, end)
            relative_delta = (last - first) - (last_point - first_point)
            speed = (max(float(np.linalg.norm(delta)) for delta in relative_delta)
                     if len(first) == 2 else
                     sum(float(np.linalg.norm(delta)) for delta in relative_delta))

            def gap_at(factor: float) -> float:
                positions = first + factor * (last - first)
                self._check_positions(positions)
                point = first_point + factor * (last_point - first_point)
                return _minimum_gap(positions, point,
                                    self.projector, self.radius)

            initial_gap = gap_at(0.0)
            if initial_gap < -self.gap_tolerance:
                raise ValueError("committed radial beam pose exceeds the gap tolerance")
            final_gap = gap_at(1.0)
            if final_gap < -self.gap_tolerance:
                return False

            def certify(left: float, right: float, gap_left: float,
                        gap_right: float, depth: int) -> bool:
                if min(gap_left, gap_right) - 0.5 * speed * (right - left) >= -self.gap_tolerance:
                    return True
                if depth >= self.settings.maximum_swept_subdivisions:
                    return False
                middle = 0.5 * (left + right)
                gap_middle = gap_at(middle)
                if gap_middle < -self.gap_tolerance:
                    return False
                return (certify(left, middle, gap_left, gap_middle, depth + 1)
                        and certify(middle, right, gap_middle, gap_right, depth + 1))

            if not certify(0.0, 1.0, initial_gap, final_gap, 0):
                return False
        return True

    def swept_increment_admissible(self, start: np.ndarray, end: np.ndarray,
                                   start_factor: float, end_factor: float) -> bool:
        result = self._swept_admissible(np.asarray(start, dtype=float).reshape(-1),
                                       np.asarray(end, dtype=float).reshape(-1),
                                       start_factor, end_factor)
        if not result:
            self.swept_trial_cutbacks += 1
        return result

    def admissible_step_fraction(self, start: np.ndarray, end: np.ndarray) -> float:
        first = np.asarray(start, dtype=float).reshape(-1)
        last = np.asarray(end, dtype=float).reshape(-1)
        if self._swept_admissible(first, last):
            return 1.0
        self.swept_trial_cutbacks += 1
        fraction = 0.5
        for _ in range(self.settings.maximum_swept_subdivisions):
            if self._swept_admissible(first, first + fraction * (last - first)):
                return fraction
            fraction *= 0.5
        return 0.0

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
            raise ValueError("radial beam tool work is nonfinite")
        return work

    def restart_contract(self) -> dict[str, object]:
        surface, body, interaction = (self.definition.surfaces[0],
                                      self.definition.rigid_bodies[0],
                                      self.definition.interactions[0])
        return {
            "profile": self.profile,
            "surface": {"name": surface.name, "element_ids": list(surface.element_ids),
                        "beam_radius": surface.beam_radius,
                        "beam_end_caps": surface.beam_end_caps},
            "body": {"name": body.name, "kind": body.kind,
                     "point": list(body.point), "normal": (
                         None if body.normal is None else list(body.normal)),
                     "radius": body.radius,
                     **({"path_translation": list(body.path_translation)}
                        if body.path_translation is not None else {})},
            "interaction": {"name": interaction.name, "surface": interaction.surface,
                            "target": interaction.target,
                            "normal_law": interaction.normal_law,
                            "line_stiffness": interaction.line_stiffness,
                            "gap_tolerance": interaction.gap_tolerance},
            "settings": {"pair_motion": self.settings.pair_motion,
                         "maximum_candidate_pairs": self.settings.maximum_candidate_pairs,
                         "line_integration_relative_tolerance": self.settings.line_integration_relative_tolerance,
                         "maximum_line_subdivisions": self.settings.maximum_line_subdivisions,
                         "maximum_swept_subdivisions": self.settings.maximum_swept_subdivisions},
        }
