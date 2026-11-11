"""Distributed frictionless contact between two classical straight B2 cells.

Each direction integrates a line potential against the other deformed cell.
The accepted potential is the half-weighted sum, differentiated for both
bodies. Projection and contact intervals are partitioned before adaptive
quadrature. This is the first beam-pair kernel, not full C2 admission.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import sparse

from .core import ContactDefinition, ContactSolverSettings, ContactValidationReport


PROFILE = "CLASSICAL_B2_B2_ROUND_TUBE_DISTRIBUTED_PAIR_PENALTY_STATIC_DEV_V0"
PROFILE_CHAIN = "CLASSICAL_B2_CHAIN_ROUND_TUBE_NEAREST_DISTRIBUTED_PAIR_STATIC_DEV_V0"
_GAUSS_8 = np.polynomial.legendre.leggauss(8)
_GAUSS_16 = np.polynomial.legendre.leggauss(16)


@dataclass(frozen=True)
class _Segment:
    element_id: int
    node_ids: tuple[int, int]
    reference: np.ndarray
    dofs: np.ndarray
    length: float

    def positions(self, displacement: np.ndarray) -> np.ndarray:
        return self.reference + displacement[self.dofs]


@dataclass(frozen=True)
class _BeamPairResults:
    profile: str
    minimum_gap: float
    contact_energy: float
    active_source_intervals: int
    integration_points: int
    maximum_integration_relative_error: float
    beam_a_force: tuple[float, float, float]
    beam_b_force: tuple[float, float, float]
    beam_a_moment: tuple[float, float, float]
    beam_b_moment: tuple[float, float, float]
    gap_tolerance_met: bool
    line_stiffness: float

    def to_dict(self) -> dict[str, object]:
        return {
            "profile": self.profile,
            "minimum_gap": self.minimum_gap,
            "contact_energy": self.contact_energy,
            "algorithmic_contact_potential": self.contact_energy,
            "active_source_intervals": self.active_source_intervals,
            "integration_points": self.integration_points,
            "maximum_integration_relative_error": self.maximum_integration_relative_error,
            "beam_a_force": list(self.beam_a_force),
            "beam_b_force": list(self.beam_b_force),
            "beam_a_moment": list(self.beam_a_moment),
            "beam_b_moment": list(self.beam_b_moment),
            "gap_tolerance_met": self.gap_tolerance_met,
            "normal_law": "penalty",
            "line_stiffness_N_per_m2": self.line_stiffness,
        }


def validate_beam_pair_contact(model, definition: ContactDefinition,
                               settings: ContactSolverSettings | None = None
                               ) -> ContactValidationReport:
    if (len(definition.surfaces) == 2
            and all(len(surface.element_ids) == 1 for surface in definition.surfaces)):
        from anyfesolver.elements import BeamElement, QuadraticBeamElement

        element_types = tuple(type(model.mesh.elements.get(surface.element_ids[0]))
                              for surface in definition.surfaces)
        if (all(kind in (BeamElement, QuadraticBeamElement)
                for kind in element_types)
                and QuadraticBeamElement in element_types):
            from .beam_curve_pair_static import validate_b3_pair_contact

            return validate_b3_pair_contact(model, definition, settings)
    if (len(definition.surfaces) == 2
            and any(len(surface.element_ids) > 1 for surface in definition.surfaces)):
        from .beam_chain_pair import validate_beam_chain_pair_contact

        return validate_beam_chain_pair_contact(model, definition, settings)
    issues: list[str] = []
    if tuple(map(len, (definition.surfaces, definition.rigid_bodies,
                       definition.interactions))) != (2, 0, 1):
        return ContactValidationReport("invalid", (
            "beam-pair profile requires two beam surfaces, no rigid body and one interaction",
        ), PROFILE)
    surface_a, surface_b = definition.surfaces
    interaction = definition.interactions[0]
    controls = ContactSolverSettings() if settings is None else settings
    if controls.pair_motion != "general":
        issues.append("beam-pair contact requires pair_motion='general'")
    if controls.line_integration_relative_tolerance is None:
        issues.append("beam-pair contact requires explicit line_integration_relative_tolerance")
    if (interaction.surface, interaction.target) != (surface_a.name, surface_b.name):
        issues.append("beam-pair interaction must name integration and target surfaces in order")
    if (interaction.normal_law != "penalty" or interaction.normal_stiffness is not None
            or interaction.line_stiffness is None or interaction.endpoint_stiffness is not None):
        issues.append("beam-pair contact requires a penalty line_stiffness only")
    if len(surface_a.element_ids) * len(surface_b.element_ids) > controls.maximum_candidate_pairs:
        issues.append("beam-pair candidate limit exceeded; no pair is discarded")
    from anyfesolver.elements import BeamElement
    from anymaterial import IsotropicMaterial

    selected_nodes: list[set[int]] = []
    for surface in (surface_a, surface_b):
        if (surface.beam_radius is None or surface.side != "midsurface"
                or surface.beam_end_caps != "none"):
            issues.append(f"surface {surface.name}: explicit radius and no endpoint caps required")
        if len(surface.element_ids) != 1:
            issues.append(f"surface {surface.name}: this first pair profile admits one B2 cell")
        nodes: set[int] = set()
        selected_nodes.append(nodes)
        for element_id in surface.element_ids:
            element = model.mesh.elements.get(element_id)
            if type(element) is not BeamElement:
                issues.append(f"element {element_id} is not an admitted classical B2 cell")
                continue
            nodes.update(int(node) for node in element.node_ids)
            if (element.generalized_section is not None
                    or element._fiber_plasticity is not None
                    or getattr(element, "_geometric_nonlinearity", None) != "von_karman"):
                issues.append(f"element {element_id} needs an elastic scalar von Karman section")
            material = model.materials.get(element.material_name)
            if (type(material) is not IsotropicMaterial
                    or material.yield_stress != 0.0
                    or material.hardening_curve is not None):
                issues.append(f"element {element_id} needs an isotropic elastic material")
            try:
                reference = np.asarray(element.get_node_coordinates(model.mesh), dtype=float)
                if not np.all(np.isfinite(reference)) or (
                    float(np.linalg.norm(reference[1] - reference[0])) <= 0.0
                ):
                    raise ValueError("zero or nonfinite reference length")
            except (ValueError, KeyError) as exc:
                issues.append(f"element {element_id}: {exc}")
    if selected_nodes[0] & selected_nodes[1]:
        issues.append("beam-pair profile does not admit shared material nodes")
    if (surface_a.beam_radius is not None and surface_b.beam_radius is not None
            and interaction.gap_tolerance >= surface_a.beam_radius + surface_b.beam_radius):
        issues.append("beam-pair gap tolerance must be smaller than combined radius")
    return ContactValidationReport("invalid" if issues else "ok", tuple(issues), PROFILE)


def _closest_on_segment(point: np.ndarray, start: np.ndarray,
                        vector: np.ndarray) -> float:
    squared_length = float(vector @ vector)
    if squared_length <= 1.0e-30:
        raise ValueError("deformed beam-pair segment has zero length")
    return float(np.clip(((point - start) @ vector) / squared_length, 0.0, 1.0))


def _minimum_segment_gap(a: np.ndarray, b: np.ndarray, radius: float) -> float:
    va = a[1] - a[0]
    vb = b[1] - b[0]
    candidates: list[float] = []
    cuts = _projection_cuts(a, b)
    for left, right in zip(cuts[:-1], cuts[1:]):
        midpoint = 0.5 * (left + right)
        projected = _closest_on_segment(a[0] + midpoint * va, b[0], vb)
        if projected == 0.0:
            relative0, relative1 = a[0] - b[0], va
        elif projected == 1.0:
            relative0, relative1 = a[0] - b[1], va
        else:
            t0 = float((a[0] - b[0]) @ vb / (vb @ vb))
            t1 = float(va @ vb / (vb @ vb))
            relative0, relative1 = a[0] - b[0] - t0 * vb, va - t1 * vb
        points = [left, right]
        squared_slope = float(relative1 @ relative1)
        if squared_slope > 0.0:
            stationary = -float(relative0 @ relative1) / squared_slope
            if left < stationary < right:
                points.append(stationary)
        candidates.extend(float(np.linalg.norm(relative0 + s * relative1))
                          for s in points)
    return min(candidates) - radius


def _projection_cuts(source: np.ndarray, target: np.ndarray) -> tuple[float, ...]:
    va = source[1] - source[0]
    vb = target[1] - target[0]
    squared_length = float(vb @ vb)
    if squared_length <= 1.0e-30:
        raise ValueError("deformed beam-pair segment has zero length")
    t0 = float((source[0] - target[0]) @ vb / squared_length)
    t1 = float(va @ vb / squared_length)
    cuts = [0.0, 1.0]
    if t1 != 0.0:
        for boundary in (0.0, 1.0):
            value = (boundary - t0) / t1
            if 0.0 < value < 1.0:
                cuts.append(value)
    return tuple(sorted(set(cuts)))


def _real_roots(coefficients: tuple[float, float, float],
                left: float, right: float) -> tuple[float, ...]:
    a, b, c = coefficients
    scale = max(abs(a), abs(b), abs(c))
    if scale == 0.0:
        return ()
    normalized = np.asarray((a, b, c), dtype=float) / scale
    roots = np.roots(normalized if abs(normalized[0]) > 64.0 * np.finfo(float).eps
                     else normalized[1:])
    return tuple(sorted(float(np.real(root)) for root in roots
                        if abs(float(np.imag(root))) <= 1.0e-12
                        and left < float(np.real(root)) < right))


def _contact_intervals(source: np.ndarray, target: np.ndarray,
                       radius: float) -> tuple[tuple[float, float], ...]:
    va = source[1] - source[0]
    vb = target[1] - target[0]
    bb = float(vb @ vb)
    t0 = float((source[0] - target[0]) @ vb / bb)
    t1 = float(va @ vb / bb)
    cuts = _projection_cuts(source, target)
    active: list[tuple[float, float]] = []
    for left, right in zip(cuts[:-1], cuts[1:]):
        mid = 0.5 * (left + right)
        raw_t = t0 + t1 * mid
        if raw_t <= 0.0:
            relative0, relative1 = source[0] - target[0], va
        elif raw_t >= 1.0:
            relative0, relative1 = source[0] - target[1], va
        else:
            relative0 = source[0] - target[0] - t0 * vb
            relative1 = va - t1 * vb
        coefficients = (float(relative1 @ relative1),
                        2.0 * float(relative0 @ relative1),
                        float(relative0 @ relative0) - radius * radius)
        subcuts = [left, *_real_roots(coefficients, left, right), right]
        for subleft, subright in zip(subcuts[:-1], subcuts[1:]):
            centre = 0.5 * (subleft + subright)
            value = (coefficients[0] * centre + coefficients[1]) * centre + coefficients[2]
            if value <= 0.0:
                active.append((subleft, subright))
    return tuple(active)


def _point_law(source: np.ndarray, target: np.ndarray, s: float,
               radius: float, source_is_a: bool) -> tuple[float, np.ndarray, np.ndarray]:
    source_position = (1.0 - s) * source[0] + s * source[1]
    target_vector = target[1] - target[0]
    raw_t = float((source_position - target[0]) @ target_vector
                  / (target_vector @ target_vector))
    t = float(np.clip(raw_t, 0.0, 1.0))
    separation = source_position - ((1.0 - t) * target[0] + t * target[1])
    return _projection_work((1.0-s,s),(1.0-t,t),separation,target_vector,
                            0.0<raw_t<1.0,radius,source_is_a)


def _projection_work(source_shape,target_shape,separation,target_vector,interior,radius,source_is_a=True):
    """Physical positional work from one resolved closest-point projection."""
    distance = float(np.linalg.norm(separation))
    if distance <= 0.0:
        raise ValueError("coincident beam centreline points cannot define a contact normal")
    direction = separation / distance
    projector = np.eye(3) - np.outer(direction, direction)
    coefficients = ([*source_shape,*(-v for v in target_shape)] if source_is_a else
                    [*(-v for v in target_shape),*source_shape])
    jacobian = np.hstack([coefficient * np.eye(3) for coefficient in coefficients])
    gradient = jacobian.T @ direction
    hessian = jacobian.T @ projector @ jacobian / distance
    if interior:
        derivative_coefficients = ([0.0, 0.0, 1.0, -1.0] if source_is_a else
                                   [1.0, -1.0, 0.0, 0.0])
        jacobian_t = np.hstack([coefficient * np.eye(3)
                                for coefficient in derivative_coefficients])
        separation_t = -target_vector
        mixed = (jacobian_t.T @ direction
                 + jacobian.T @ projector @ separation_t / distance)
        curvature = float(separation_t @ projector @ separation_t / distance)
        if curvature <= 0.0:
            raise ValueError("beam-pair projection has a singular interior minimum")
        hessian -= np.outer(mixed, mixed) / curvature
    return distance - radius, gradient, hessian


class _BeamPairManager:
    def __init__(self, model, definition: ContactDefinition,
                 settings: ContactSolverSettings | None = None,
                 initial_displacement: np.ndarray | None = None):
        self.settings = ContactSolverSettings() if settings is None else settings
        report = validate_beam_pair_contact(model, definition, self.settings)
        if not report.valid:
            raise ValueError("unsupported beam-pair contact: " + "; ".join(report.issues))
        self.model = model
        self.definition = definition
        self.profile = report.profile
        self.stiffness = definition.interactions[0].line_stiffness
        self.gap_tolerance = definition.interactions[0].gap_tolerance
        self.radius = sum(surface.beam_radius for surface in definition.surfaces)
        self.end_caps = "none"
        self.swept_trial_cutbacks = 0
        self.segments = []
        for surface in definition.surfaces:
            element_id = surface.element_ids[0]
            element = model.mesh.elements[element_id]
            reference = np.asarray(element.get_node_coordinates(model.mesh), dtype=float)
            dofs = np.asarray([model.mesh.dof_manager.get_node_dofs(node)[:3]
                               for node in element.node_ids], dtype=np.intp)
            self.segments.append(_Segment(
                element_id, tuple(int(node) for node in element.node_ids),
                reference, dofs, float(np.linalg.norm(reference[1] - reference[0])),
            ))
        n_dof = model.mesh.dof_manager.total_dofs
        initial = np.zeros(n_dof) if initial_displacement is None else np.asarray(
            initial_displacement, dtype=float,
        ).reshape(-1)
        if initial.size != n_dof or not np.all(np.isfinite(initial)):
            raise ValueError("beam-pair initial displacement is invalid")
        if self._minimum_gap(initial) < -256.0 * np.finfo(float).eps * max(
            self.radius, *(segment.length for segment in self.segments)
        ):
            raise ValueError("beam-pair contact has unresolved initial penetration")

    def _positions(self, u: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        positions = tuple(segment.positions(u) for segment in self.segments)
        for position in positions:
            if not np.all(np.isfinite(position)) or float(np.linalg.norm(
                position[1] - position[0]
            )) <= 1.0e-15:
                raise ValueError("deformed beam-pair segment has zero or invalid length")
        return positions

    def _minimum_gap(self, u: np.ndarray) -> float:
        return _minimum_segment_gap(*self._positions(u), self.radius)

    def _integrate_direction(self, source: np.ndarray, target: np.ndarray,
                             reference_length: float, source_is_a: bool
                             ) -> tuple[float, np.ndarray, np.ndarray, int, int, float]:
        intervals = _contact_intervals(source, target, self.radius)
        total_energy = 0.0
        total_force = np.zeros(12)
        total_tangent = np.zeros((12, 12))
        points = 0
        maximum_error = 0.0

        def quadrature(left: float, right: float, rule):
            nodes, weights = rule
            centre, half = 0.5 * (left + right), 0.5 * (right - left)
            energy, force, tangent = 0.0, np.zeros(12), np.zeros((12, 12))
            for node, weight in zip(nodes, weights):
                s = centre + half * float(node)
                gap, gradient, hessian = _point_law(
                    source, target, s, self.radius, source_is_a,
                )
                measure = half * reference_length * float(weight)
                energy += 0.5 * self.stiffness * gap * gap * measure
                force += self.stiffness * gap * gradient * measure
                tangent += self.stiffness * (
                    np.outer(gradient, gradient) + gap * hessian
                ) * measure
            return energy, force, tangent

        def refine(left: float, right: float, depth: int):
            nonlocal points, maximum_error
            low = quadrature(left, right, _GAUSS_8)
            high = quadrature(left, right, _GAUSS_16)
            points += 24
            energy_scale = max(abs(high[0]), 1.0e-14 * self.stiffness
                               * self.radius**2 * reference_length)
            force_scale = max(float(np.linalg.norm(high[1])),
                              1.0e-14 * self.stiffness * self.radius * reference_length)
            tangent_scale = max(float(np.linalg.norm(high[2])),
                                1.0e-14 * self.stiffness * reference_length)
            error = max(
                abs(high[0] - low[0]) / energy_scale,
                float(np.linalg.norm(high[1] - low[1])) / force_scale,
                float(np.linalg.norm(high[2] - low[2])) / tangent_scale,
            )
            if error <= self.settings.line_integration_relative_tolerance:
                maximum_error = max(maximum_error, error)
                return high
            if depth >= self.settings.maximum_line_subdivisions:
                raise ValueError("beam-pair integration subdivision budget exhausted")
            middle = 0.5 * (left + right)
            first = refine(left, middle, depth + 1)
            second = refine(middle, right, depth + 1)
            return (first[0] + second[0], first[1] + second[1],
                    first[2] + second[2])

        for left, right in intervals:
            energy, force, tangent = refine(left, right, 0)
            total_energy += energy
            total_force += force
            total_tangent += tangent
        return total_energy, total_force, total_tangent, len(intervals), points, maximum_error

    def evaluate(self, displacement: np.ndarray, *, tangent: bool = True):
        n_dof = self.model.mesh.dof_manager.total_dofs
        u = np.asarray(displacement, dtype=float).reshape(-1)
        if u.size != n_dof or not np.all(np.isfinite(u)):
            raise ValueError("beam-pair displacement is invalid")
        a, b = self._positions(u)
        first = self._integrate_direction(a, b, self.segments[0].length, True)
        second = self._integrate_direction(b, a, self.segments[1].length, False)
        energy = 0.5 * (first[0] + second[0])
        local_force = 0.5 * (first[1] + second[1])
        local_tangent = 0.5 * (first[2] + second[2])
        dofs = np.r_[self.segments[0].dofs.reshape(-1),
                     self.segments[1].dofs.reshape(-1)]
        force = np.zeros(n_dof)
        np.add.at(force, dofs, local_force)
        if tangent:
            rows, cols = np.meshgrid(dofs, dofs, indexing="ij")
            stiffness = sparse.coo_matrix(
                (local_tangent.reshape(-1), (rows.reshape(-1), cols.reshape(-1))),
                shape=(n_dof, n_dof),
            ).tocsr()
        else:
            stiffness = None
        physical_a = -local_force[:6].reshape(2, 3)
        physical_b = -local_force[6:].reshape(2, 3)
        force_a = np.sum(physical_a, axis=0)
        force_b = np.sum(physical_b, axis=0)
        moment_a = np.sum(np.cross(a, physical_a), axis=0)
        moment_b = np.sum(np.cross(b, physical_b), axis=0)
        minimum = _minimum_segment_gap(a, b, self.radius)
        results = _BeamPairResults(
            self.profile, minimum, energy, first[3] + second[3],
            first[4] + second[4], max(first[5], second[5]),
            tuple(float(x) for x in force_a), tuple(float(x) for x in force_b),
            tuple(float(x) for x in moment_a), tuple(float(x) for x in moment_b),
            minimum >= -self.gap_tolerance, self.stiffness,
        )
        return force, stiffness, results

    def _swept_admissible(self, start: np.ndarray, end: np.ndarray) -> bool:
        start_positions = self._positions(start)
        end_positions = self._positions(end)
        relative_speed = max(float(np.linalg.norm(
            (end_positions[0][i] - start_positions[0][i])
            - (end_positions[1][j] - start_positions[1][j])
        )) for i in range(2) for j in range(2))

        def gap_at(factor: float) -> float:
            positions = tuple(first + factor * (last - first)
                              for first, last in zip(start_positions, end_positions))
            return _minimum_segment_gap(*positions, self.radius)

        first_gap = gap_at(0.0)
        if first_gap < -self.gap_tolerance:
            raise ValueError("committed beam-pair pose exceeds the gap tolerance")
        last_gap = gap_at(1.0)
        if last_gap < -self.gap_tolerance:
            return False

        def certify(left: float, right: float, gap_left: float,
                    gap_right: float, depth: int) -> bool:
            if min(gap_left, gap_right) - 0.5 * relative_speed * (right - left) >= -self.gap_tolerance:
                return True
            if depth >= self.settings.maximum_swept_subdivisions:
                return False
            middle = 0.5 * (left + right)
            gap_middle = gap_at(middle)
            if gap_middle < -self.gap_tolerance:
                return False
            return (certify(left, middle, gap_left, gap_middle, depth + 1)
                    and certify(middle, right, gap_middle, gap_right, depth + 1))

        return certify(0.0, 1.0, first_gap, last_gap, 0)

    def swept_increment_admissible(self, start: np.ndarray, end: np.ndarray,
                                   _start_factor: float,
                                   _end_factor: float) -> bool:
        admissible = self._swept_admissible(
            np.asarray(start, dtype=float).reshape(-1),
            np.asarray(end, dtype=float).reshape(-1),
        )
        if not admissible:
            self.swept_trial_cutbacks += 1
        return admissible

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

    def restart_contract(self) -> dict[str, object]:
        interaction = self.definition.interactions[0]
        return {
            "profile": self.profile,
            "surfaces": [
                {"name": surface.name, "element_ids": list(surface.element_ids),
                 "beam_radius": surface.beam_radius,
                 "beam_end_caps": surface.beam_end_caps}
                for surface in self.definition.surfaces
            ],
            "interaction": {
                "name": interaction.name, "surface": interaction.surface,
                "target": interaction.target,
                "line_stiffness": interaction.line_stiffness,
                "gap_tolerance": interaction.gap_tolerance,
                "normal_law": interaction.normal_law,
            },
            "settings": {
                "pair_motion": self.settings.pair_motion,
                "maximum_candidate_pairs": self.settings.maximum_candidate_pairs,
                "line_integration_relative_tolerance": self.settings.line_integration_relative_tolerance,
                "maximum_line_subdivisions": self.settings.maximum_line_subdivisions,
                "maximum_swept_subdivisions": self.settings.maximum_swept_subdivisions,
            },
        }
