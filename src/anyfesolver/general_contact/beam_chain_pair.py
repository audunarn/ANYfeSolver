"""Symmetric B2-chain contact with one nearest target cell per source point.

The physical potential is half the integral of the nearest-cell line law in
each direction. Projection, Voronoi ownership and activation are partitioned
before quadrature. Cell IDs and material node topology own the features.
"""

from __future__ import annotations

import numpy as np
from scipy import sparse

from .beam_pair import (
    PROFILE_CHAIN, _BeamPairManager, _BeamPairResults, _GAUSS_8, _GAUSS_16,
    _Segment, _minimum_segment_gap, _point_law, _projection_cuts, _real_roots,
)
from .core import ContactDefinition, ContactSolverSettings, ContactValidationReport

PROFILE_CONSISTENT = "CLASSICAL_B2_CHAIN_ROUND_TUBE_NEAREST_CONSISTENT_STATIC_DEV_V1"


def _chain_topology(segments: tuple[_Segment, ...]) -> None:
    neighbours: dict[int, set[int]] = {}
    edges = set()
    for segment in segments:
        a, b = segment.node_ids
        edge = tuple(sorted((a, b)))
        if a == b or edge in edges:
            raise ValueError("beam chain has a duplicate or collapsed cell")
        edges.add(edge)
        neighbours.setdefault(a, set()).add(b)
        neighbours.setdefault(b, set()).add(a)
    if (sum(len(adjacent) == 1 for adjacent in neighbours.values()) != 2
            or any(len(adjacent) > 2 for adjacent in neighbours.values())):
        raise ValueError("beam chain must have two physical ends and no branch or cycle")
    seen = set()
    frontier = [next(iter(neighbours))]
    while frontier:
        node = frontier.pop()
        if node not in seen:
            seen.add(node)
            frontier.extend(neighbours[node] - seen)
    if len(seen) != len(neighbours):
        raise ValueError("beam chain cells must be connected through material nodes")


def validate_beam_chain_pair_contact(model, definition: ContactDefinition,
                                     settings: ContactSolverSettings | None = None,
                                     ) -> ContactValidationReport:
    controls = ContactSolverSettings() if settings is None else settings
    profile = PROFILE_CONSISTENT if controls.beam_chain_tangent == "consistent_dev" else PROFILE_CHAIN
    issues = []
    if tuple(map(len, (definition.surfaces, definition.rigid_bodies,
                       definition.interactions))) != (2, 0, 1):
        return ContactValidationReport("invalid", (
            "beam-chain pair needs two beam surfaces and one interaction",
        ), profile)
    a, b = definition.surfaces
    interaction = definition.interactions[0]
    if (interaction.surface, interaction.target) != (a.name, b.name):
        issues.append("beam-chain interaction must name source and target in surface order")
    if (interaction.normal_law != "penalty" or interaction.normal_stiffness is not None
            or interaction.line_stiffness is None or interaction.endpoint_stiffness is not None):
        issues.append("beam-chain pair requires only penalty line stiffness")
    if controls.pair_motion != "general":
        issues.append("beam-chain pair requires pair_motion='general'")
    if controls.line_integration_relative_tolerance is None:
        issues.append("beam-chain pair requires explicit line integration tolerance")
    if controls.beam_chain_tangent == "finite_difference" and controls.sliding_difference_step is None:
        issues.append("beam-chain pair requires explicit dimensional sliding_difference_step")
    if (len(a.element_ids) * len(b.element_ids) > controls.maximum_candidate_pairs
            or len(a.element_ids) > 16 or len(b.element_ids) > 16):
        issues.append("beam-chain pair exceeds its admitted cell resource limit; no contact is dropped")
    from anyfesolver.elements import BeamElement
    from anymaterial import IsotropicMaterial

    selected_nodes = []
    for surface in (a, b):
        if (surface.beam_radius is None or surface.side != "midsurface"
                or surface.beam_end_caps != "none"):
            issues.append(f"surface {surface.name} needs a round radius and no caps")
        if not surface.element_ids:
            issues.append(f"surface {surface.name} has no beam cells")
        segments = []
        nodes = set()
        for element_id in surface.element_ids:
            element = model.mesh.elements.get(element_id)
            if type(element) is not BeamElement:
                issues.append(f"element {element_id} is not an admitted classical B2 cell")
                continue
            node_ids = tuple(int(n) for n in element.node_ids)
            nodes.update(node_ids)
            if (element.generalized_section is not None
                    or element._fiber_plasticity is not None
                    or getattr(element, "_geometric_nonlinearity", None) != "von_karman"):
                issues.append(f"element {element_id} needs an elastic scalar von Karman section")
            material = model.materials.get(element.material_name)
            if (type(material) is not IsotropicMaterial or material.yield_stress != 0.0
                    or material.hardening_curve is not None):
                issues.append(f"element {element_id} needs an isotropic elastic material")
            try:
                reference = np.asarray(element.get_node_coordinates(model.mesh), dtype=float)
                length = float(np.linalg.norm(reference[1] - reference[0]))
                if not np.all(np.isfinite(reference)) or length <= 0.0:
                    raise ValueError("zero or nonfinite reference length")
                segments.append(_Segment(element_id, node_ids, reference,
                                         np.zeros((2, 3), dtype=np.intp), length))
            except (ValueError, KeyError) as exc:
                issues.append(f"element {element_id}: {exc}")
        if len(segments) == len(surface.element_ids):
            try:
                _chain_topology(tuple(segments))
            except ValueError as exc:
                issues.append(f"surface {surface.name}: {exc}")
        selected_nodes.append(nodes)
    if selected_nodes[0] & selected_nodes[1]:
        issues.append("beam-chain pair surfaces must not share material nodes")
    return ContactValidationReport("invalid" if issues else "ok", tuple(issues), profile)


def _distance_coefficients(source: np.ndarray, target: np.ndarray,
                           midpoint: float) -> tuple[float, float, float]:
    va = source[1] - source[0]
    vb = target[1] - target[0]
    length2 = float(vb @ vb)
    if length2 <= 1.0e-30:
        raise ValueError("deformed beam-chain cell has zero length")
    t0 = float((source[0] - target[0]) @ vb / length2)
    t1 = float(va @ vb / length2)
    projection = t0 + t1 * midpoint
    if projection <= 0.0:
        r0, r1 = source[0] - target[0], va
    elif projection >= 1.0:
        r0, r1 = source[0] - target[1], va
    else:
        r0, r1 = source[0] - target[0] - t0 * vb, va - t1 * vb
    return float(r1 @ r1), 2.0 * float(r0 @ r1), float(r0 @ r0)


def _owned_intervals(source: np.ndarray, targets: tuple[np.ndarray, ...],
                     target_ids: tuple[int, ...], radius: float,
                     ) -> tuple[tuple[int, float, float], ...]:
    """Exact B2 projection, nearest-cell and contact cuts in source s in [0,1]."""
    cuts = {0.0, 1.0}
    for target in targets:
        cuts.update(_projection_cuts(source, target))
    projection_cuts = sorted(cuts)
    all_cuts = set(projection_cuts)
    for left, right in zip(projection_cuts[:-1], projection_cuts[1:]):
        midpoint = 0.5 * (left + right)
        coefficients = [_distance_coefficients(source, target, midpoint)
                        for target in targets]
        for i, first in enumerate(coefficients):
            for second in coefficients[i + 1:]:
                difference = tuple(x - y for x, y in zip(first, second))
                all_cuts.update(_real_roots(difference, left, right))
    sorted_cuts = sorted(all_cuts)
    intervals = []
    for left, right in zip(sorted_cuts[:-1], sorted_cuts[1:]):
        if right - left <= 1.0e-12:
            continue
        midpoint = 0.5 * (left + right)
        coefficients = [_distance_coefficients(source, target, midpoint)
                        for target in targets]
        nearest = min(range(len(targets)), key=lambda j: (
            np.polyval(coefficients[j], midpoint), target_ids[j],
        ))
        a, b, c = coefficients[nearest]
        activation = (a, b, c - radius * radius)
        subcuts = [left, *_real_roots(activation, left, right), right]
        for start, stop in zip(subcuts[:-1], subcuts[1:]):
            if stop - start <= 1.0e-12:
                continue
            centre = 0.5 * (start + stop)
            if np.polyval(activation, centre) <= 0.0:
                if (intervals and intervals[-1][0] == nearest
                        and abs(intervals[-1][2] - start) <= 1.0e-12):
                    intervals[-1] = (nearest, intervals[-1][1], stop)
                else:
                    intervals.append((nearest, start, stop))
    return tuple(intervals)


class _BeamChainPairManager(_BeamPairManager):
    def __init__(self, model, definition: ContactDefinition,
                 settings: ContactSolverSettings | None = None,
                 initial_displacement: np.ndarray | None = None):
        self.settings = ContactSolverSettings() if settings is None else settings
        report = validate_beam_chain_pair_contact(model, definition, self.settings)
        if not report.valid:
            raise ValueError("unsupported beam-chain pair: " + "; ".join(report.issues))
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
            cells = []
            for element_id in surface.element_ids:
                element = model.mesh.elements[element_id]
                reference = np.asarray(element.get_node_coordinates(model.mesh), dtype=float)
                dofs = np.asarray([model.mesh.dof_manager.get_node_dofs(node)[:3]
                                   for node in element.node_ids], dtype=np.intp)
                cells.append(_Segment(
                    element_id, tuple(int(n) for n in element.node_ids),
                    reference, dofs, float(np.linalg.norm(reference[1] - reference[0])),
                ))
            self.segments.append(tuple(cells))
        n_dof = model.mesh.dof_manager.total_dofs
        initial = np.zeros(n_dof) if initial_displacement is None else np.asarray(
            initial_displacement, dtype=float,
        ).reshape(-1)
        if initial.size != n_dof or not np.all(np.isfinite(initial)):
            raise ValueError("beam-chain pair initial displacement is invalid")
        scale = max(self.radius, *(cell.length for side in self.segments for cell in side))
        if self._minimum_gap(initial) < -256.0 * np.finfo(float).eps * scale:
            raise ValueError("beam-chain pair has unresolved initial penetration")

    def _positions(self, u: np.ndarray):
        positions = tuple(tuple(cell.positions(u) for cell in side)
                          for side in self.segments)
        for side in positions:
            for cell in side:
                if (not np.all(np.isfinite(cell))
                        or float(np.linalg.norm(cell[1] - cell[0])) <= 1.0e-15):
                    raise ValueError("deformed beam-chain cell has zero or invalid length")
        return positions

    def _minimum_gap(self, u: np.ndarray) -> float:
        a, b = self._positions(u)
        return min(_minimum_segment_gap(first, second, self.radius)
                   for first in a for second in b)

    def _integrate(self, source: np.ndarray, target: np.ndarray,
                   reference_length: float, source_is_a: bool):
        def quadrature(left, right, rule):
            nodes, weights = rule
            centre, half = 0.5 * (left + right), 0.5 * (right - left)
            energy, force = 0.0, np.zeros(12)
            for node, weight in zip(nodes, weights):
                s = centre + half * float(node)
                gap, gradient, _ = _point_law(
                    source, target, s, self.radius, source_is_a,
                )
                measure = half * reference_length * float(weight)
                energy += 0.5 * self.stiffness * gap * gap * measure
                force += self.stiffness * gap * gradient * measure
            return energy, force

        def refine(left, right, depth):
            low = quadrature(left, right, _GAUSS_8)
            high = quadrature(left, right, _GAUSS_16)
            energy_scale = max(abs(high[0]), 1.0e-14 * self.stiffness
                               * self.radius**2 * reference_length)
            force_scale = max(float(np.linalg.norm(high[1])),
                              1.0e-14 * self.stiffness * self.radius * reference_length)
            error = max(abs(high[0] - low[0]) / energy_scale,
                        float(np.linalg.norm(high[1] - low[1])) / force_scale)
            if error <= self.settings.line_integration_relative_tolerance:
                return high[0], high[1], 24, error
            if depth >= self.settings.maximum_line_subdivisions:
                raise ValueError(
                    "beam-chain integration subdivision budget exhausted: "
                    f"interval=({left:.12g},{right:.12g}), relative_error={error:.6g}"
                )
            middle = 0.5 * (left + right)
            first = refine(left, middle, depth + 1)
            second = refine(middle, right, depth + 1)
            return (first[0] + second[0], first[1] + second[1],
                    first[2] + second[2], max(first[3], second[3]))

        return refine

    def _assemble(self, u: np.ndarray):
        n_dof = self.model.mesh.dof_manager.total_dofs
        if u.size != n_dof or not np.all(np.isfinite(u)):
            raise ValueError("beam-chain pair displacement is invalid")
        positions = self._positions(u)
        force = np.zeros(n_dof)
        energy = 0.0
        count = points = 0
        maximum_error = 0.0
        for direction in (0, 1):
            source_cells = self.segments[direction]
            target_cells = self.segments[1 - direction]
            source_positions = positions[direction]
            target_positions = positions[1 - direction]
            target_ids = tuple(cell.element_id for cell in target_cells)
            for source_cell, source in zip(source_cells, source_positions):
                windows = _owned_intervals(source, target_positions,
                                           target_ids, self.radius)
                count += len(windows)
                for target_index, left, right in windows:
                    target_cell = target_cells[target_index]
                    target = target_positions[target_index]
                    integrate = self._integrate(source, target, source_cell.length,
                                                direction == 0)
                    try:
                        local_energy, local_force, local_points, local_error = integrate(
                            left, right, 0,
                        )
                    except ValueError as exc:
                        raise ValueError(
                            f"beam-chain cells {source_cell.element_id}/"
                            f"{target_cell.element_id}: {exc}"
                        ) from exc
                    energy += 0.5 * local_energy
                    points += local_points
                    maximum_error = max(maximum_error, local_error)
                    a_cell, b_cell = ((source_cell, target_cell) if direction == 0
                                      else (target_cell, source_cell))
                    dofs = np.r_[a_cell.dofs.reshape(-1), b_cell.dofs.reshape(-1)]
                    np.add.at(force, dofs, 0.5 * local_force)
        return force, energy, count, points, maximum_error

    def evaluate(self, displacement: np.ndarray, *, tangent: bool = True):
        u = np.asarray(displacement, dtype=float).reshape(-1)
        consistent = self.settings.beam_chain_tangent == "consistent_dev"
        if consistent:
            from .beam_chain_consistent import assemble_consistent_chain
            force, stiffness, energy, count, points, error, interfaces = assemble_consistent_chain(self,u,tangent)
            self.last_force_assemblies = 1
            self.last_moving_owner_interfaces = interfaces
        else:
            force, energy, count, points, error = self._assemble(u)
            stiffness = None
            self.last_force_assemblies = 1
            self.last_moving_owner_interfaces = None
        if tangent and not consistent:
            columns = sorted({int(dof) for side in self.segments for cell in side
                              for dof in cell.dofs.reshape(-1)})
            rows, cols, values = [], [], []
            step = self.settings.sliding_difference_step
            for column in columns:
                perturbation = np.zeros_like(u)
                perturbation[column] = step
                plus = self._assemble(u + perturbation)[0]
                minus = self._assemble(u - perturbation)[0]
                self.last_force_assemblies += 2
                derivative = (plus - minus) / (2.0 * step)
                nonzero = np.flatnonzero(derivative)
                rows.extend(int(row) for row in nonzero)
                cols.extend([column] * len(nonzero))
                values.extend(float(derivative[row]) for row in nonzero)
            stiffness = sparse.coo_matrix((values, (rows, cols)),
                                          shape=(len(u), len(u))).tocsr()
        wrenches = []
        for side in self.segments:
            nodes = {node for cell in side for node in cell.node_ids}
            body_force = np.zeros(3)
            body_moment = np.zeros(3)
            for node in nodes:
                dofs = self.model.mesh.dof_manager.get_node_dofs(node)[:3]
                physical = -force[dofs]
                reference = np.asarray(self.model.mesh.get_node(node).coords(), dtype=float)
                body_force += physical
                body_moment += np.cross(reference + u[dofs], physical)
            wrenches.append((tuple(map(float, body_force)),
                             tuple(map(float, body_moment))))
        minimum = self._minimum_gap(u)
        result = _BeamPairResults(
            self.profile, minimum, energy, count, points, error,
            wrenches[0][0], wrenches[1][0], wrenches[0][1], wrenches[1][1],
            minimum >= -self.gap_tolerance, self.stiffness,
        )
        return force, stiffness, result

    def _swept_admissible(self, start: np.ndarray, end: np.ndarray) -> bool:
        first = self._positions(start)
        last = self._positions(end)
        relative_speed = max(float(np.linalg.norm(
            (last[0][i][m] - first[0][i][m])
            - (last[1][j][n] - first[1][j][n])
        )) for i in range(len(first[0])) for j in range(len(first[1]))
            for m in range(2) for n in range(2))

        def gap_at(factor):
            positions = tuple(tuple(a + factor * (b - a)
                                    for a, b in zip(side_a, side_b))
                              for side_a, side_b in zip(first, last))
            return min(_minimum_segment_gap(a, b, self.radius)
                       for a in positions[0] for b in positions[1])

        first_gap = gap_at(0.0)
        if first_gap < -self.gap_tolerance:
            raise ValueError("committed beam-chain pose exceeds the gap tolerance")
        last_gap = gap_at(1.0)
        if last_gap < -self.gap_tolerance:
            return False

        def certify(left, right, gap_left, gap_right, depth):
            if (min(gap_left, gap_right) - 0.5 * relative_speed * (right - left)
                    >= -self.gap_tolerance):
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

    def restart_contract(self) -> dict[str, object]:
        contract = super().restart_contract()
        contract["settings"]["sliding_difference_step"] = self.settings.sliding_difference_step
        if self.settings.beam_chain_tangent == "consistent_dev":
            contract["settings"]["beam_chain_tangent"] = "consistent_dev"
            contract["tangent_geometry_revision"] = "moving_nearest_owner_interface_v1"
        return contract
